"""Live quantum kernel SVM inference - the actual QML half of the
classical+quantum hybrid (notebooks 05/06), exposed over HTTP. There is no
persisted quantum "model" file: QuantumKernelSVM is trained fresh per
request, because the quantum kernel is O(n^2) circuit evaluations -
keeping n in the tens (not full-image pixel counts) is what keeps this
interactive on a simulator and feasible at all on real hardware queue
times.

Two request shapes, for two different purposes:
  - /kernel-svm: single-chip explorer - "try the quantum kernel on THIS
    chip I picked". Fine for interactive exploration, but a one-chip
    number isn't comparable to the classical RF/U-Net registry metrics
    (those are evaluated across dozens of chips), so these runs are never
    surfaced on the Compare page (apps/api/routers/compare.py) - see
    is_benchmark below.
  - /kernel-svm/benchmark: the fair-comparison run - pools its train/test
    pixels across many chips the same way notebooks 05/06 do
    (utils.fusion.pixel_features.sample_balanced_pixels_across_chips: train
    pixels from the sen1floods11 train split, test pixels from this
    project's own event-holdout split), so it's evaluated on the same kind
    of whole-dataset sample the classical models are. Jobs from this path
    are tagged is_benchmark=True; compare.py only ever reads those for its
    quantum series.
"""

import time

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from apps.api.routers.auth import get_current_user
from utils.ai.classic.sen1floods11_dataset import load_custom_split, load_split, read_label, read_s1
from utils.ai.objectives.registry import evaluate
from utils.fusion.pixel_features import build_feature_cube, cube_to_pixel_table, sample_balanced_pixels_across_chips
from utils.jobs import engine as jobs_engine
from utils.qml import ibm_quantum
from utils.qml.hybrid_classifier import QuantumBackend, QuantumKernelSVM

router = APIRouter(prefix="/api/quantum", tags=["quantum"])

MAX_N_TRAIN = 24
MAX_N_TEST = 12

# Same seeds notebooks 05/06 use for their multi-chip pools - keeps a
# benchmark run here reproducible/comparable against the notebooks, not
# just internally consistent with itself.
BENCHMARK_TRAIN_SEED = 1
BENCHMARK_TEST_SEED = 3


class _IbmCredentials(BaseModel):
    backend: str = "ibm"  # "ibm"
    force_simulation: bool = True
    # Only used when force_simulation is False - lets a caller connect with
    # their own IBM Quantum account for this one run instead of requiring
    # IBM_QUANTUM_TOKEN/IBM_QUANTUM_INSTANCE to already be set on the server
    # (same channel/token/instance triple QiskitRuntimeService itself takes).
    # Never persisted: not written to disk, not included in job history.
    ibm_token: str | None = None
    ibm_instance: str | None = None
    ibm_channel: str | None = None


class QuantumRequest(_IbmCredentials):
    chip_id: str
    n_train: int = 16
    n_test: int = 8


class BenchmarkRequest(_IbmCredentials):
    n_train: int = 20  # matches notebooks 05/06's N_TRAIN
    n_test: int = 10  # matches notebooks 05/06's N_TEST


class ConnectRequest(BaseModel):
    ibm_token: str | None = None
    ibm_instance: str | None = None
    ibm_channel: str | None = None


class ConnectResponse(BaseModel):
    connected: bool
    channel: str
    n_backends: int
    backend_name: str


class QuantumResponse(BaseModel):
    chip_id: str | None  # None for a benchmark run (pooled across many chips) - see is_benchmark
    is_benchmark: bool
    backend: str
    is_real_hardware: bool
    backend_name: str
    n_train: int
    n_test: int
    n_train_chips: int | None = None  # only set for a benchmark run
    n_test_chips: int | None = None
    n_fit_circuits: int
    n_predict_circuits: int
    metrics: dict[str, float]
    duration_s: float


def _balanced_sample(chip_id: str, n_train: int, n_test: int, seed: int = 7):
    try:
        s1 = read_s1(chip_id)
        label = read_label(chip_id)
    except Exception:
        raise HTTPException(404, f"chip '{chip_id}' not found")

    cube = build_feature_cube(s1)
    x_all, y_all = cube_to_pixel_table(cube, label, raw_s1=s1)

    rng = np.random.default_rng(seed)
    water_idx = np.where(y_all == 1)[0]
    land_idx = np.where(y_all == 0)[0]
    n_each = (n_train + n_test) // 2
    if len(water_idx) < n_each or len(land_idx) < n_each:
        raise HTTPException(
            400,
            f"chip '{chip_id}' doesn't have {n_each} pixels of both classes "
            f"(has {len(water_idx)} water, {len(land_idx)} land) - pick a bigger n_train/n_test or a different chip",
        )
    chosen = np.concatenate(
        [
            rng.choice(water_idx, size=n_each, replace=False),
            rng.choice(land_idx, size=n_each, replace=False),
        ]
    )
    rng.shuffle(chosen)
    x_sample, y_sample = x_all[chosen], y_all[chosen]
    return (
        x_sample[:n_train],
        y_sample[:n_train],
        x_sample[n_train : n_train + n_test],
        y_sample[n_train : n_train + n_test],
    )


@router.post("/connect", response_model=ConnectResponse)
def connect_ibm_quantum(req: ConnectRequest, username: str = Depends(get_current_user)):
    """Fast pre-flight check for the Real Hardware picker: authenticates
    and lists the account's real backends, but never queries per-backend
    queue status (that's the slow part, done only once a job actually
    submits - see pick_backend) so this comes back in a few seconds even
    though it's a genuine network round trip, not a canned response."""
    try:
        service = ibm_quantum.get_ibm_service(token=req.ibm_token, channel=req.ibm_channel, instance=req.ibm_instance)
    except Exception as e:
        raise HTTPException(400, f"IBM Quantum connection failed: {e}")
    if service is None:
        raise HTTPException(
            400,
            "no IBM Quantum credentials available - enter an API token (and instance CRN), "
            "or set IBM_QUANTUM_TOKEN / IBM_QUANTUM_INSTANCE on the server",
        )
    try:
        backends = ibm_quantum.list_backends(service)
    except Exception as e:
        raise HTTPException(400, f"connected, but couldn't list backends: {e}")

    real_backends = [b for b in backends if not b.simulator]
    return ConnectResponse(
        connected=True,
        channel=req.ibm_channel or "ibm_cloud",
        n_backends=len(real_backends),
        backend_name=real_backends[0].name if real_backends else "(no real backends visible on this account)",
    )


def _connect_service(req: _IbmCredentials):
    """Resolves force_simulation/ibm_token/instance/channel into a live
    QiskitRuntimeService, or None for a local simulator - shared by both
    /kernel-svm and /kernel-svm/benchmark. Raises a clear HTTPException on
    a bad token or missing credentials rather than silently falling back,
    since the caller explicitly asked for real hardware."""
    if req.force_simulation:
        return None
    try:
        service = ibm_quantum.get_ibm_service(token=req.ibm_token, channel=req.ibm_channel, instance=req.ibm_instance)
    except Exception as e:
        # A bad/expired token, wrong instance CRN, etc. raise here - surface
        # the real reason instead of a bare 500.
        raise HTTPException(400, f"IBM Quantum connection failed: {e}")
    if service is None:
        raise HTTPException(
            400,
            "real hardware was requested but no IBM Quantum credentials are available - "
            "enter an API token (and instance CRN) in the Real Hardware fields, or set "
            "IBM_QUANTUM_TOKEN / IBM_QUANTUM_INSTANCE on the server",
        )
    return service


def _run_kernel_svm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    req: _IbmCredentials,
    username: str,
    label: str,
    is_benchmark: bool,
    chip_id: str | None,
    n_train_chips: int | None = None,
    n_test_chips: int | None = None,
) -> QuantumResponse:
    """Shared fit/predict/record/respond path for both /kernel-svm (one
    chip) and /kernel-svm/benchmark (pooled across many chips) - the only
    difference between the two endpoints is where x_train/y_train/x_test/
    y_test came from."""
    start = time.time()
    service = _connect_service(req)
    is_real_hardware = service is not None
    if service is None:
        backend_name = "aer_simulator"
    else:
        try:
            backend_name = ibm_quantum.pick_backend(service, min_qubits=x_train.shape[1]).name
        except Exception as e:
            raise HTTPException(400, f"IBM Quantum connected, but couldn't pick a backend: {e}")
    model = QuantumKernelSVM(backend=QuantumBackend.IBM, service=service)

    try:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
    except Exception as e:
        # Most likely on the real-hardware path: the account can't run the
        # job right now (usage limit, backend down, etc.) - job.result() is
        # timeout-bounded (see ibm_quantum.JOB_RESULT_TIMEOUT_S) so this
        # surfaces the real reason within a bounded time instead of hanging.
        raise HTTPException(400, f"quantum circuit execution failed: {e}")
    duration_s = round(time.time() - start, 3)

    metrics = evaluate("flood-segmentation", y_pred, y_test)

    n_train, n_test = len(y_train), len(y_test)
    rounded_metrics = {k: round(v, 4) for k, v in metrics.items()}
    n_fit_circuits = (n_train * (n_train + 1)) // 2
    n_predict_circuits = n_train * n_test

    jobs_engine.record_finished_job(
        label=label,
        triggered_by=username,
        started_at=start,
        ended_at=start + duration_s,
        status="success",
        kind="quantum",
        extra={
            "chip_id": chip_id,
            "is_benchmark": is_benchmark,
            "n_train_chips": n_train_chips,
            "n_test_chips": n_test_chips,
            "backend": req.backend,
            "backend_name": backend_name,
            "is_real_hardware": is_real_hardware,
            "n_train": n_train,
            "n_test": n_test,
            "n_fit_circuits": n_fit_circuits,
            "n_predict_circuits": n_predict_circuits,
            "metrics": rounded_metrics,
        },
    )

    return QuantumResponse(
        chip_id=chip_id,
        is_benchmark=is_benchmark,
        backend=req.backend,
        is_real_hardware=is_real_hardware,
        backend_name=backend_name,
        n_train=n_train,
        n_test=n_test,
        n_train_chips=n_train_chips,
        n_test_chips=n_test_chips,
        n_fit_circuits=n_fit_circuits,
        n_predict_circuits=n_predict_circuits,
        metrics=rounded_metrics,
        duration_s=duration_s,
    )


@router.post("/kernel-svm", response_model=QuantumResponse)
def run_quantum_kernel_svm(req: QuantumRequest, username: str = Depends(get_current_user)):
    """Single-chip explorer - see module docstring. Not used by the
    Compare page (is_benchmark=False); use /kernel-svm/benchmark for a
    number that's fairly comparable to the classical models."""
    if req.n_train > MAX_N_TRAIN or req.n_test > MAX_N_TEST:
        raise HTTPException(
            400,
            f"n_train capped at {MAX_N_TRAIN}, n_test at {MAX_N_TEST} to keep this request interactive "
            "- the quantum kernel is O(n^2) circuit evaluations, a real cost of the method",
        )
    if req.backend != "ibm":
        raise HTTPException(400, f"unknown backend '{req.backend}', expected 'ibm'")

    x_train, y_train, x_test, y_test = _balanced_sample(req.chip_id, req.n_train, req.n_test)

    return _run_kernel_svm(
        x_train,
        y_train,
        x_test,
        y_test,
        req,
        username,
        label=f"quantum kernel SVM: {req.chip_id} ({req.backend})",
        is_benchmark=False,
        chip_id=req.chip_id,
    )


@router.post("/kernel-svm/benchmark", response_model=QuantumResponse)
def run_quantum_kernel_svm_benchmark(req: BenchmarkRequest, username: str = Depends(get_current_user)):
    """The fair-comparison run - see module docstring. Pools train pixels
    across the sen1floods11 train split and test pixels across this
    project's own event-holdout split (same sampling function, same seeds,
    as notebooks 05/06's fair-comparison fix), instead of a single chosen
    chip, so this run is evaluated on the same kind of whole-dataset sample
    the classical RF/U-Net registry metrics are. compare.py only reads
    is_benchmark=True jobs for its quantum series, so an ad hoc
    /kernel-svm exploration never shows up there as if it were this
    number."""
    if req.n_train > MAX_N_TRAIN or req.n_test > MAX_N_TEST:
        raise HTTPException(
            400,
            f"n_train capped at {MAX_N_TRAIN}, n_test at {MAX_N_TEST} to keep this request interactive "
            "- the quantum kernel is O(n^2) circuit evaluations, a real cost of the method",
        )
    if req.backend != "ibm":
        raise HTTPException(400, f"unknown backend '{req.backend}', expected 'ibm'")

    try:
        x_train, y_train, train_chip_ids = sample_balanced_pixels_across_chips(
            load_split("train"), req.n_train, seed=BENCHMARK_TRAIN_SEED
        )
        x_test, y_test, test_chip_ids = sample_balanced_pixels_across_chips(
            load_custom_split("test"), req.n_test, seed=BENCHMARK_TEST_SEED
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    return _run_kernel_svm(
        x_train,
        y_train,
        x_test,
        y_test,
        req,
        username,
        label=f"quantum kernel SVM benchmark ({req.backend}, multi-chip)",
        is_benchmark=True,
        chip_id=None,
        n_train_chips=len(set(train_chip_ids)),
        n_test_chips=len(set(test_chip_ids)),
    )
