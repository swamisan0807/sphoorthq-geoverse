"""Live quantum kernel SVM inference - the actual QML half of the
classical+quantum hybrid (notebooks 05/06), exposed over HTTP. There is no
persisted quantum "model" file: QuantumKernelSVM is trained fresh per
request, because the quantum kernel is O(n^2) circuit evaluations -
keeping n in the tens (not full-image pixel counts) is what keeps this
interactive on a simulator and feasible at all on real hardware queue
times.

Pools its train/test pixels across many chips the same way notebooks
05/06 do (utils.fusion.pixel_features.sample_balanced_pixels_across_chips:
train pixels from the sen1floods11 train split, test pixels from this
project's own event-holdout split), so it's evaluated on the same kind of
whole-dataset sample the classical RF/U-Net registry metrics are - that's
what apps/api/routers/compare.py's Compare page reads for its quantum
series.

(An earlier version of this endpoint let a caller pick one chip to test
the quantum kernel against - useful for poking at a specific scene, but a
one-chip number isn't a fair comparison next to the classical models,
evaluated across dozens of chips. Removed rather than kept alongside this
one, to avoid two similar-looking numbers where only one is actually
comparable.)
"""

import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from apps.api.routers.auth import get_current_user
from utils.ai.classic.sen1floods11_dataset import load_custom_split, load_split
from utils.ai.objectives.registry import evaluate
from utils.fusion.pixel_features import sample_balanced_pixels_across_chips
from utils.jobs import engine as jobs_engine
from utils.qml import ibm_quantum
from utils.qml.hybrid_classifier import QuantumBackend, QuantumKernelSVM

router = APIRouter(prefix="/api/quantum", tags=["quantum"])

MAX_N_TRAIN = 24
MAX_N_TEST = 12

# Same seeds notebooks 05/06 use for their multi-chip pools - keeps a run
# here reproducible/comparable against the notebooks, not just internally
# consistent with itself.
TRAIN_SEED = 1
TEST_SEED = 3


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
    backend: str
    is_real_hardware: bool
    backend_name: str
    n_train: int
    n_test: int
    n_train_chips: int
    n_test_chips: int
    n_fit_circuits: int
    n_predict_circuits: int
    metrics: dict[str, float]
    duration_s: float


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
    QiskitRuntimeService, or None for a local simulator. Raises a clear
    HTTPException on a bad token or missing credentials rather than
    silently falling back, since the caller explicitly asked for real
    hardware."""
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


@router.post("/kernel-svm", response_model=QuantumResponse)
def run_quantum_kernel_svm(req: QuantumRequest, username: str = Depends(get_current_user)):
    """Trains + evaluates the quantum kernel SVM - see module docstring for
    why the sample is pooled across many chips rather than one."""
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
            load_split("train"), req.n_train, seed=TRAIN_SEED
        )
        x_test, y_test, test_chip_ids = sample_balanced_pixels_across_chips(
            load_custom_split("test"), req.n_test, seed=TEST_SEED
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

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
    n_train_chips, n_test_chips = len(set(train_chip_ids)), len(set(test_chip_ids))
    rounded_metrics = {k: round(v, 4) for k, v in metrics.items()}
    n_fit_circuits = (n_train * (n_train + 1)) // 2
    n_predict_circuits = n_train * n_test

    jobs_engine.record_finished_job(
        label=f"quantum kernel SVM ({req.backend}, multi-chip)",
        triggered_by=username,
        started_at=start,
        ended_at=start + duration_s,
        status="success",
        kind="quantum",
        extra={
            # Every job recorded here pools across many chips (see module
            # docstring) - is_benchmark stays True so compare.py's Compare
            # page can keep filtering on it without caring whether a given
            # job predates this field ever existing.
            "is_benchmark": True,
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
