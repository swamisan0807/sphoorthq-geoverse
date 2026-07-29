"""Live quantum kernel SVM inference - the actual QML half of the
classical+quantum hybrid (notebooks 05/06), exposed over HTTP. There is no
persisted quantum "model" file: QuantumKernelSVM is trained fresh on a
small balanced pixel sample from a chosen chip, same as the notebooks,
because the quantum kernel is O(n^2) circuit evaluations - keeping n in
the tens (not full-image pixel counts) is what keeps this interactive on
a simulator and feasible at all on real hardware queue times.
"""

import time

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.ai.classic.sen1floods11_dataset import read_label, read_s1
from src.ai.objectives.registry import evaluate
from src.api.routers.auth import get_current_user
from src.fusion.pixel_features import build_feature_cube, cube_to_pixel_table
from src.jobs import engine as jobs_engine
from src.qml import braket_quantum, ibm_quantum
from src.qml.hybrid_classifier import QuantumBackend, QuantumKernelSVM

router = APIRouter(prefix="/api/quantum", tags=["quantum"])

MAX_N_TRAIN = 24
MAX_N_TEST = 12


class QuantumRequest(BaseModel):
    chip_id: str
    backend: str = "ibm"  # "ibm" | "braket"
    n_train: int = 16
    n_test: int = 8
    force_simulation: bool = True


class QuantumResponse(BaseModel):
    chip_id: str
    backend: str
    is_real_hardware: bool
    backend_name: str
    n_train: int
    n_test: int
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


@router.post("/kernel-svm", response_model=QuantumResponse)
def run_quantum_kernel_svm(req: QuantumRequest, username: str = Depends(get_current_user)):
    if req.n_train > MAX_N_TRAIN or req.n_test > MAX_N_TEST:
        raise HTTPException(
            400,
            f"n_train capped at {MAX_N_TRAIN}, n_test at {MAX_N_TEST} to keep this request interactive "
            "- the quantum kernel is O(n^2) circuit evaluations, a real cost of the method",
        )
    if req.backend not in ("ibm", "braket"):
        raise HTTPException(400, f"unknown backend '{req.backend}', expected 'ibm' or 'braket'")

    x_train, y_train, x_test, y_test = _balanced_sample(req.chip_id, req.n_train, req.n_test)

    start = time.time()
    if req.backend == "ibm":
        service = None if req.force_simulation else ibm_quantum.get_ibm_service()
        is_real_hardware = service is not None
        backend_name = (
            "aer_simulator (local fallback)"
            if service is None
            else ibm_quantum.pick_backend(service, min_qubits=x_train.shape[1]).name
        )
        model = QuantumKernelSVM(backend=QuantumBackend.IBM, service=service)
    else:
        device = None if req.force_simulation else braket_quantum.get_braket_device()
        is_real_hardware = device is not None and "LocalSimulator" not in type(device).__name__
        backend_name = "LocalSimulator" if device is None else type(device).__name__
        model = QuantumKernelSVM(backend=QuantumBackend.BRAKET, device=device)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    duration_s = round(time.time() - start, 3)

    metrics = evaluate("flood-segmentation", y_pred, y_test)

    n_train, n_test = len(y_train), len(y_test)
    rounded_metrics = {k: round(v, 4) for k, v in metrics.items()}

    jobs_engine.record_finished_job(
        label=f"quantum kernel SVM: {req.chip_id} ({req.backend})",
        triggered_by=username,
        started_at=start,
        ended_at=start + duration_s,
        status="success",
        kind="quantum",
        extra={
            "chip_id": req.chip_id,
            "backend": req.backend,
            "backend_name": backend_name,
            "is_real_hardware": is_real_hardware,
            "n_train": n_train,
            "n_test": n_test,
            "n_fit_circuits": (n_train * (n_train + 1)) // 2,
            "n_predict_circuits": n_train * n_test,
            "metrics": rounded_metrics,
        },
    )

    return QuantumResponse(
        chip_id=req.chip_id,
        backend=req.backend,
        is_real_hardware=is_real_hardware,
        backend_name=backend_name,
        n_train=n_train,
        n_test=n_test,
        n_fit_circuits=(n_train * (n_train + 1)) // 2,
        n_predict_circuits=n_train * n_test,
        metrics=rounded_metrics,
        duration_s=duration_s,
    )
