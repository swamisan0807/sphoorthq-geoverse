"""MRF/QAOA segmentation on real MSTAR SAR vehicle-target chips - the same
utils.mrf.energy pipeline and classical/quantum solvers as
utils.mrf.solvers, this project's whole segmentation target.

The unary term comes from the chip's own brightness in decibels, measured
against that chip's background (utils/mrf/mstar_scenes.py). MSTAR's public
release has no per-pixel mask, so when the SARBake overlay for a chip is
available (utils/mrf/sarbake.py) the results are also scored against it;
otherwise they are reported as energy only, with metrics=None."""

import json
import queue
import threading
import time

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from apps.api.routers.auth import get_current_user
from utils.jobs import engine as jobs_engine
from utils.mrf.energy import build_patch_energy, inject_gaussian_noise
from utils.mrf.mstar_scenes import CATALOG as MSTAR_CATALOG
from utils.mrf.mstar_scenes import MSTAR_DIR
from utils.mrf.mstar_scenes import chip_info as mstar_chip_info
from utils.mrf.mstar_scenes import chip_preview_png_base64
from utils.mrf.mstar_scenes import (
    THREE_CLASS_SHADOW_Z,
    THREE_CLASS_TARGET_Z,
    THREE_CLASS_Z_SCALE,
    Z_SPAN,
    target_probability,
    z_to_intensity,
)
from utils.mrf.mstar_scenes import read_calibrated_window as read_mstar_window
from utils.mrf import sarbake
from utils.mrf.multilabel_energy import N_LABELS, build_multilabel_energy, energy_multilabel
from utils.mrf.multilabel_energy import build_qubo_multilabel as build_shadow_qubo
from utils.mrf.multilabel_energy import solve_alpha_expansion as solve_shadow_classical
from utils.mrf.solvers import (
    QuantumSolverResult,
    SolverResult,
    explain_classiq_error,
    run_classical,
    run_qaoa_with_retry,
    run_quantum,
)
from utils.mstar import pattern_recognition
from utils.mstar.reader import read_mstar_chip
from utils.qml import classiq_qaoa

router = APIRouter(prefix="/api/mstar", tags=["mstar"])


class MstarChipInfo(BaseModel):
    chip_id: str
    target_type: str
    target_serial: str
    depression_deg: float
    azimuth_deg: float
    height: int
    width: int
    bright_row: int  # centroid of the real target blob - see mstar_scenes._bright_center
    bright_col: int
    no_object_row: int  # real, verified target-free window - THALES's "images without object" case
    no_object_col: int
    low_snr_row: int  # real, verified low-contrast target-edge window - THALES's "low SNR" case
    low_snr_col: int


class MstarRequest(BaseModel):
    chip_id: str
    row0: int = 0
    col0: int = 0
    size: int = 4
    smoothness: float = 1.0
    contrast_sigma: float = 0.15
    noise_sigma: float = 0.0  # synthetic Gaussian noise added on top of the real magnitude - robustness testing, see utils.mrf.energy.inject_gaussian_noise
    run_quantum: bool = True
    qaoa_reps: int = classiq_qaoa.DEFAULT_REPS
    qaoa_max_iterations: int = classiq_qaoa.DEFAULT_MAX_ITERATIONS
    qaoa_num_shots: int = classiq_qaoa.DEFAULT_NUM_SHOTS
    qaoa_backend: str = "simulator"


class MstarResponse(BaseModel):
    chip_id: str
    target_type: str
    target_serial: str
    chip_height: int
    chip_width: int
    chip_preview_png: str  # base64 PNG of the FULL chip's real magnitude (for display only)
    row0: int
    col0: int
    size: int
    roi_intensity: list[list[float]]  # the actual calibrated intensity the unary term was built from
    reference_mask: list[list[int]] | None = None  # SARBake mask for this window (1 = target), when available
    classical: SolverResult
    quantum: QuantumSolverResult | None
    note: str | None = None


@router.get("/chips", response_model=list[MstarChipInfo])
def list_mstar_chips(username: str = Depends(get_current_user)):
    """Real, locally-cached MSTAR target chips (see utils/mrf/mstar_scenes.py)
    - a fixed catalog, not an arbitrary filesystem path."""
    chips = []
    for chip_id in MSTAR_CATALOG:
        try:
            chips.append(MstarChipInfo(**mstar_chip_info(chip_id)))
        except FileNotFoundError:
            continue  # cataloged but not actually present on this machine - skip rather than 500 the whole list
    return chips


def _validate_and_load(req: MstarRequest):
    if req.size < 1:
        raise HTTPException(400, "size must be >= 1")
    if req.run_quantum and req.size * req.size > classiq_qaoa.MAX_QUBITS:
        raise HTTPException(
            400,
            f"a {req.size}x{req.size} ROI needs {req.size * req.size} qubits, capped at "
            f"{classiq_qaoa.MAX_QUBITS} for QAOA - use a smaller ROI, or set run_quantum=false "
            "for the classical graph-cut baseline only",
        )
    try:
        info = mstar_chip_info(req.chip_id)
        intensity = read_mstar_window(req.chip_id, req.row0, req.col0, req.size)
        preview_png = chip_preview_png_base64(req.chip_id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404 if isinstance(e, FileNotFoundError) else 400, str(e))
    # Real synthetic noise, applied here (not deeper in build_patch_energy) so
    # roi_intensity in the response is exactly what both solvers actually saw,
    # not the clean chip magnitude - see MstarRequest.noise_sigma.
    intensity = inject_gaussian_noise(intensity, req.noise_sigma)
    return info, intensity, preview_png


def _build_response(
    req: MstarRequest, info: dict, intensity, preview_png: str, classical_labels, classical_energy_val, classical_duration, quantum_result
) -> MstarResponse:
    mask = sarbake.load_mask_window(req.chip_id, req.row0, req.col0, req.size)
    scored = mask is not None
    classical_metrics = sarbake.score_binary(classical_labels, mask) if scored else None
    if scored and quantum_result is not None:
        quantum_result = quantum_result.model_copy(
            update={"metrics": sarbake.score_binary(np.array(quantum_result.labels), mask)}
        )
    return MstarResponse(
        chip_id=req.chip_id,
        target_type=info["target_type"],
        target_serial=info["target_serial"],
        chip_height=info["height"],
        chip_width=info["width"],
        chip_preview_png=preview_png,
        row0=req.row0,
        col0=req.col0,
        size=req.size,
        roi_intensity=[[round(float(v), 4) for v in row] for row in intensity.tolist()],
        reference_mask=(mask == sarbake.TARGET).astype(int).tolist() if mask is not None else None,
        classical=SolverResult(
            labels=classical_labels.tolist(),
            energy=round(classical_energy_val, 4),
            metrics=classical_metrics,
            duration_s=round(classical_duration, 4),
        ),
        quantum=quantum_result,
        note=(
            (
                "Both results are scored against the SARBake mask for this chip. The mask is generated from the "
                "vehicle's 3D CAD model rather than drawn by hand, so the scores are a good reference but not perfect "
                "ground truth. Shadow pixels count as background in this two-class view. On small tiles the scores are "
                "noisy and tend to be low, because a few speckled pixels provide little evidence."
                if scored
                else "No per-pixel mask is available for this chip, so only energy is reported."
            )
            + (
                f" Gaussian noise (sigma={req.noise_sigma}) was added to the real signal as a robustness test, so both "
                "solvers processed the noisy version rather than the clean chip."
                if req.noise_sigma > 0
                else ""
            )
        ),
    )


def _record_job(req: MstarRequest, info: dict, username: str, job_started: float, classical_energy_val: float, quantum_result) -> None:
    jobs_engine.record_finished_job(
        label=f"MRF graph-cut{' + QAOA' if req.run_quantum else ''} (MSTAR: {req.chip_id}, {req.size}x{req.size})",
        triggered_by=username,
        started_at=job_started,
        ended_at=time.time(),
        status="success",
        kind="quantum",
        extra={
            "is_benchmark": False,
            "source": "mstar",
            "chip_id": req.chip_id,
            "target_type": info["target_type"],
            "roi": {"row0": req.row0, "col0": req.col0, "size": req.size},
            "ran_quantum": req.run_quantum,
            "classical_energy": round(classical_energy_val, 4),
            "quantum_energy": quantum_result.energy if quantum_result else None,
            "qaoa_backend": req.qaoa_backend if req.run_quantum else None,
        },
    )


@router.post("", response_model=MstarResponse)
def run_mstar(req: MstarRequest, username: str = Depends(get_current_user)):
    info, intensity, preview_png = _validate_and_load(req)
    problem = build_patch_energy(
        target_probability(intensity), intensity, smoothness=req.smoothness, contrast_sigma=req.contrast_sigma
    )
    job_started = time.time()

    classical_labels, classical_energy_val, classical_duration = run_classical(problem)
    _, quantum_result = run_quantum(
        problem, req.size, req.run_quantum, req.qaoa_reps, req.qaoa_max_iterations, req.qaoa_num_shots, req.qaoa_backend
    )

    _record_job(req, info, username, job_started, classical_energy_val, quantum_result)
    return _build_response(req, info, intensity, preview_png, classical_labels, classical_energy_val, classical_duration, quantum_result)


@router.post("/stream")
def run_mstar_stream(req: MstarRequest, username: str = Depends(get_current_user)):
    """Same pipeline as POST /api/mstar, but streamed as newline-delimited
    JSON events instead of one blocking response - each QAOA iteration is a
    genuine ~4-5s round trip to Classiq's platform (not local computation to
    poll), so this surfaces real progress (iteration index, current
    shot-weighted mean energy, best energy seen so far) as it actually
    happens, not a fake "loading..." spinner over a 60-90s wait."""
    info, intensity, preview_png = _validate_and_load(req)
    problem = build_patch_energy(
        target_probability(intensity), intensity, smoothness=req.smoothness, contrast_sigma=req.contrast_sigma
    )

    def event_stream():
        job_started = time.time()
        classical_labels, classical_energy_val, classical_duration = run_classical(problem)
        yield json.dumps(
            {"type": "classical_done", "energy": round(classical_energy_val, 4), "duration_s": round(classical_duration, 4)}
        ) + "\n"

        quantum_result = None
        if req.run_quantum:
            progress_q: "queue.Queue" = queue.Queue()
            outcome: dict = {}

            def on_iteration(iteration: int, mean_energy: float, best_energy: float) -> None:
                progress_q.put(
                    {
                        "type": "progress",
                        "iteration": iteration,
                        "max_iterations": req.qaoa_max_iterations,
                        "mean_energy": round(mean_energy, 4),
                        "best_energy": round(best_energy, 4),
                    }
                )

            def worker() -> None:
                try:
                    _, qr = run_quantum(
                        problem,
                        req.size,
                        True,
                        req.qaoa_reps,
                        req.qaoa_max_iterations,
                        req.qaoa_num_shots,
                        req.qaoa_backend,
                        on_iteration=on_iteration,
                    )
                    outcome["result"] = qr
                except HTTPException as e:
                    outcome["error"] = str(e.detail)
                except Exception as e:
                    outcome["error"] = str(e)
                finally:
                    progress_q.put(None)  # sentinel: worker is done

            t = threading.Thread(target=worker, daemon=True)
            t.start()
            while True:
                item = progress_q.get()
                if item is None:
                    break
                yield json.dumps(item) + "\n"
            t.join()

            if "error" in outcome:
                yield json.dumps({"type": "error", "detail": outcome["error"]}) + "\n"
                return
            quantum_result = outcome["result"]

        _record_job(req, info, username, job_started, classical_energy_val, quantum_result)
        response = _build_response(
            req, info, intensity, preview_png, classical_labels, classical_energy_val, classical_duration, quantum_result
        )
        yield json.dumps({"type": "done", "result": json.loads(response.model_dump_json())}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


# --- 3-class (background/target/shadow) segmentation --------------------
# THALES's own "beyond binary" extension: "a third zone, which is the
# shadow of the vehicle, can be labelled... denoted as class 2." See
# utils/mrf/multilabel_energy.py's module docstring for what's genuinely
# different here (approximate alpha-expansion classical solver instead of
# exact graph-cut; one-hot QUBO tripling the qubit cost per pixel).

MAX_SHADOW_QAOA_SIZE = 2  # size*size*N_LABELS(=3) <= MAX_QUBITS=25 -> size <= 2 (3x3=27 qubits exceeds the real ceiling)


class MstarShadowRequest(BaseModel):
    chip_id: str
    row0: int = 0
    col0: int = 0
    size: int = 4
    smoothness: float = 1.0
    contrast_sigma: float = 0.15
    run_quantum: bool = False  # off by default - most sizes exceed the real qubit ceiling for Q=3
    qaoa_reps: int = classiq_qaoa.DEFAULT_REPS
    qaoa_max_iterations: int = classiq_qaoa.DEFAULT_MAX_ITERATIONS
    qaoa_num_shots: int = classiq_qaoa.DEFAULT_NUM_SHOTS
    qaoa_backend: str = "simulator"


class ShadowSolverResult(BaseModel):
    labels: list[list[int]]  # 0=background, 1=target, 2=shadow
    energy: float
    duration_s: float
    metrics: dict[str, float] | None = None  # scored against the SARBake mask when one is available


class ShadowQuantumSolverResult(ShadowSolverResult):
    n_qubits: int
    backend: str
    circuit_width: int
    circuit_depth: int | None
    cost_history: list[float]


class MstarShadowResponse(BaseModel):
    chip_id: str
    target_type: str
    target_serial: str
    chip_height: int
    chip_width: int
    chip_preview_png: str
    row0: int
    col0: int
    size: int
    background_mu: float  # the background reference level, in calibrated intensity units
    reference_mask: list[list[int]] | None = None  # SARBake mask for this window (0/1/2), when available
    classical: ShadowSolverResult
    quantum: ShadowQuantumSolverResult | None
    note: str | None = None


@router.post("/shadow", response_model=MstarShadowResponse)
def run_mstar_shadow(req: MstarShadowRequest, username: str = Depends(get_current_user)):
    if req.size < 1:
        raise HTTPException(400, "size must be >= 1")
    n_qubits = req.size * req.size * N_LABELS
    if req.run_quantum and n_qubits > classiq_qaoa.MAX_QUBITS:
        raise HTTPException(
            400,
            f"a {req.size}x{req.size} 3-class ROI needs {n_qubits} qubits ({N_LABELS} per pixel, one-hot), capped "
            f"at {classiq_qaoa.MAX_QUBITS} for QAOA - use size<={MAX_SHADOW_QAOA_SIZE}, or set run_quantum=false "
            "for the classical (alpha-expansion) baseline only",
        )

    try:
        info = mstar_chip_info(req.chip_id)
        intensity = read_mstar_window(req.chip_id, req.row0, req.col0, req.size)
        preview_png = chip_preview_png_base64(req.chip_id)
        bg_mu = float(z_to_intensity(0.0))
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404 if isinstance(e, FileNotFoundError) else 400, str(e))

    problem = build_multilabel_energy(
        intensity,
        bg_mu,
        smoothness=req.smoothness,
        contrast_sigma=req.contrast_sigma,
        target_mu=float(z_to_intensity(THREE_CLASS_TARGET_Z)),
        shadow_mu=float(z_to_intensity(THREE_CLASS_SHADOW_Z)),
        unary_scale=(Z_SPAN / THREE_CLASS_Z_SCALE) ** 2,
    )
    job_started = time.time()

    start = time.time()
    classical_labels = solve_shadow_classical(problem)
    classical_duration = time.time() - start
    classical_energy = energy_multilabel(problem, classical_labels)

    quantum_result = None
    if req.run_quantum:
        qubo, const = build_shadow_qubo(problem)
        start = time.time()
        try:
            qaoa = run_qaoa_with_retry(
                qubo,
                const,
                n_qubits,
                reps=req.qaoa_reps,
                max_iterations=req.qaoa_max_iterations,
                num_shots=req.qaoa_num_shots,
                backend=req.qaoa_backend,
            )
        except Exception as e:
            raise HTTPException(400, explain_classiq_error(e))
        duration = time.time() - start

        from utils.mrf.multilabel_energy import one_hot_to_labels

        q_labels = one_hot_to_labels(qaoa.labels, req.size, req.size)
        quantum_result = ShadowQuantumSolverResult(
            labels=q_labels.tolist(),
            energy=round(float(energy_multilabel(problem, q_labels)), 4),
            duration_s=round(duration, 3),
            n_qubits=qaoa.n_qubits,
            backend=qaoa.backend,
            circuit_width=qaoa.circuit_width,
            circuit_depth=qaoa.circuit_depth,
            cost_history=[round(c, 4) for c in qaoa.optimizer_cost_history],
        )

    jobs_engine.record_finished_job(
        label=f"3-class MRF alpha-expansion{' + QAOA' if req.run_quantum else ''} (MSTAR: {req.chip_id}, {req.size}x{req.size})",
        triggered_by=username,
        started_at=job_started,
        ended_at=time.time(),
        status="success",
        kind="quantum",
        extra={
            "is_benchmark": False,
            "source": "mstar_shadow",
            "chip_id": req.chip_id,
            "roi": {"row0": req.row0, "col0": req.col0, "size": req.size},
            "ran_quantum": req.run_quantum,
            "classical_energy": round(float(classical_energy), 4),
            "quantum_energy": quantum_result.energy if quantum_result else None,
        },
    )

    mask = sarbake.load_mask_window(req.chip_id, req.row0, req.col0, req.size)
    classical_metrics = sarbake.score_three_class(classical_labels, mask) if mask is not None else None
    if mask is not None and quantum_result is not None:
        quantum_result = quantum_result.model_copy(
            update={"metrics": sarbake.score_three_class(np.array(quantum_result.labels), mask)}
        )
    return MstarShadowResponse(
        chip_id=req.chip_id,
        target_type=info["target_type"],
        target_serial=info["target_serial"],
        chip_height=info["height"],
        chip_width=info["width"],
        chip_preview_png=preview_png,
        row0=req.row0,
        col0=req.col0,
        size=req.size,
        background_mu=round(bg_mu, 4),
        reference_mask=mask.astype(int).tolist() if mask is not None else None,
        classical=ShadowSolverResult(
            labels=classical_labels.tolist(),
            energy=round(float(classical_energy), 4),
            duration_s=round(classical_duration, 4),
            metrics=classical_metrics,
        ),
        quantum=quantum_result,
        note=(
            "This adds a third class, the vehicle's shadow, as suggested in the THALES brief. The classical result "
            "comes from alpha-expansion, which is approximate rather than exact as in the two-class version. Shadow is "
            "modelled as a dark, low-return area, and the result is scored against the SARBake mask when one is "
            "available (a reference derived from the vehicle's CAD model, not a hand-drawn answer key)."
        ),
    )


# --- Pattern detection (target-type recognition) -------------------------
# THALES's own document lists "Pattern detection" with no definition at
# all. This project's disclosed, literature-grounded interpretation (see
# utils/mstar/pattern_recognition.py's module docstring): recognize WHICH
# real vehicle type (BMP2/BTR70/T72) a chip shows, via a real
# correlation-filter match against class templates built from the actual
# MSTAR train split - not a fabricated task, and not a fabricated result:
# accuracy is measured on the real MSTAR test split, not asserted.


class PatternClassInfo(BaseModel):
    class_name: str
    n_train_samples: int


class PatternEvaluationResponse(BaseModel):
    accuracy: float
    n_test_samples: int
    confusion: dict[str, dict[str, int]]
    classes: list[PatternClassInfo]
    note: str


@router.get("/pattern/evaluation", response_model=PatternEvaluationResponse)
def get_pattern_evaluation(username: str = Depends(get_current_user)):
    """Real accuracy of the target-type recognizer, measured on the real
    MSTAR test split (not the 32-chip curated UI catalog) - cached after
    the first call (utils/mstar/pattern_recognition.py), since it's a
    ~1300-chip evaluation (~20-40s cold)."""
    result = pattern_recognition.evaluate()
    return PatternEvaluationResponse(
        accuracy=result.accuracy,
        n_test_samples=result.n_test_samples,
        confusion=result.confusion,
        classes=[
            PatternClassInfo(class_name=c, n_train_samples=result.n_train_samples[c])
            for c in pattern_recognition.TARGET_CLASSES
        ],
        note=(
            f"We checked {result.n_test_samples} real MSTAR test chips against templates built from "
            f"{sum(result.n_train_samples.values())} real training chips, across {len(pattern_recognition.TARGET_CLASSES)} "
            "vehicle types (BMP2, BTR70, T72). This is a deliberately simple, easy-to-follow method, not a deep "
            "model, so modern classifiers score much higher. The number is what we actually measured, not one "
            "we polished."
        ),
    )


# --- Synthetic validation (real Accuracy/IoU/Dice) ------------------------
# Real MSTAR chips have no per-pixel ground truth, so Accuracy/IoU/Dice
# can't be honestly computed against them anywhere in this app (see every
# other endpoint's `note`). This is the one place those metrics exist:
# THALES's own worked example (quoted directly from their brief) plus a few
# small generated cases with known ground truth by construction
# (utils/mrf/synthetic_validation.py) - never real MSTAR.


class SyntheticValidationRequest(BaseModel):
    run_quantum: bool = False  # real QAOA is slow (~60-90s/case) and costs real Classiq API calls - opt in
    qaoa_reps: int = classiq_qaoa.DEFAULT_REPS
    qaoa_max_iterations: int = classiq_qaoa.DEFAULT_MAX_ITERATIONS
    qaoa_num_shots: int = classiq_qaoa.DEFAULT_NUM_SHOTS
    qaoa_backend: str = "simulator"
    only_case: int | None = None  # run just this case, so the page can fetch one at a time instead of one long request


class ValidationCaseResult(BaseModel):
    name: str
    source: str
    size: int
    intensity: list[list[float]]
    true_labels: list[list[int]]
    classical: SolverResult
    quantum: QuantumSolverResult | None


class SyntheticValidationResponse(BaseModel):
    cases: list[ValidationCaseResult]
    note: str
    total_cases: int


@router.post("/synthetic-validation", response_model=SyntheticValidationResponse)
def run_synthetic_validation(req: SyntheticValidationRequest, username: str = Depends(get_current_user)):
    from utils.mrf.metrics import compute_metrics
    from utils.mrf.synthetic_validation import build_validation_cases, problem_for_case

    all_cases = build_validation_cases()
    if req.only_case is not None and not 0 <= req.only_case < len(all_cases):
        raise HTTPException(400, f"only_case must be between 0 and {len(all_cases) - 1}")

    results = []
    for index, case in enumerate(all_cases):
        if req.only_case is not None and index != req.only_case:
            continue
        problem = problem_for_case(case)
        size = case.intensity.shape[0]

        c_labels, c_energy, c_dur = run_classical(problem)
        c_metrics = compute_metrics(c_labels, case.true_labels)
        classical_result = SolverResult(
            labels=c_labels.tolist(), energy=round(c_energy, 4), metrics=c_metrics, duration_s=round(c_dur, 4)
        )

        q_labels, q_result = run_quantum(
            problem,
            size=size,
            run_quantum=req.run_quantum,
            reps=req.qaoa_reps,
            max_iterations=req.qaoa_max_iterations,
            num_shots=req.qaoa_num_shots,
            backend=req.qaoa_backend,
        )
        if q_result is not None and q_labels is not None:
            q_result = q_result.model_copy(update={"metrics": compute_metrics(q_labels, case.true_labels)})

        results.append(
            ValidationCaseResult(
                name=case.name,
                source=case.source,
                size=size,
                intensity=[[round(float(v), 4) for v in row] for row in case.intensity.tolist()],
                true_labels=case.true_labels.tolist(),
                classical=classical_result,
                quantum=q_result,
            )
        )

    return SyntheticValidationResponse(
        cases=results,
        note=(
            "These scores are exact because every case has a known correct answer by construction: THALES's own "
            "worked example plus a few small cases generated for this project. None of the cases is an MSTAR chip. "
            "Real chips are scored against the SARBake masks instead, in the Metrics tab and on the About page."
        ),
        total_cases=len(all_cases),
    )


class PatternRequest(BaseModel):
    chip_id: str


class PatternResponse(BaseModel):
    chip_id: str
    actual_class: str
    predicted_class: str
    correct: bool
    scores: dict[str, float]  # real correlation coefficient per class


@router.post("/pattern", response_model=PatternResponse)
def run_pattern_detection(req: PatternRequest, username: str = Depends(get_current_user)):
    if req.chip_id not in MSTAR_CATALOG:
        raise HTTPException(400, f"unknown chip '{req.chip_id}', available: {list(MSTAR_CATALOG)}")

    chip_path = MSTAR_DIR / MSTAR_CATALOG[req.chip_id]
    if not chip_path.exists():
        raise HTTPException(404, f"chip file not found locally: {chip_path}")
    chip = read_mstar_chip(chip_path)

    if chip.target_type not in pattern_recognition.TARGET_CLASSES:
        raise HTTPException(
            400,
            f"'{req.chip_id}' is a {chip.target_type} chip - pattern detection only recognizes real vehicle "
            f"classes ({', '.join(pattern_recognition.TARGET_CLASSES)}), not the SLICY calibration target",
        )

    templates = pattern_recognition.build_templates()
    patch = pattern_recognition.extract_target_patch(chip.magnitude, templates.patch_size)
    if patch is None:
        raise HTTPException(400, f"couldn't locate a real target blob in '{req.chip_id}' to classify")

    predicted, scores = pattern_recognition.classify_patch(patch, templates)
    return PatternResponse(
        chip_id=req.chip_id,
        actual_class=chip.target_type,
        predicted_class=predicted,
        correct=predicted == chip.target_type,
        scores={k: round(v, 4) for k, v in scores.items()},
    )
