"""Shared classical/quantum solver plumbing for utils.mrf.energy.PatchEnergy
problems - the same two ways to minimize the SAME energy (see energy.py's
module docstring) that apps/api/routers/mstar.py drives, factored out so
more than one router can share it without importing from another router
module."""

import time

import httpx
import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel

from utils.mrf.energy import PatchEnergy, build_qubo, energy, solve_graph_cut
from utils.qml import classiq_qaoa

# Dropped connections and gateway errors (502/503/504) from Classiq's cloud are brief; a later attempt normally works.
_TRANSIENT_ERRORS = (httpx.TransportError, ConnectionError, TimeoutError)
_TRANSIENT_STATUS_CODES = (502, 503, 504)
_RETRY_PAUSES_S = (5, 15, 30)  # waits between attempts, so there are len() + 1 attempts in all


def _is_transient(e: Exception) -> bool:
    return isinstance(e, _TRANSIENT_ERRORS) or getattr(e, "status_code", None) in _TRANSIENT_STATUS_CODES


class SolverResult(BaseModel):
    labels: list[list[int]]
    energy: float
    metrics: dict[str, float] | None
    duration_s: float


class QuantumSolverResult(SolverResult):
    n_qubits: int
    reps: int
    backend: str
    circuit_width: int
    circuit_depth: int | None
    num_shots: int
    optimized_params: list[float]
    cost_history: list[float]  # shot-weighted mean QUBO energy per COBYLA iteration - the "energy convergence" trace


def explain_classiq_error(e: Exception) -> str:
    """Turns a raw Classiq failure into something a person can act on. A
    login that has lapsed (the multi-factor check, or an expired token) is
    by far the most common cause, so it gets its own plain message; anything
    else keeps the real reason attached rather than being hidden."""
    text = str(e).strip()
    if "mfa_required" in text or "Expired token" in text:
        return (
            "Classiq wants you to log in again (it asks for a multi-factor check every so often). "
            "Follow the Classiq login step in the README, then run this again. "
            "Everything classical keeps working in the meantime."
        )
    if _is_transient(e):
        detail = f"HTTP {e.status_code}" if getattr(e, "status_code", None) else type(e).__name__
        return (
            f"Classiq's servers did not respond properly ({detail}) and did not recover after "
            f"{len(_RETRY_PAUSES_S) + 1} attempts. This is usually a temporary problem on their side or with the "
            "internet connection: wait a minute and run this again. Everything classical keeps working in the meantime."
        )
    return f"The quantum run on Classiq failed: {text or type(e).__name__}"


def run_qaoa_with_retry(*args, **kwargs):
    """classiq_qaoa.run_qaoa(), retried when Classiq's cloud drops the connection or answers 502/503/504."""
    for pause in (*_RETRY_PAUSES_S, None):
        try:
            return classiq_qaoa.run_qaoa(*args, **kwargs)
        except Exception as e:
            if pause is None or not _is_transient(e):
                raise
            time.sleep(pause)


def run_classical(problem: PatchEnergy) -> tuple[np.ndarray, float, float]:
    start = time.time()
    labels = solve_graph_cut(problem)
    duration = time.time() - start
    return labels, energy(problem, labels), duration


def run_quantum(
    problem: PatchEnergy,
    size: int,
    run_quantum: bool,
    reps: int,
    max_iterations: int,
    num_shots: int,
    backend: str,
    on_iteration=None,
) -> tuple[np.ndarray | None, QuantumSolverResult | None]:
    """Returns (labels_or_None, QuantumSolverResult_or_None) - labels kept
    separate so callers can score metrics against their own ground truth,
    when they have one. on_iteration, if given, is forwarded to
    classiq_qaoa.run_qaoa() for real per-iteration progress (see that
    function's docstring)."""
    if not run_quantum:
        return None, None
    if size * size > classiq_qaoa.MAX_QUBITS:
        raise HTTPException(
            400,
            f"a {size}x{size} ROI needs {size * size} qubits, capped at {classiq_qaoa.MAX_QUBITS} for QAOA - "
            "use a smaller size, or set run_quantum=false for the classical baseline only",
        )
    qubo, const = build_qubo(problem)
    start = time.time()
    try:
        qaoa = run_qaoa_with_retry(
            qubo,
            const,
            problem.n,
            reps=reps,
            max_iterations=max_iterations,
            num_shots=num_shots,
            backend=backend,
            on_iteration=on_iteration,
        )
    except Exception as e:
        # No local-simulator fallback for Classiq (see utils/qml/classiq_qaoa.py's
        # module docstring) - surface the real reason (not authenticated, synthesis
        # failed, backend unavailable, ...) rather than a bare 500.
        raise HTTPException(400, explain_classiq_error(e))
    duration = time.time() - start

    labels = qaoa.labels.reshape(size, size)
    result = QuantumSolverResult(
        labels=labels.tolist(),
        energy=round(qaoa.best_energy, 4),
        metrics=None,  # caller fills this in if it has ground truth
        duration_s=round(duration, 3),
        n_qubits=qaoa.n_qubits,
        reps=qaoa.reps,
        backend=qaoa.backend,
        circuit_width=qaoa.circuit_width,
        circuit_depth=qaoa.circuit_depth,
        num_shots=qaoa.num_shots,
        optimized_params=[round(p, 5) for p in qaoa.optimized_params],
        cost_history=[round(c, 4) for c in qaoa.optimizer_cost_history],
    )
    return labels, result
