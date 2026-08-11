"""Real IBM Quantum integration via Qiskit Runtime.

Connects to IBM Quantum's cloud API (QiskitRuntimeService). Two ways to
authenticate:
  1. A saved account - `QiskitRuntimeService.save_account(...)` once,
     persisted to this machine's local Qiskit config (~/.qiskit/), then
     `get_ibm_service()` with no arguments picks it up automatically.
  2. An explicit token - set IBM_QUANTUM_TOKEN (and optionally
     IBM_QUANTUM_INSTANCE / IBM_QUANTUM_CHANNEL), or pass token= directly.

With neither available, functions here fall back to a local AerSimulator so
the pipeline still runs end-to-end in demo mode - that fallback is explicit
and logged, never silent, since a simulator result is not the same claim as
a real QPU run.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
from qiskit_aer.primitives import Sampler as AerSampler

CONNECT_TIMEOUT_S = 15.0  # bounds account auth + backend-catalog calls
LEAST_BUSY_TIMEOUT_S = 8.0  # bounds the per-backend queue-status comparison
JOB_RESULT_TIMEOUT_S = 600.0  # bounds each job.result() wait (queue + execution)


def _call_with_timeout(fn, timeout_s: float, *args, **kwargs):
    """Runs fn in a helper thread and raises TimeoutError if it doesn't
    finish within timeout_s. IBM Cloud calls (auth, backend listing,
    queue-status lookups) are real network round trips with no timeout of
    their own, and a rate-limited/over-quota account can make them hang far
    longer than a web request should ever block for. Note: on timeout the
    underlying call keeps running in its thread until it finishes on its
    own (Python can't forcibly cancel a blocked network call) - this just
    stops the caller from waiting on it."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout_s)
        except FutureTimeoutError:
            raise TimeoutError(f"{getattr(fn, '__name__', fn)} did not respond within {timeout_s}s")


def _get_result_with_retry(job, retries: int = 2, backoff_s: float = 5.0, timeout_s: float = JOB_RESULT_TIMEOUT_S):
    """Real IBM jobs occasionally hit transient network errors (SSL/TLS
    blips) fetching the result of an already-completed job - retrying
    result() re-fetches from the same job, it does not resubmit or consume
    additional QPU time, so this is safe/free to retry.

    Each attempt is bounded to timeout_s (RuntimeJobV2.result() takes a
    native timeout) - a job that can't run at all (account over its usage
    limit, backend down, etc.) would otherwise block here indefinitely,
    since result() waits for real queue + execution time with no default
    cutoff. Raises the underlying job-status error on timeout so the
    caller sees *why* (e.g. "usage limit"), not a generic hang."""
    last_error = None
    for attempt in range(retries):
        try:
            return job.result(timeout=timeout_s)
        except Exception as e:  # noqa: BLE001
            last_error = e
            if attempt < retries - 1:
                time.sleep(backoff_s * (attempt + 1))
    raise last_error


def get_ibm_service(
    token: str | None = None, channel: str | None = None, instance: str | None = None
):
    """Returns a live QiskitRuntimeService, or None if neither a saved
    account nor an explicit token is available (caller falls back to a
    local simulator in that case). Bounded to CONNECT_TIMEOUT_S - IBM Cloud
    auth is a real network round trip, and this should fail fast with a
    clear TimeoutError rather than hang a request indefinitely if the
    account/API is slow (e.g. rate-limited)."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = token or os.environ.get("IBM_QUANTUM_TOKEN")
    channel = channel or os.environ.get("IBM_QUANTUM_CHANNEL")
    instance = instance or os.environ.get("IBM_QUANTUM_INSTANCE")

    if token:
        return _call_with_timeout(
            QiskitRuntimeService, CONNECT_TIMEOUT_S, channel=channel or "ibm_cloud", token=token, instance=instance
        )

    try:
        return _call_with_timeout(QiskitRuntimeService, CONNECT_TIMEOUT_S)  # uses the saved default account, if any
    except Exception:
        return None


def list_backends(service, timeout_s: float = CONNECT_TIMEOUT_S):
    """Real backend catalog for this account, bounded the same way as
    get_ibm_service - a plain listing call, no queue-status comparison, so
    this is the fast half of connecting (see pick_backend for the slower,
    least-busy-aware half used once a job is actually about to run)."""
    return _call_with_timeout(service.backends, timeout_s)


def pick_backend(service, min_qubits: int, prefer_least_busy: bool = True):
    """Picks a real IBM backend with enough qubits. Raises if none qualify -
    caller decides whether to fall back to simulation.

    prefer_least_busy queries live queue status across every qualifying
    backend to find the shortest queue - useful, but it's O(n backends)
    network calls and can be slow (worse on a rate-limited/over-quota
    account). Bounded to LEAST_BUSY_TIMEOUT_S; if it doesn't answer in
    time, this falls back to the first qualifying backend instead of
    blocking the caller - a fast, merely-not-optimal pick beats a hang."""
    candidates = [b for b in list_backends(service) if b.num_qubits >= min_qubits and not b.simulator]
    if not candidates:
        raise RuntimeError(f"no IBM backend with >= {min_qubits} qubits available on this account")
    if prefer_least_busy:
        try:
            return _call_with_timeout(service.least_busy, LEAST_BUSY_TIMEOUT_S, min_num_qubits=min_qubits)
        except TimeoutError:
            pass
    return candidates[0]


def build_feature_map(n_features: int, reps: int = 2) -> QuantumCircuit:
    """ZZFeatureMap: angle-encodes classical SAR features (e.g. VV, VH,
    VV/VH ratio, local texture stats) into qubit rotations with pairwise
    entangling terms - the entanglement is what lets the quantum kernel
    capture feature *interactions* a classical linear kernel can't."""
    return ZZFeatureMap(feature_dimension=n_features, reps=reps, entanglement="linear")


def run_feature_map(
    features: np.ndarray,
    service=None,
    backend_name: str | None = None,
    shots: int = 1024,
) -> dict:
    """Executes the feature-map circuit for one feature vector and returns
    measurement counts. Uses a real IBM backend if `service` is a live
    QiskitRuntimeService, otherwise AerSimulator (local, no network call).

    Returns {"counts": dict, "backend": str, "is_real_hardware": bool}.
    """
    n_features = len(features)
    fm = build_feature_map(n_features)
    circuit = fm.assign_parameters(features)
    circuit.measure_all()

    if service is not None:
        from qiskit_ibm_runtime import SamplerV2

        backend = service.backend(backend_name) if backend_name else pick_backend(service, n_features)
        transpiled = _transpile_for_backend(circuit, backend)
        sampler = SamplerV2(mode=backend)
        job = sampler.run([transpiled], shots=shots)
        result = _get_result_with_retry(job)
        counts = result[0].data.meas.get_counts()
        return {"counts": counts, "backend": backend.name, "is_real_hardware": True}

    sampler = AerSampler()
    job = sampler.run(circuits=[circuit], shots=shots)
    counts = job.result().quasi_dists[0].binary_probabilities()
    return {"counts": counts, "backend": "aer_simulator", "is_real_hardware": False}


def _transpile_for_backend(circuit: QuantumCircuit, backend) -> QuantumCircuit:
    from qiskit import transpile

    return transpile(circuit, backend=backend, optimization_level=1)


def quantum_kernel_entry(x1: np.ndarray, x2: np.ndarray, service=None, backend_name: str | None = None) -> float:
    """Fidelity-based quantum kernel value K(x1, x2) = |<phi(x1)|phi(x2)>|^2,
    estimated via the overlap circuit's all-zero measurement probability -
    the standard quantum kernel estimation trick (Havlicek et al. 2019)."""
    n_features = len(x1)
    fm = build_feature_map(n_features)
    circuit = fm.assign_parameters(x1)
    circuit.compose(fm.assign_parameters(x2).inverse(), inplace=True)
    circuit.measure_all()

    if service is not None:
        from qiskit_ibm_runtime import SamplerV2

        backend = service.backend(backend_name) if backend_name else pick_backend(service, n_features)
        transpiled = _transpile_for_backend(circuit, backend)
        sampler = SamplerV2(mode=backend)
        job = sampler.run([transpiled], shots=1024)
        counts = _get_result_with_retry(job)[0].data.meas.get_counts()
    else:
        sampler = AerSampler()
        result = sampler.run(circuits=[circuit], shots=1024).result()
        probs = result.quasi_dists[0].binary_probabilities()
        counts = {k: int(v * 1024) for k, v in probs.items()}

    zero_state = "0" * n_features
    total = sum(counts.values())
    return counts.get(zero_state, 0) / total if total else 0.0


def _build_overlap_circuit(x1: np.ndarray, x2: np.ndarray) -> QuantumCircuit:
    n_features = len(x1)
    fm = build_feature_map(n_features)
    circuit = fm.assign_parameters(x1)
    circuit.compose(fm.assign_parameters(x2).inverse(), inplace=True)
    circuit.measure_all()
    return circuit


def compute_gram_matrix_batched(
    rows: np.ndarray,
    cols: np.ndarray | None = None,
    service=None,
    backend_name: str | None = None,
    shots: int = 1024,
) -> np.ndarray:
    """Real-hardware-efficient gram matrix: builds every needed overlap
    circuit up front and submits them all as ONE Qiskit Runtime job
    (SamplerV2 accepts a list of circuits - one job, one queue wait,
    N results), instead of quantum_kernel_entry's one-job-per-pair
    approach. This is the fix for the O(n^2)-jobs bottleneck that made a
    24x12 gram matrix take 7-30 hours of real queue time in testing - the
    same 24x12 matrix here is one job instead of ~300.

    cols=None means symmetric (rows vs itself, e.g. a training gram
    matrix) - only the upper triangle is actually computed and submitted,
    mirrored into the lower triangle after the fact.
    """
    symmetric = cols is None
    other = rows if symmetric else cols
    n, m = len(rows), len(other)

    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(m):
            if symmetric and j < i:
                continue
            pairs.append((i, j))

    if service is None:
        gram = np.zeros((n, m))
        for i, j in pairs:
            val = quantum_kernel_entry(rows[i], other[j], service=None)
            gram[i, j] = val
            if symmetric:
                gram[j, i] = val
        return gram

    from qiskit_ibm_runtime import SamplerV2

    n_features = rows.shape[1]
    backend = service.backend(backend_name) if backend_name else pick_backend(service, n_features)
    circuits = [_transpile_for_backend(_build_overlap_circuit(rows[i], other[j]), backend) for i, j in pairs]

    sampler = SamplerV2(mode=backend)
    job = sampler.run(circuits, shots=shots)
    result = _get_result_with_retry(job)

    zero_state = "0" * n_features
    gram = np.zeros((n, m))
    for idx, (i, j) in enumerate(pairs):
        counts = result[idx].data.meas.get_counts()
        total = sum(counts.values())
        val = counts.get(zero_state, 0) / total if total else 0.0
        gram[i, j] = val
        if symmetric:
            gram[j, i] = val
    return gram
