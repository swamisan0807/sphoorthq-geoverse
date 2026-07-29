"""Real AWS Braket integration.

Mirrors src/qml/ibm_quantum.py's interface (build_feature_map,
quantum_kernel_entry) so the two backends are directly swappable/comparable
in src/qml/hybrid_classifier.py. Auth via standard AWS credential chain
(env vars, ~/.aws/credentials, IAM role) - nothing hardcoded. Device is
selected via BRAKET_DEVICE_ARN (e.g. a managed simulator like SV1, or a
real QPU such as IonQ/Rigetti/IQM); with no ARN configured, falls back to
Braket's LocalSimulator (runs on this machine, no AWS call, no cost).
"""

import os

import numpy as np
from braket.circuits import Circuit, FreeParameter
from braket.devices import LocalSimulator

DEFAULT_SIMULATOR_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"


def get_braket_device(device_arn: str | None = None):
    """Returns a live AwsDevice if BRAKET_DEVICE_ARN (or device_arn) is set
    and AWS credentials resolve, otherwise a local simulator. The caller is
    told which one it got via device.name / is_real_hardware detection in
    run_feature_map below - never silently misreport which one ran."""
    arn = device_arn or os.environ.get("BRAKET_DEVICE_ARN")
    if not arn:
        return LocalSimulator()

    from braket.aws import AwsDevice

    return AwsDevice(arn)


def build_feature_map(n_features: int, reps: int = 2) -> Circuit:
    """ZZ-style angle-encoding feature map, Braket-native equivalent of
    qiskit's ZZFeatureMap in src/qml/ibm_quantum.py - same encoding scheme
    so kernel values are comparable across backends."""
    circuit = Circuit()
    params = [FreeParameter(f"x{i}") for i in range(n_features)]

    for _ in range(reps):
        for i in range(n_features):
            circuit.h(i)
            circuit.rz(i, 2 * params[i])
        for i in range(n_features - 1):
            circuit.cnot(i, i + 1)
            circuit.rz(i + 1, 2 * (np.pi - params[i]) * (np.pi - params[i + 1]))
            circuit.cnot(i, i + 1)

    return circuit


def _bind_params(circuit: Circuit, features: np.ndarray) -> Circuit:
    bindings = {f"x{i}": float(v) for i, v in enumerate(features)}
    return circuit.make_bound_circuit(bindings)


def run_feature_map(
    features: np.ndarray, device=None, device_arn: str | None = None, shots: int = 1024
) -> dict:
    """Executes the feature-map circuit for one feature vector and returns
    measurement counts. Returns {"counts": dict, "backend": str,
    "is_real_hardware": bool}."""
    device = device or get_braket_device(device_arn)
    n_features = len(features)

    circuit = build_feature_map(n_features)
    bound = _bind_params(circuit, features)

    task = device.run(bound, shots=shots)
    result = task.result()
    counts = dict(result.measurement_counts)

    is_real_hardware = "LocalSimulator" not in type(device).__name__ and "simulator" not in getattr(
        device, "arn", ""
    ).lower()
    backend_name = getattr(device, "name", type(device).__name__)
    return {"counts": counts, "backend": backend_name, "is_real_hardware": is_real_hardware}


def quantum_kernel_entry(
    x1: np.ndarray, x2: np.ndarray, device=None, device_arn: str | None = None, shots: int = 1024
) -> float:
    """Fidelity-based quantum kernel value, same estimation approach as
    src/qml/ibm_quantum.quantum_kernel_entry - measure the overlap circuit's
    all-zero probability."""
    device = device or get_braket_device(device_arn)
    n_features = len(x1)

    fm = build_feature_map(n_features)
    circuit = _bind_params(fm, x1)
    inverse_fm = build_feature_map(n_features).adjoint()
    circuit.add_circuit(_bind_params(inverse_fm, x2))

    task = device.run(circuit, shots=shots)
    counts = dict(task.result().measurement_counts)

    zero_state = "0" * n_features
    total = sum(counts.values())
    return counts.get(zero_state, 0) / total if total else 0.0


def _build_overlap_circuit(x1: np.ndarray, x2: np.ndarray) -> Circuit:
    n_features = len(x1)
    fm = build_feature_map(n_features)
    circuit = _bind_params(fm, x1)
    inverse_fm = build_feature_map(n_features).adjoint()
    circuit.add_circuit(_bind_params(inverse_fm, x2))
    return circuit


def compute_gram_matrix_batched(
    rows: np.ndarray,
    cols: np.ndarray | None = None,
    device=None,
    device_arn: str | None = None,
    shots: int = 1024,
) -> np.ndarray:
    """Mirrors src/qml/ibm_quantum.compute_gram_matrix_batched: submits every
    needed overlap circuit as one `run_batch` call on a real AwsDevice
    instead of one `run` call per pair - the same fix for the O(n^2)-jobs
    real-hardware bottleneck, on the Braket side. LocalSimulator has no
    batch endpoint, but it's local/free/fast, so it just loops instead."""
    device = device or get_braket_device(device_arn)
    symmetric = cols is None
    other = rows if symmetric else cols
    n, m = len(rows), len(other)

    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(m):
            if symmetric and j < i:
                continue
            pairs.append((i, j))

    n_features = rows.shape[1]
    zero_state = "0" * n_features
    gram = np.zeros((n, m))

    if hasattr(device, "run_batch"):
        circuits = [_build_overlap_circuit(rows[i], other[j]) for i, j in pairs]
        batch = device.run_batch(circuits, shots=shots)
        results = batch.results()
        for (i, j), result in zip(pairs, results):
            counts = dict(result.measurement_counts)
            total = sum(counts.values())
            val = counts.get(zero_state, 0) / total if total else 0.0
            gram[i, j] = val
            if symmetric:
                gram[j, i] = val
        return gram

    for i, j in pairs:
        val = quantum_kernel_entry(rows[i], other[j], device=device)
        gram[i, j] = val
        if symmetric:
            gram[j, i] = val
    return gram
