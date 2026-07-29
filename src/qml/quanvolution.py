"""Quanvolutional filtering (Henderson et al. 2020, "Quanvolutional Neural
Networks"): a FIXED (untrained, randomly-initialized) small quantum
circuit applied like a convolution - slid across an image in patches,
each patch's pixel values angle-encoded, one output channel per qubit
from Z-expectation measurements. Unlike the quantum bottleneck
(quantum_layer.py), this circuit is not trained - it's a fixed random
feature extractor, the same role a random-projection convolution plays
classically.

Real and batched (every patch across the whole image is submitted as one
Estimator.run() call, not one call per patch) - but still O(H*W/stride^2)
circuit *evaluations*, a real and substantial cost at full image
resolution: a 512x512 chip at patch_size=2/stride=2 is 65,536 patches x 4
observables = 262,144 circuit-observable evaluations. Measured at ~595
evals/sec locally (a 32x32 crop: 1,024 evals in 1.72s - see notebook 10's
timing cell), that extrapolates to ~7.3 minutes for one full chip - real,
not fabricated, but why this is demonstrated on a small crop rather than
applied to the full training set.
"""

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Estimator as AerEstimator


def _build_quanv_circuit(n_qubits: int, seed: int) -> tuple[QuantumCircuit, ParameterVector]:
    rng = np.random.default_rng(seed)
    params = ParameterVector("x", n_qubits)
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(params[i], i)
    for i in range(n_qubits - 1):
        qc.cz(i, i + 1)
    # fixed random single-qubit rotations, deterministic given seed - this
    # is what makes the circuit a *fixed* feature extractor, not trained
    for i in range(n_qubits):
        qc.ry(float(rng.uniform(0, np.pi)), i)
    return qc, params


def quanvolve(image: np.ndarray, patch_size: int = 2, stride: int = 2, seed: int = 0) -> np.ndarray:
    """image: (H, W), values expected roughly in [0, 1]. Returns
    (n_qubits, H_out, W_out) where n_qubits = patch_size**2 - one output
    channel per qubit, same spatial-downsampling semantics as a classical
    strided convolution."""
    n_qubits = patch_size * patch_size
    qc, _params = _build_quanv_circuit(n_qubits, seed=seed)
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (n_qubits - i - 1)) for i in range(n_qubits)]

    h, w = image.shape
    h_out = (h - patch_size) // stride + 1
    w_out = (w - patch_size) // stride + 1

    patch_angles = []
    for i in range(0, h_out * stride, stride):
        for j in range(0, w_out * stride, stride):
            patch = image[i : i + patch_size, j : j + patch_size].reshape(-1)
            patch_angles.append(np.clip(patch, 0.0, 1.0) * np.pi)

    n_patches = len(patch_angles)
    circuits = [qc] * (n_patches * n_qubits)
    obs_list = [observables[i % n_qubits] for i in range(n_patches * n_qubits)]
    param_vals = [angles for angles in patch_angles for _ in range(n_qubits)]

    estimator = AerEstimator()
    result = estimator.run(circuits, obs_list, param_vals).result()

    values = np.array(result.values).reshape(n_patches, n_qubits)
    return values.reshape(h_out, w_out, n_qubits).transpose(2, 0, 1)
