"""Quantum-kernel classifier for SAR water/not-water pixels, combining
classical feature engineering with a real quantum kernel computed on IBM
Quantum (or a local simulator fallback).

This is the ML+QML hybrid: classical feature extraction (src/processing,
src/features) produces a low-dimensional per-pixel/superpixel feature
vector (VV, VH, VV/VH ratio, local texture), which becomes the input to a
quantum feature map. A classical SVM then classifies using the
quantum-computed kernel (Gram) matrix instead of a classical kernel -
this is the quantum kernel method (Havlicek et al. 2019, Schuld & Killoran
2019): quantum computation contributes the kernel's similarity measure,
everything else stays classical.
"""

from enum import Enum

import numpy as np
from sklearn.svm import SVC

from src.qml import ibm_quantum


class QuantumBackend(str, Enum):
    IBM = "ibm"


def compute_kernel_entry(x1: np.ndarray, x2: np.ndarray, backend: QuantumBackend, **kwargs) -> float:
    if backend == QuantumBackend.IBM:
        return ibm_quantum.quantum_kernel_entry(x1, x2, **kwargs)
    raise ValueError(f"unknown backend: {backend}")


def compute_gram_matrix(
    features: np.ndarray, backend: QuantumBackend, symmetric_input: np.ndarray | None = None, **kwargs
) -> np.ndarray:
    """features: (n_samples, n_qubits). Returns the (n, n) quantum kernel
    Gram matrix, via the batched path (src/qml/ibm_quantum.py:
    compute_gram_matrix_batched) - one real job submission for the whole
    matrix instead of one per pair. Still O(n^2) circuit *evaluations*
    (a fundamental cost of quantum kernel methods), but no longer O(n^2)
    *job submissions* - that distinction is what makes a 24x12 matrix take
    one job's queue wait instead of ~300 jobs' worth. Keep n in the tens,
    not full-image pixel counts - a single job's circuit count still has
    practical limits on real hardware."""
    if backend == QuantumBackend.IBM:
        return ibm_quantum.compute_gram_matrix_batched(features, symmetric_input, **kwargs)
    raise ValueError(f"unknown backend: {backend}")


class QuantumKernelSVM:
    """Trains a classical SVM on a quantum-computed kernel matrix - the
    quantum-classical hybrid model itself. `backend` picks which real
    quantum cloud service supplies the kernel."""

    def __init__(self, backend: QuantumBackend, **backend_kwargs):
        self.backend = backend
        self.backend_kwargs = backend_kwargs
        self.svm = SVC(kernel="precomputed")
        self._train_features: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "QuantumKernelSVM":
        gram = compute_gram_matrix(features, self.backend, **self.backend_kwargs)
        self.svm.fit(gram, labels)
        self._train_features = features
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._train_features is None:
            raise RuntimeError("call fit() before predict()")
        gram = compute_gram_matrix(
            features, self.backend, symmetric_input=self._train_features, **self.backend_kwargs
        )
        return self.svm.predict(gram)
