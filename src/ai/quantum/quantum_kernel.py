"""Quantum kernel SVM for pixel/superpixel classification.

Alternative to the QCNN: instead of a trainable quantum layer inside a deep
net, this computes a quantum kernel (fidelity between feature-encoded quantum
states) and feeds it to a classical SVM - viable on small labeled sets
(e.g. per-superpixel classification) where full deep training isn't justified.
"""

import numpy as np
import pennylane as qml
from sklearn.svm import SVC


def make_kernel(n_qubits: int, backend: str = "default.qubit"):
    dev = qml.device(backend, wires=n_qubits)

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        qml.AngleEmbedding(x1, wires=range(n_qubits))
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits))
        return qml.probs(wires=range(n_qubits))

    def kernel(x1: np.ndarray, x2: np.ndarray) -> float:
        return kernel_circuit(x1, x2)[0]

    return kernel


def gram_matrix(x: np.ndarray, kernel) -> np.ndarray:
    n = len(x)
    gram = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            gram[i, j] = gram[j, i] = kernel(x[i], x[j])
    return gram


def train_quantum_svm(x_train: np.ndarray, y_train: np.ndarray, n_qubits: int) -> tuple[SVC, callable]:
    kernel = make_kernel(n_qubits)
    gram = gram_matrix(x_train, kernel)
    clf = SVC(kernel="precomputed")
    clf.fit(gram, y_train)
    return clf, kernel
