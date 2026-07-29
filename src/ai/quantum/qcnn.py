"""Quantum convolutional patch encoder (PennyLane, torch-interface).

Encodes small (e.g. 4x4) pixel patches into rotation angles, applies a
parameterized entangling circuit, and reads out expectation values as a
feature vector - a quantum analogue of a conv layer's local receptive field.
Meant to sit inside src/ai/quantum/hybrid_unet.py as one bottleneck block,
not to replace the whole network (current NISQ devices/simulators can't
scale to full-image qubit counts).
"""

import pennylane as qml
import torch
from torch import nn

N_QUBITS = 4  # one qubit per pixel in a 2x2 patch


def _circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(w)) for w in range(N_QUBITS)]


class QuantumConvLayer(nn.Module):
    """Applies the quantum circuit as a 2x2, stride-2 patch encoder over a
    single-channel feature map, producing N_QUBITS output channels."""

    def __init__(self, n_layers: int = 2, backend: str = "default.qubit"):
        super().__init__()
        dev = qml.device(backend, wires=N_QUBITS)
        qnode = qml.QNode(_circuit, dev, interface="torch", diff_method="backprop")
        weight_shapes = {"weights": (n_layers, N_QUBITS, 3)}
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) with H, W even. Returns (B, N_QUBITS, H//2, W//2)."""
        b, _, h, w = x.shape
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # (B,1,H/2,W/2,2,2)
        patches = patches.reshape(b, h // 2, w // 2, 4)

        out = torch.zeros(b, N_QUBITS, h // 2, w // 2, device=x.device)
        for i in range(h // 2):
            for j in range(w // 2):
                q_out = self.qlayer(patches[:, i, j, :])
                out[:, :, i, j] = q_out
        return out
