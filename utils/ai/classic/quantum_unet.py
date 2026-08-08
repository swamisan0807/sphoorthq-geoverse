"""Quantum-enhanced U-Net: same encoder/decoder as `unet.UNet`, but the
bottleneck feature map is additionally gated by a real quantum circuit -
global-average-pool the bottleneck -> compress to n_qubits -> quantum
layer (utils/qml/quantum_layer.py) -> expand back to a per-channel gate ->
multiply (squeeze-and-excitation-style, quantum instead of classical).
This is the "Quantum-Enhanced Bottleneck" from the reference architecture,
trained end-to-end (see quantum_layer.py's docstring for why AdamW alone,
not a split classical/SPSA optimizer).

Kept out of unet.py on purpose: this module requires
qiskit-machine-learning (requirements.txt) - the plain UNet does
not, and importing this file pulls that whole dependency chain in.
"""

import torch
from torch import nn

from utils.ai.classic.unet import DoubleConv
from utils.qml.quantum_layer import build_quantum_bottleneck


class QuantumEnhancedUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 1,
        base_channels: int = 16,
        n_qubits: int = 6,
    ):
        super().__init__()
        c = base_channels

        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.enc4 = DoubleConv(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(c * 8, c * 16)
        bottleneck_channels = c * 16

        self.quantum_compress = nn.Linear(bottleneck_channels, n_qubits)
        self.quantum_layer = build_quantum_bottleneck(n_qubits)
        self.quantum_expand = nn.Linear(n_qubits, bottleneck_channels)

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = DoubleConv(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = DoubleConv(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = DoubleConv(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = DoubleConv(c * 2, c)

        self.head = nn.Conv2d(c, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        pooled = b.mean(dim=[2, 3])  # (B, C) - global average pool, one vector per sample
        # tanh + scale keeps the angle-encoded inputs in a sane range for
        # the quantum feature map (raw pooled activations are unbounded)
        angles = torch.tanh(self.quantum_compress(pooled)) * (torch.pi / 2)
        quantum_out = self.quantum_layer(angles)  # (B, n_qubits) - real circuit, real gradients
        gate = torch.sigmoid(self.quantum_expand(quantum_out))  # (B, C) channel gate
        b = b * gate.unsqueeze(-1).unsqueeze(-1)

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)
