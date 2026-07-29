"""Hybrid model: classical U-Net encoder/decoder with a quantum bottleneck block.

Keeps the expensive spatial downsampling classical (quantum simulators can't
handle full-resolution feature maps), and uses the quantum layer only at the
bottleneck where the feature map is small enough to be tractable.
"""

import torch
from torch import nn

from src.ai.classic.unet import DoubleConv
from src.ai.quantum.qcnn import QuantumConvLayer


class HybridUNet(nn.Module):
    def __init__(self, in_channels: int = 11, num_classes: int = 1, base_channels: int = 32):
        super().__init__()
        c = base_channels

        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.pool = nn.MaxPool2d(2)

        # classical projection down to 1 channel before the quantum layer -
        # N_QUBITS scales linearly with cost on a simulator, keep it minimal.
        self.pre_quantum = nn.Conv2d(c * 4, 1, 1)
        self.quantum = QuantumConvLayer(n_layers=2)
        self.post_quantum = nn.Conv2d(4, c * 4, 1)  # N_QUBITS=4 -> back to c*4

        self.up3 = nn.ConvTranspose2d(c * 4, c * 4, 2, stride=2)
        self.dec3 = DoubleConv(c * 4 + c * 4, c * 2)
        self.up2 = nn.ConvTranspose2d(c * 2, c * 2, 2, stride=2)
        self.dec2 = DoubleConv(c * 2 + c * 2, c)
        self.up1 = nn.ConvTranspose2d(c, c, 2, stride=2)
        self.dec1 = DoubleConv(c + c, c)

        self.head = nn.Conv2d(c, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        q_in = self.pre_quantum(self.pool(e3))
        q_out = self.quantum(q_in)
        bottleneck = self.post_quantum(q_out)

        d3 = self.dec3(torch.cat([self.up3(bottleneck), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)
