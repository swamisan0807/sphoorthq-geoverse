"""Model-level (late) fusion: per-modality encoders merged before the segmentation head.

Alternative to early fusion (src/fusion/stacking.py) - useful when modalities have
very different statistics (SAR vs optical vs categorical land cover) and a shared
first conv layer would wash out modality-specific structure.
"""

import torch
from torch import nn


class ModalityEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LateFusionBackbone(nn.Module):
    """Encodes each modality separately, concatenates features on the channel axis."""

    def __init__(self, modality_channels: dict[str, int], out_channels_per_modality: int = 32):
        super().__init__()
        self.encoders = nn.ModuleDict(
            {name: ModalityEncoder(ch, out_channels_per_modality) for name, ch in modality_channels.items()}
        )
        self.out_channels = out_channels_per_modality * len(modality_channels)

    def forward(self, modality_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        features = [self.encoders[name](tensor) for name, tensor in modality_inputs.items()]
        return torch.cat(features, dim=1)
