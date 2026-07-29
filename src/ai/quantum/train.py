"""Training loop for HybridUNet - mirrors src/ai/classic/train.py so classic
and quantum models are benchmarked on identical data/splits/metrics."""

import torch
from torch.utils.data import DataLoader

from src.ai.classic.losses import ComboLoss
from src.ai.classic.train import PatchDataset
from src.ai.quantum.hybrid_unet import HybridUNet
from src.processing.tiling import Patch


def train(
    train_patches: list[Patch],
    val_patches: list[Patch],
    in_channels: int,
    epochs: int = 20,
    batch_size: int = 4,  # smaller default - quantum layer is the bottleneck cost
    lr: float = 1e-3,
    device: str = "cpu",  # quantum simulators are typically CPU-bound
) -> HybridUNet:
    model = HybridUNet(in_channels=in_channels, num_classes=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = ComboLoss()

    train_loader = DataLoader(PatchDataset(train_patches), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(PatchDataset(val_patches), batch_size=batch_size)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).unsqueeze(1)
                val_loss += criterion(model(x), y).item() * x.size(0)

        print(
            f"epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss / len(train_patches):.4f} "
            f"val_loss={val_loss / max(len(val_patches), 1):.4f}"
        )

    return model
