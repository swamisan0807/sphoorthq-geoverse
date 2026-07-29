"""Training loop for the classic U-Net. Wired against src/processing/tiling.Patch
lists - swap the dataset with a real DataLoader once datasets/exports is populated."""

import torch
from torch.utils.data import DataLoader, Dataset

from src.ai.classic.losses import ComboLoss
from src.ai.classic.unet import UNet
from src.processing.tiling import Patch


class PatchDataset(Dataset):
    def __init__(self, patches: list[Patch]):
        self.patches = patches

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        p = self.patches[idx]
        x = torch.from_numpy(p.array).float()
        y = torch.from_numpy(p.label).long() if p.label is not None else None
        return x, y


def train(
    train_patches: list[Patch],
    val_patches: list[Patch],
    in_channels: int,
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> UNet:
    model = UNet(in_channels=in_channels, num_classes=1).to(device)
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
            logits = model(x)
            loss = criterion(logits, y)
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
