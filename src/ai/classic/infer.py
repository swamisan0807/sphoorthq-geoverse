import numpy as np
import torch

from src.ai.classic.unet import UNet


def predict(model: UNet, cube: np.ndarray, device: str = "cpu", threshold: float = 0.5) -> np.ndarray:
    """cube: (bands, H, W) -> binary mask (H, W)."""
    model.eval()
    x = torch.from_numpy(cube).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
    return (probs.squeeze().cpu().numpy() > threshold).astype(np.uint8)
