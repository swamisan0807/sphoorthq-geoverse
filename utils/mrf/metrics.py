"""Real Accuracy/IoU/Dice - computable only when real ground truth exists,
which real MSTAR chips do not (see utils/mrf/mstar_scenes.py and every
apps/api/routers/mstar.py response's `note`). Used exclusively by
utils/mrf/synthetic_validation.py, where real known labels genuinely exist
(THALES's own worked examples, and small generated cases with a known
target region) - never applied to a real MSTAR result."""

import numpy as np


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """Accuracy over all pixels; IoU/Dice for the "target" class (any label
    >= 1, i.e. not background) - the standard convention for a single-number
    object-segmentation score, including when true/pred are Q=3 (shadow
    counted as part of "not background" here, same as accuracy's per-pixel
    exact-label-match already treats it as its own class)."""
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    if pred_flat.shape != true_flat.shape:
        raise ValueError(f"shape mismatch: pred {pred_flat.shape} vs true {true_flat.shape}")

    accuracy = float(np.mean(pred_flat == true_flat))

    pred_target = pred_flat >= 1
    true_target = true_flat >= 1
    intersection = int(np.sum(pred_target & true_target))
    union = int(np.sum(pred_target | true_target))
    iou = float(intersection / union) if union > 0 else 1.0  # both all-background - trivially perfect agreement

    denom = int(np.sum(pred_target)) + int(np.sum(true_target))
    dice = float(2 * intersection / denom) if denom > 0 else 1.0

    return {"accuracy": accuracy, "iou": iou, "dice": dice}
