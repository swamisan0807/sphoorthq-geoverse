import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        target_flat = target.view(target.size(0), -1).float()

        intersection = (probs_flat * target_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class TverskyLoss(nn.Module):
    """Generalized Dice with independent FP/FN weighting - use alpha>beta to
    penalize false positives more (fewer spurious flood/water pixels)."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).view(-1)
        target = target.view(-1).float()

        tp = (probs * target).sum()
        fp = (probs * (1 - target)).sum()
        fn = ((1 - probs) * target).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky


class ComboLoss(nn.Module):
    """BCE + Dice - BCE stabilizes early training, Dice targets IoU directly."""

    def __init__(self, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        return (1 - self.dice_weight) * self.bce(logits, target) + self.dice_weight * self.dice(
            logits, target
        )


class MaskedComboLoss(nn.Module):
    """ComboLoss that excludes no-data pixels (e.g. sen1floods11's label=-1)
    from both terms, via a boolean valid_mask - BCEWithLogitsLoss has no
    native masking support, so it's computed per-pixel and mean-reduced
    over valid pixels only; Dice is computed on values already zeroed at
    invalid pixels in both prediction and target, which is exact for Dice
    (a masked-out pixel contributes 0 to both intersection and union)."""

    def __init__(self, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.dice = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        target = target.float()
        mask = valid_mask.float()

        bce_per_pixel = self.bce(logits, target)
        bce = (bce_per_pixel * mask).sum() / mask.sum().clamp(min=1.0)

        masked_probs_logits = logits * mask + (1 - mask) * -10.0  # push invalid pixels' sigmoid to ~0
        dice = self.dice(masked_probs_logits, target * mask)

        return (1 - self.dice_weight) * bce + self.dice_weight * dice
