"""Reference masks for real MSTAR chips, from the SARBake overlays.

MSTAR's public release has no per-pixel masks. SARBake (Malmgren-Hansen and
Nobel-Jorgensen, 2015; CC BY 4.0, doi:10.17632/jxhsg8tj7g.3) fills that gap:
for each MSTAR chip it provides a 128x128 label mask (0 background, 1 target,
2 shadow) generated from a 3D CAD model of the vehicle at the chip's viewing
angle. The masks are a good reference, not hand-annotated ground truth: they
come from CAD geometry and cover the bright vehicle return, not every pixel a
person might outline.

Only the CSV masks are needed. They are looked up by the chip's own file name
(e.g. HB03333.015) inside the folder for its vehicle type. Set SARBAKE_DIR to
the folder that holds the BMP2, BTR70 and T72 sub-folders; it defaults to
datasets/raw/sarbake. When a mask is missing, every function here returns None
and the app falls back to reporting energy only.
"""

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from utils.core.paths import DATASETS_DIR
from utils.mrf.mstar_scenes import CATALOG

BACKGROUND, TARGET, SHADOW = 0, 1, 2


def sarbake_dir() -> Path:
    return Path(os.environ.get("SARBAKE_DIR", DATASETS_DIR / "raw" / "sarbake"))


@lru_cache(maxsize=256)
def _read_csv(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter=",").astype(np.int8)


def mask_path_for_file(vehicle: str, file_name: str) -> Path:
    return sarbake_dir() / vehicle.upper() / f"{file_name}.csv"


def load_mask(chip_id: str) -> np.ndarray | None:
    """Full 128x128 mask for a catalog chip, or None if it isn't available."""
    rel = CATALOG.get(chip_id)
    if rel is None:
        return None
    parts = Path(rel).parts  # e.g. TARGETS/TEST/15_DEG/T72/SN_132/HB03333.015
    if parts[0] != "TARGETS":  # SLICY has no SARBake masks
        return None
    path = mask_path_for_file(parts[3], parts[-1])
    if not path.exists():
        return None
    return _read_csv(str(path))


def load_mask_window(chip_id: str, row0: int, col0: int, size: int) -> np.ndarray | None:
    full = load_mask(chip_id)
    if full is None:
        return None
    return full[row0 : row0 + size, col0 : col0 + size]


def boundary_window(chip_id: str, size: int, search: int = 16) -> tuple[int, int] | None:
    """Top-left corner of a size x size window on the vehicle's edge: the one
    whose share of target pixels is closest to half, searched within `search`
    pixels of the chip's bright centre (ties go to the window nearest it).
    This is where a segmentation actually has to decide between target and
    background, unlike a window deep inside the vehicle. None without a mask."""
    from utils.mrf.mstar_scenes import chip_info

    mask = load_mask(chip_id)
    if mask is None:
        return None
    info = chip_info(chip_id)
    h, w = mask.shape
    best = None
    for r in range(max(0, info["bright_row"] - search), min(h - size, info["bright_row"] + search) + 1):
        for c in range(max(0, info["bright_col"] - search), min(w - size, info["bright_col"] + search) + 1):
            share = float(np.mean(mask[r : r + size, c : c + size] == TARGET))
            key = (abs(share - 0.5), abs(r - info["bright_row"]) + abs(c - info["bright_col"]))
            if best is None or key < best[0]:
                best = (key, (r, c))
    return best[1]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.sum(a | b))
    return float(np.sum(a & b) / union) if union else 1.0


def score_binary(pred: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Two-class score against the mask's target class (shadow counts as
    background here, since the two-class model has no shadow label)."""
    p = pred.astype(bool)
    t = mask == TARGET
    tp = int(np.sum(p & t))
    denom = int(np.sum(p)) + int(np.sum(t))
    return {
        "accuracy": float(np.mean(p == t)),
        "iou": _iou(p, t),
        "dice": float(2 * tp / denom) if denom else 1.0,
        "precision": float(tp / np.sum(p)) if np.sum(p) else 1.0,
        "recall": float(tp / np.sum(t)) if np.sum(t) else 1.0,
    }


def score_three_class(pred: np.ndarray, mask: np.ndarray) -> dict[str, float | None]:
    """Accuracy and per-class IoU. A class that is absent from both the
    prediction and the mask has no IoU (None), and the mean covers only the
    classes that occur, so a tile without shadow is not credited with a perfect
    shadow score."""
    ious: list[float | None] = []
    for k in (BACKGROUND, TARGET, SHADOW):
        p, t = pred == k, mask == k
        ious.append(_iou(p, t) if (p | t).any() else None)
    present = [v for v in ious if v is not None]
    return {
        "accuracy": float(np.mean(pred == mask)),
        "iou_background": ious[0],
        "iou_target": ious[1],
        "iou_shadow": ious[2],
        "mean_iou": float(np.mean(present)),
    }
