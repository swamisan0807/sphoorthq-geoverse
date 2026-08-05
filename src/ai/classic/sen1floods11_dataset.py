"""Real PyTorch Dataset over the sen1floods11 HandLabeled chips.

Bridges the raw downloaded GeoTIFFs (datasets/raw/sen1floods11/) to
training-ready tensors. S1 backscatter values in this dataset are already
calibrated (dB-like, roughly [-50, 1]) - normalized here per-channel using
fixed sane bounds rather than dataset statistics, since VV/VH physical
ranges are well known and stable across scenes (avoids batch-to-batch
normalization drift on a 252-chip training set).

Label convention (sen1floods11): -1 = no data, 0 = not water, 1 = water.
No-data LABEL pixels are excluded via valid_mask. Separately, ~9% of chips
(confirmed by direct inspection) have actual NaN/Inf pixels in the raw S1
GeoTIFF itself - a nodata sentinel independent of the label's -1 convention,
and NOT always at the same pixel locations (e.g. Pakistan_43105 has 733 NaN
S1 pixels, zero of which coincide with a -1 label). Left unhandled, NaN
silently propagates into every consumer: a per-pixel classifier gets NaN
feature rows, and a CNN's convolutions spread a single NaN pixel across a
growing receptive field until most of the output is NaN. Both `normalize_s1`
and `valid_mask` below account for this independently of the label mask.
"""

import csv

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from src.core.paths import RAW_DIR

SEN1FLOODS11_ROOT = RAW_DIR / "sen1floods11"
HAND_LABELED_DIR = SEN1FLOODS11_ROOT / "data" / "flood_events" / "HandLabeled"
SPLITS_DIR = SEN1FLOODS11_ROOT / "splits" / "flood_handlabeled"

S1_DB_MIN, S1_DB_MAX = -50.0, 1.0  # fixed physical bounds for Sentinel-1 sigma0 dB


def load_split(split: str) -> list[tuple[str, str]]:
    """split: 'train' | 'valid' | 'test'. Returns list of (s1_filename, label_filename)."""
    path = SPLITS_DIR / f"flood_{split}_data.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return [(row[0], row[1]) for row in csv.reader(f) if row]


def chip_id_from_s1_filename(s1_filename: str) -> str:
    return s1_filename.removesuffix("_S1Hand.tif")


def read_s1(chip_id: str) -> np.ndarray:
    with rasterio.open(HAND_LABELED_DIR / "S1Hand" / f"{chip_id}_S1Hand.tif") as src:
        return src.read().astype(np.float32)  # (2, H, W) VV, VH


def read_label(chip_id: str) -> np.ndarray:
    with rasterio.open(HAND_LABELED_DIR / "LabelHand" / f"{chip_id}_LabelHand.tif") as src:
        return src.read(1).astype(np.int64)  # (H, W), values in {-1, 0, 1}


def normalize_s1(s1: np.ndarray) -> np.ndarray:
    clipped = np.clip(s1, S1_DB_MIN, S1_DB_MAX)
    normalized = (clipped - S1_DB_MIN) / (S1_DB_MAX - S1_DB_MIN)  # -> [0, 1]
    # np.clip does not remove NaN/Inf - sanitize explicitly so a nodata
    # pixel can never silently reach a model as a real value.
    return np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)


def s1_nodata_mask(s1: np.ndarray) -> np.ndarray:
    """True where the raw (pre-normalization) S1 pixel is NaN/Inf in any
    channel - independent of the label's own -1 no-data convention."""
    return np.isnan(s1).any(axis=0) | np.isinf(s1).any(axis=0)


class Sen1Floods11Dataset(Dataset):
    def __init__(self, split: str):
        self.pairs = load_split(split)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        s1_filename, _label_filename = self.pairs[idx]
        chip_id = chip_id_from_s1_filename(s1_filename)

        raw_s1 = read_s1(chip_id)
        s1 = normalize_s1(raw_s1)
        label = read_label(chip_id)

        valid_mask = (label != -1) & ~s1_nodata_mask(raw_s1)
        target = np.where(valid_mask, label, 0).astype(np.float32)

        return {
            "chip_id": chip_id,
            "x": torch.from_numpy(s1),
            "y": torch.from_numpy(target),
            "valid_mask": torch.from_numpy(valid_mask),
        }
