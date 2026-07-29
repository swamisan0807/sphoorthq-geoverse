"""Real PyTorch Dataset over the sen1floods11 HandLabeled chips.

Bridges the raw downloaded GeoTIFFs (datasets/raw/sen1floods11/) to
training-ready tensors. S1 backscatter values in this dataset are already
calibrated (dB-like, roughly [-50, 1]) - normalized here per-channel using
fixed sane bounds rather than dataset statistics, since VV/VH physical
ranges are well known and stable across scenes (avoids batch-to-batch
normalization drift on a 252-chip training set).

Label convention (sen1floods11): -1 = no data, 0 = not water, 1 = water.
No-data pixels are excluded from the loss via a mask channel.
"""

import csv
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

SEN1FLOODS11_ROOT = Path(r"d:\project-raw-data\sphoorthq-geoverse\datasets\raw\sen1floods11")
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
    return (clipped - S1_DB_MIN) / (S1_DB_MAX - S1_DB_MIN)  # -> [0, 1]


class Sen1Floods11Dataset(Dataset):
    def __init__(self, split: str):
        self.pairs = load_split(split)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        s1_filename, _label_filename = self.pairs[idx]
        chip_id = chip_id_from_s1_filename(s1_filename)

        s1 = normalize_s1(read_s1(chip_id))
        label = read_label(chip_id)

        valid_mask = label != -1
        target = np.where(valid_mask, label, 0).astype(np.float32)

        return {
            "chip_id": chip_id,
            "x": torch.from_numpy(s1),
            "y": torch.from_numpy(target),
            "valid_mask": torch.from_numpy(valid_mask),
        }
