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
import os
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from utils.core.paths import PROCESSED_DIR, RAW_DIR

SEN1FLOODS11_ROOT = RAW_DIR / "sen1floods11"
HAND_LABELED_DIR = SEN1FLOODS11_ROOT / "data" / "flood_events" / "HandLabeled"
SPLITS_DIR = SEN1FLOODS11_ROOT / "splits" / "flood_handlabeled"

# Source of truth is s3://sphoorthq-geoverse/datasets/raw/sen1floods11/ (see
# notebooks/01_ingest.ipynb) - datasets/raw/ is a working cache under it, not
# itself authoritative. The notebook bulk-downloads everything up front for
# training; the live API (apps/api/routers/catalog.py, inference.py) instead
# reads through _ensure_local() below, which fetches only the one file a
# given request actually needs, on first access - same "S3 primary, local
# disk lazily caches it" pattern utils/registry/model_registry.py already
# uses for model files, so Catalog/Inference work right after datasets/raw/
# is deleted (or never populated) without requiring a manual bulk re-ingest.
DATA_S3_BUCKET = os.environ.get("DATA_S3_BUCKET", "sphoorthq-geoverse")
DATA_S3_REGION = os.environ.get("DATA_S3_REGION", "us-east-1")
_DATA_S3_PREFIX = "datasets/raw/sen1floods11/"

_s3_connector = None


def _get_s3_connector():
    global _s3_connector
    if _s3_connector is None:
        from utils.ingestion.s3_connector import S3Connector

        _s3_connector = S3Connector(DATA_S3_BUCKET, region_name=DATA_S3_REGION)
    return _s3_connector


def _ensure_local(local_path: Path, s3_key: str) -> Path:
    """Returns local_path, fetching it from S3 (bucket DATA_S3_BUCKET, this
    exact key) first if it's not already on disk. Lets any S3/credentials
    error propagate as-is - callers already handle a missing/unreadable file
    (FileNotFoundError from open(), rasterio's own error from a bad/absent
    .tif), so a failed S3 fetch surfaces the same way a missing local file
    always did, not a new silent failure mode."""
    if local_path.exists():
        return local_path
    _get_s3_connector().download_object(s3_key, local_path)
    return local_path


def _ensure_local_raw(local_path: Path) -> Path:
    """_ensure_local for anything under SEN1FLOODS11_ROOT - its path relative
    to that root is exactly the S3 key relative to _DATA_S3_PREFIX, since
    datasets/raw/sen1floods11/ is a one-to-one mirror of that prefix (see
    notebooks/01_ingest.ipynb)."""
    rel_key = str(local_path.relative_to(SEN1FLOODS11_ROOT)).replace("\\", "/")
    return _ensure_local(local_path, _DATA_S3_PREFIX + rel_key)

# This project's own split, distinct from the one shipped in SPLITS_DIR above
# (see notebooks/03_feature_engineering.ipynb "Custom train/test split") -
# an event-level holdout (entire flood events kept out of train/valid, not
# just individual chips) rather than the shipped split's per-chip-random
# partition, which lets same-event chips (highly spatially/temporally
# correlated) land in both train and test. Lives under datasets/processed/
# (a derived artifact this project generated), not datasets/raw/ (a mirror
# of the original untouched source).
CUSTOM_SPLITS_DIR = PROCESSED_DIR / "splits"

S1_DB_MIN, S1_DB_MAX = -50.0, 1.0  # fixed physical bounds for Sentinel-1 sigma0 dB


def load_split(split: str) -> list[tuple[str, str]]:
    """split: 'train' | 'valid' | 'test'. Returns list of (s1_filename, label_filename).
    This is the default split shipped with sen1floods11 itself - see
    load_custom_split() below for this project's own event-holdout split."""
    path = _ensure_local_raw(SPLITS_DIR / f"flood_{split}_data.csv")
    with open(path, newline="", encoding="utf-8") as f:
        return [(row[0], row[1]) for row in csv.reader(f) if row]


def load_custom_split(split: str) -> list[tuple[str, str]]:
    """Same (s1_filename, label_filename) shape as load_split(), but reads
    this project's own event-holdout split instead of the default per-chip
    one shipped with sen1floods11 - see notebooks/03_feature_engineering.ipynb
    "Custom train/test split" for how it's generated and why. Same S3
    read-through cache as load_split()/read_s1()/read_label() - these CSVs
    already live at s3://{DATA_S3_BUCKET}/datasets/processed/splits/ (that
    upload is a one-off from when notebook 03 first generated them, not
    something download_object() re-derives), so a missing local copy is
    fetched from there instead of requiring notebook 03 to be re-run."""
    file_name = f"flood_custom_{split}_data.csv"
    path = _ensure_local(CUSTOM_SPLITS_DIR / file_name, f"datasets/processed/splits/{file_name}")
    with open(path, newline="", encoding="utf-8") as f:
        return [(row[0], row[1]) for row in csv.reader(f) if row]


def chip_id_from_s1_filename(s1_filename: str) -> str:
    return s1_filename.removesuffix("_S1Hand.tif")


def read_s1(chip_id: str) -> np.ndarray:
    path = _ensure_local_raw(HAND_LABELED_DIR / "S1Hand" / f"{chip_id}_S1Hand.tif")
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)  # (2, H, W) VV, VH


def read_label(chip_id: str) -> np.ndarray:
    path = _ensure_local_raw(HAND_LABELED_DIR / "LabelHand" / f"{chip_id}_LabelHand.tif")
    with rasterio.open(path) as src:
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
