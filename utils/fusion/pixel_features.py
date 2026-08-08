"""Per-pixel feature vector shared by the classical model (notebook 04) and
the quantum kernel (notebook 05) - keeping the feature set identical across
both is what makes the classical-vs-quantum comparison in notebook 06 fair.
"""

import numpy as np

from utils.ai.classic.sen1floods11_dataset import normalize_s1, s1_nodata_mask
from utils.fusion.feature_engineering import local_statistics, polarimetric_difference, polarimetric_ratio

FEATURE_NAMES = ["vv", "vh", "ratio", "difference", "vv_local_mean", "vv_local_std"]


def build_feature_cube(s1_db: np.ndarray) -> np.ndarray:
    """s1_db: (2, H, W) raw VV/VH in dB. Returns (6, H, W) feature cube,
    channel order matching FEATURE_NAMES, each channel roughly in [0, 1].

    Raw VV/VH is sanitized (NaN/Inf -> 0.0) BEFORE deriving ratio/difference/
    local-texture features, not after - local_statistics' uniform_filter
    would otherwise spread a single NaN pixel across its whole window before
    any later cleanup had a chance to run. Callers that need to exclude
    fabricated-from-nodata pixels entirely (not just prevent NaN) should
    combine label validity with utils.ai.classic.sen1floods11_dataset.s1_nodata_mask(s1_db) -
    see cube_to_pixel_table.
    """
    vv = np.nan_to_num(s1_db[0], nan=0.0, posinf=0.0, neginf=0.0)
    vh = np.nan_to_num(s1_db[1], nan=0.0, posinf=0.0, neginf=0.0)

    ratio = polarimetric_ratio(vv, vh)
    diff = polarimetric_difference(vv, vh)
    stats = local_statistics(vv, window=7)

    vv_n = normalize_s1(vv[None])[0]
    vh_n = normalize_s1(vh[None])[0]
    ratio_n = (np.clip(ratio, 0, 5) / 5).astype(np.float32)
    diff_n = ((np.clip(diff, -10, 10) + 10) / 20).astype(np.float32)
    mean_n = normalize_s1(stats["mean"][None])[0]
    std_n = (np.clip(stats["std"], 0, 5) / 5).astype(np.float32)

    return np.stack([vv_n, vh_n, ratio_n, diff_n, mean_n, std_n], axis=0)


def cube_to_pixel_table(
    feature_cube: np.ndarray, label: np.ndarray, raw_s1: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Flattens to one row per valid pixel: label != -1, AND (if raw_s1 is
    given) not a S1 nodata pixel (see utils.ai.classic.sen1floods11_dataset -
    ~9% of chips have NaN S1 pixels that do NOT always coincide with a -1
    label, confirmed by direct inspection; skipping raw_s1 silently trains
    on fabricated-from-NaN feature rows)."""
    valid = label != -1
    if raw_s1 is not None:
        valid = valid & ~s1_nodata_mask(raw_s1)
    channels, h, w = feature_cube.shape
    x = feature_cube.reshape(channels, h * w).T[valid.reshape(-1)]
    y = label.reshape(-1)[valid.reshape(-1)]
    return x, y
