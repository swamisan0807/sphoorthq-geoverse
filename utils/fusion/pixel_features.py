"""Per-pixel feature vector shared by the classical model (notebook 04) and
the quantum kernel (notebook 05) - keeping the feature set identical across
both is what makes the classical-vs-quantum comparison in notebook 06 fair.
"""

import numpy as np

from utils.ai.classic.sen1floods11_dataset import (
    chip_id_from_s1_filename,
    normalize_s1,
    read_label,
    read_s1,
    s1_nodata_mask,
)
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


def sample_balanced_pixels_across_chips(
    pairs: list[tuple[str, str]], n_pixels: int, seed: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Draws n_pixels (half water, half not-water) spread across as many
    distinct chips of `pairs` as needed - at most one water pixel and one
    not-water pixel taken per chip - instead of every pixel coming from a
    single chip.

    This exists for the quantum kernel notebooks (05, 06): a quantum kernel
    Gram matrix is O(n^2) circuit evaluations, so n_pixels has to stay small
    (tens, not the millions of pixels notebook 04's classical RF trains on),
    but a small sample still shouldn't be *one chip's* pixels only - that's
    a single scene's speckle/texture/incidence-angle characteristics, not
    the dataset's. Walking distinct chips for a handful of pixels each gives
    the same whole-dataset chip diversity notebook 04 gets from its
    many-chip `build_pixel_dataset` pool, just spread thin enough to fit the
    quantum kernel's circuit budget.

    Returns (x, y, chip_ids) - chip_ids records which chips actually
    contributed a pixel (for logging/reproducibility), one entry per chip
    that contributed at least one pixel.
    """
    rng = np.random.default_rng(seed)
    n_each = n_pixels // 2
    chip_order = rng.permutation(len(pairs))

    xs, ys, chip_ids = [], [], []
    water_taken = land_taken = 0
    for idx in chip_order:
        if water_taken >= n_each and land_taken >= n_each:
            break
        s1_filename, _ = pairs[idx]
        chip_id = chip_id_from_s1_filename(s1_filename)
        s1 = read_s1(chip_id)
        label = read_label(chip_id)
        cube = build_feature_cube(s1)
        x, y = cube_to_pixel_table(cube, label, raw_s1=s1)

        water_idx = np.where(y == 1)[0]
        land_idx = np.where(y == 0)[0]
        used_chip = False
        if water_taken < n_each and len(water_idx):
            pick = rng.choice(water_idx, size=1)
            xs.append(x[pick])
            ys.append(y[pick])
            water_taken += 1
            used_chip = True
        if land_taken < n_each and len(land_idx):
            pick = rng.choice(land_idx, size=1)
            xs.append(x[pick])
            ys.append(y[pick])
            land_taken += 1
            used_chip = True
        if used_chip:
            chip_ids.append(chip_id)

    if water_taken < n_each or land_taken < n_each:
        raise ValueError(
            f"only found {water_taken} water / {land_taken} not-water pixels "
            f"across all {len(pairs)} chips - need {n_each} of each"
        )

    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    perm = rng.permutation(len(y_all))
    return x_all[perm], y_all[perm], chip_ids
