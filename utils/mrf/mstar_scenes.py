"""Real MSTAR SAR vehicle-target chips as an unary/intensity source for the
same utils.mrf.energy pipeline used elsewhere in this project (see that
module's docstring) - this project's segmentation target, replacing
sen1floods11 flood segmentation.

Honest about what this is NOT:
  - the MSTAR public target-chip release has no per-pixel segmentation
    ground truth. Each chip's Phoenix header gives real, per-chip metadata
    (target type, serial number, azimuth, depression angle - see
    utils.mstar.reader) but not a target/background mask, so - exactly like
    utils.mrf.sentinel1_raw and the /api/mrf/upload path - the chip's own
    normalized magnitude is used directly as unary/intensity evidence, and
    results are reported as energy only, honestly, with metrics=None.
  - no radiometric calibration beyond what MSTAR's own SensorCalibrationFactor
    already applied when the magnitude data was produced.

What IS real: actual X-band SAR magnitude/phase measured by Sandia National
Lab's radar (see each chip's Phoenix header - CollectionDate, Site,
SensorName), decoded directly from the on-disk chip files.
"""

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

from utils.core.paths import DATASETS_DIR
from utils.mstar.reader import MstarChip, read_mstar_chip

MSTAR_DIR = DATASETS_DIR / "raw" / "mstar" / "MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY"

# Small, explicit catalog of locally-cached real chips - not an arbitrary
# user-supplied filesystem path (same reasoning as utils.mrf.sentinel1_raw's
# CATALOG: an endpoint that reads whatever path a caller names is a real
# path-traversal risk regardless of this being a single-user dev/demo app).
# Two different real azimuth views per real target serial number, for every
# serial in the standard MSTAR public release (BMP2: 3 serials, BTR70: 1,
# T72: 3), at both 17-deg-train/15-deg-test depressions, plus SLICY at both
# collected depressions - 30 chips total, all real files already on disk
# (the full ~3,000-chip release is present locally too; this is a curated,
# vetted subset for the UI, not a limit on what's downloaded).
CATALOG: dict[str, str] = {
    "bmp2_9563_test_1": "TARGETS/TEST/15_DEG/BMP2/SN_9563/HB03333.000",
    "bmp2_9563_test_2": "TARGETS/TEST/15_DEG/BMP2/SN_9563/HB03380.000",
    "bmp2_9563_train_1": "TARGETS/TRAIN/17_DEG/BMP2/SN_9563/HB03787.000",
    "bmp2_9563_train_2": "TARGETS/TRAIN/17_DEG/BMP2/SN_9563/HB03820.000",
    "bmp2_9566_test_1": "TARGETS/TEST/15_DEG/BMP2/SN_9566/HB03333.001",
    "bmp2_9566_test_2": "TARGETS/TEST/15_DEG/BMP2/SN_9566/HB03379.001",
    "bmp2_9566_train_1": "TARGETS/TRAIN/17_DEG/BMP2/SN_9566/HB03787.001",
    "bmp2_9566_train_2": "TARGETS/TRAIN/17_DEG/BMP2/SN_9566/HB03820.001",
    "bmp2_c21_test_1": "TARGETS/TEST/15_DEG/BMP2/SN_C21/HB03333.002",
    "bmp2_c21_test_2": "TARGETS/TEST/15_DEG/BMP2/SN_C21/HB03379.002",
    "bmp2_c21_train_1": "TARGETS/TRAIN/17_DEG/BMP2/SN_C21/HB03787.002",
    "bmp2_c21_train_2": "TARGETS/TRAIN/17_DEG/BMP2/SN_C21/HB03820.002",
    "btr70_c71_test_1": "TARGETS/TEST/15_DEG/BTR70/SN_C71/HB03333.004",
    "btr70_c71_test_2": "TARGETS/TEST/15_DEG/BTR70/SN_C71/HB03379.004",
    "btr70_c71_train_1": "TARGETS/TRAIN/17_DEG/BTR70/SN_C71/HB03787.004",
    "btr70_c71_train_2": "TARGETS/TRAIN/17_DEG/BTR70/SN_C71/HB03820.004",
    "t72_132_test_1": "TARGETS/TEST/15_DEG/T72/SN_132/HB03333.015",
    "t72_132_test_2": "TARGETS/TEST/15_DEG/T72/SN_132/HB03379.015",
    "t72_132_train_1": "TARGETS/TRAIN/17_DEG/T72/SN_132/HB03787.015",
    "t72_132_train_2": "TARGETS/TRAIN/17_DEG/T72/SN_132/HB03820.015",
    "t72_812_test_1": "TARGETS/TEST/15_DEG/T72/SN_812/HB03333.016",
    "t72_812_test_2": "TARGETS/TEST/15_DEG/T72/SN_812/HB03379.016",
    "t72_812_train_1": "TARGETS/TRAIN/17_DEG/T72/SN_812/HB03787.016",
    "t72_812_train_2": "TARGETS/TRAIN/17_DEG/T72/SN_812/HB03820.016",
    "t72_s7_test_1": "TARGETS/TEST/15_DEG/T72/SN_S7/HB03333.017",
    "t72_s7_test_2": "TARGETS/TEST/15_DEG/T72/SN_S7/HB03379.017",
    "t72_s7_train_1": "TARGETS/TRAIN/17_DEG/T72/SN_S7/HB03787.017",
    "t72_s7_train_2": "TARGETS/TRAIN/17_DEG/T72/SN_S7/HB03820.017",
    "slicy_15deg_1": "SLICY/15_DEG/HB14931.015",
    "slicy_15deg_2": "SLICY/15_DEG/HB14963.015",
    "slicy_30deg_1": "SLICY/30_DEG/HB15269.015",
    "slicy_30deg_2": "SLICY/30_DEG/HB15310.015",
}

_chip_cache: dict[str, MstarChip] = {}


def _load(chip_id: str) -> MstarChip:
    if chip_id not in CATALOG:
        raise ValueError(f"unknown chip '{chip_id}', available: {list(CATALOG)}")
    if chip_id not in _chip_cache:
        path = MSTAR_DIR / CATALOG[chip_id]
        if not path.exists():
            raise FileNotFoundError(f"chip file not found locally: {path}")
        _chip_cache[chip_id] = read_mstar_chip(path)
    return _chip_cache[chip_id]


def _bright_center(chip_id: str) -> tuple[int, int]:
    """(row, col) of the chip's real target peak - the brightest pixel
    within the real target blob, not the chip's geometric center and not a
    naive global argmax. MSTAR chips center the target roughly, but not
    exactly (real sensor/collection offsets), so the geometric center of a
    128x128 chip can land on plain background clutter (verified:
    row0=62,col0=62 on bmp2_9563_test_1 is genuinely background, intensity
    0.001-0.05).

    Two things checked directly, not assumed:
      - A naive "brightest pixel in the whole chip" is unsafe: the top 5%
        of pixels by intensity are real SAR speckle noise scattered across
        the ENTIRE 128x128 chip (row/col range both 0-127), not
        concentrated on the target.
      - The CENTROID of the real target blob (below) isn't quite enough
        either: it correctly lands in the target's general region, but a
        4x4/5x5 ROI centered there can still miss the actual peak return
        by a few pixels, which matters because the classical/quantum
        pairwise smoothness term can out-vote a moderate unary signal.
    So: connected-component labeling (scipy.ndimage.label) on a bright
    mask isolates the compact target blob (100-300+ contiguous pixels)
    from scattered speckle, then this returns the single brightest pixel
    WITHIN that blob - the strongest real unary signal available, not
    just "somewhere near the target"."""
    from scipy import ndimage

    m = _normalized_magnitude(chip_id)
    mask = m >= np.percentile(m, 97)
    labeled, n_components = ndimage.label(mask)
    if n_components == 0:
        h, w = m.shape
        return h // 2, w // 2
    sizes = ndimage.sum(mask, labeled, range(1, n_components + 1))
    largest_label = int(np.argmax(sizes)) + 1
    blob_only = np.where(labeled == largest_label, m, -1.0)
    peak_row, peak_col = np.unravel_index(np.argmax(blob_only), blob_only.shape)
    return int(peak_row), int(peak_col)


_PRESET_SEARCH_SIZE = 5  # window size used to search for preset ROIs - a good row0/col0 for any smaller size too


def _no_object_region(chip_id: str) -> tuple[int, int]:
    """(row, col) of a real, verified target-free window - THALES's own
    validation set explicitly includes "images without object (all pixels
    labelled 0)"; this finds that case for real within an already-real
    target chip (every chip has plenty of pure-clutter background outside
    the target blob) rather than needing a separate no-target dataset.
    Picks the window with the lowest peak intensity anywhere in the chip -
    genuinely no bright return at all, not just "not the brightest"."""
    m = _normalized_magnitude(chip_id)
    h, w = m.shape
    s = _PRESET_SEARCH_SIZE
    best_row, best_col, best_score = 0, 0, float("inf")
    for r0 in range(0, h - s, s):
        for c0 in range(0, w - s, s):
            score = m[r0 : r0 + s, c0 : c0 + s].max()
            if score < best_score:
                best_score, best_row, best_col = score, r0, c0
    return best_row, best_col


def _low_snr_region(chip_id: str) -> tuple[int, int]:
    """(row, col) of a real window at the target blob's edge - THALES's
    "low signal-to-noise ratio (object pixels intensity close to
    background pixels intensity)" case, found for real: the boundary of
    the actual target return (not its bright core, not pure background)
    is genuinely lower-contrast, since a real target's edge pixels are a
    partial/grazing radar return. Falls back to the bright center if no
    such boundary pixel is found (e.g. an unusually small or absent blob)."""
    from scipy import ndimage

    m = _normalized_magnitude(chip_id)
    h, w = m.shape
    s = _PRESET_SEARCH_SIZE
    mask = m >= np.percentile(m, 97)
    labeled, n_components = ndimage.label(mask)
    if n_components == 0:
        return _bright_center(chip_id)
    sizes = ndimage.sum(mask, labeled, range(1, n_components + 1))
    largest_label = int(np.argmax(sizes)) + 1
    blob_rows, blob_cols = np.where(labeled == largest_label)

    best = None
    for r, c in zip(blob_rows.tolist(), blob_cols.tolist()):
        r0, c0 = max(0, min(r - s // 2, h - s)), max(0, min(c - s // 2, w - s))
        win = m[r0 : r0 + s, c0 : c0 + s]
        if 0.2 <= win.max() <= 0.5:  # a real but weak target return, not the bright core and not pure background
            spread = float(win.max() - win.min())
            if best is None or spread < best[0]:
                best = (spread, r0, c0)
    if best is None:
        return _bright_center(chip_id)
    return best[1], best[2]


def chip_info(chip_id: str) -> dict:
    """Real per-chip metadata straight from the file's own Phoenix header -
    target type/serial/angles, never inferred from the catalog key. Also
    includes three real, verified ROI presets: the target's peak return
    (bright_row/col), a genuinely target-free background window
    (no_object_row/col), and a real low-contrast target-edge window
    (low_snr_row/col) - THALES's own validation set explicitly calls for
    all three cases (normal, no-object, low-SNR), found here within the
    real chip data already on disk rather than needing a separate dataset."""
    chip = _load(chip_id)
    h, w = chip.magnitude.shape
    bright_row, bright_col = _bright_center(chip_id)
    no_object_row, no_object_col = _no_object_region(chip_id)
    low_snr_row, low_snr_col = _low_snr_region(chip_id)
    return {
        "chip_id": chip_id,
        "target_type": chip.target_type,
        "target_serial": chip.target_serial,
        "depression_deg": chip.depression_deg,
        "azimuth_deg": round(chip.azimuth_deg, 1),
        "height": h,
        "width": w,
        "bright_row": bright_row,
        "bright_col": bright_col,
        "no_object_row": no_object_row,
        "no_object_col": no_object_col,
        "low_snr_row": low_snr_row,
        "low_snr_col": low_snr_col,
    }


def _normalized_magnitude(chip_id: str) -> np.ndarray:
    chip = _load(chip_id)
    m = chip.magnitude.astype(np.float64)
    log_m = np.log1p(m)
    lo, hi = log_m.min(), log_m.max()
    if hi <= lo:
        return np.zeros_like(log_m)
    return (log_m - lo) / (hi - lo)


def background_median(chip_id: str) -> float:
    """This chip's real median normalized intensity - the grounded
    "typical background clutter" reference utils.mrf.multilabel_energy's
    3-class (background/target/shadow) unary term uses, computed from the
    actual chip rather than an assumed constant (real per-chip medians
    checked directly: 0.04-0.06 across several chips, not all the same)."""
    return float(np.median(_normalized_magnitude(chip_id)))


def chip_preview_png_base64(chip_id: str) -> str:
    """The full chip's real magnitude, rendered for display - the same
    dB-scaled, percentile-stretched grayscale render used throughout this
    project to visually verify a decoded chip actually looks like a SAR
    vehicle-target image (bright return + shadow in real clutter), not a
    fabricated preview. Base64-encoded PNG so it travels inline in the API
    response instead of needing a second authenticated image request."""
    chip = _load(chip_id)
    m = chip.magnitude.astype(np.float64)
    db = 20 * np.log10(m + 1e-6)
    lo, hi = np.percentile(db, 1), np.percentile(db, 99.5)
    norm = np.clip((db - lo) / max(hi - lo, 1e-9), 0, 1)
    img = (norm * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def read_intensity_window(chip_id: str, row0: int, col0: int, size: int) -> np.ndarray:
    """Reads a size x size window of real MSTAR magnitude (log-scaled,
    normalized to [0, 1] over the full chip so a small ROI keeps the same
    contrast reference as the whole 128x128 chip), the same real intensity
    signal utils.mrf.energy.build_patch_energy needs for its unary/pairwise
    terms."""
    full = _normalized_magnitude(chip_id)
    h, w = full.shape
    if row0 < 0 or col0 < 0 or row0 + size > h or col0 + size > w:
        raise ValueError(f"window (row0={row0}, col0={col0}, size={size}) is out of bounds for a {h}x{w} chip")
    return full[row0 : row0 + size, col0 : col0 + size]


# --- Decibel-scaled intensity, calibrated against the SARBake masks ----------
# The log1p/min-max intensity above is dominated by each chip's few brightest
# spikes, so almost every pixel ends up near 0.05 and a vehicle's return looks
# like background. Scored against the SARBake masks (utils/mrf/sarbake.py) it
# labels only about 13 of roughly 320 target pixels. The functions below
# measure each pixel in decibels relative to its own chip's background (robust
# z-score: median and MAD of the dB image), which puts vehicle returns and
# shadows well away from clutter. The constants were chosen on the MSTAR TRAIN
# split and are scored on the TEST split (scripts/sarbake_eval.py).

Z_LO = -4.0  # z-score mapped to intensity 0
Z_SPAN = 12.0  # intensity = (z - Z_LO) / Z_SPAN, clipped to [0, 1]
BINARY_Z_THRESHOLD = 0.8  # two-class: P(target) = sigmoid(z - this)
THREE_CLASS_TARGET_Z = 1.5  # three-class label means, in z units; background is 0 by construction
THREE_CLASS_SHADOW_Z = -1.0
THREE_CLASS_Z_SCALE = 3.0  # distance (in z) that costs one unit of unary energy


def z_to_intensity(z):
    return np.clip((np.asarray(z, dtype=np.float64) - Z_LO) / Z_SPAN, 0.0, 1.0)


def intensity_to_z(intensity):
    return np.asarray(intensity, dtype=np.float64) * Z_SPAN + Z_LO


def calibrate_magnitude(magnitude: np.ndarray) -> np.ndarray:
    """Calibrated intensity in [0, 1] for a raw MSTAR magnitude plane."""
    db = 20 * np.log10(magnitude.astype(np.float64) + 1e-6)
    med = np.median(db)
    mad = np.median(np.abs(db - med)) * 1.4826 + 1e-9
    return z_to_intensity((db - med) / mad)


def calibrated_intensity(chip_id: str) -> np.ndarray:
    return calibrate_magnitude(_load(chip_id).magnitude)


def read_calibrated_window(chip_id: str, row0: int, col0: int, size: int) -> np.ndarray:
    full = calibrated_intensity(chip_id)
    h, w = full.shape
    if row0 < 0 or col0 < 0 or row0 + size > h or col0 + size > w:
        raise ValueError(f"window (row0={row0}, col0={col0}, size={size}) is out of bounds for a {h}x{w} chip")
    return full[row0 : row0 + size, col0 : col0 + size]


def target_probability(intensity: np.ndarray) -> np.ndarray:
    """Per-pixel P(target) for the two-class model, from calibrated intensity."""
    return 1.0 / (1.0 + np.exp(-(intensity_to_z(intensity) - BINARY_Z_THRESHOLD)))
