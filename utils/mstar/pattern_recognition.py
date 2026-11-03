"""Real target-type recognition on real MSTAR chips - this project's
chosen, disclosed interpretation of the THALES brief's undefined "Pattern
detection" bullet (the brief lists it with no elaboration: "Pattern
detection - ..."). Grounded in actual MSTAR literature the brief itself
cites (Mahalanobis et al.'s correlation-filter target recognition,
Bhanu & Jones's MSTAR target-variant recognition) rather than an invented
task: given a chip, recognize WHICH real vehicle type it is (BMP2 / BTR70
/ T72) from its real SAR signature.

Buildable honestly, unlike the shadow-class unary term, because real
ground truth exists here: every chip's own Phoenix header already states
its true target type (utils.mstar.reader). Templates are built from the
REAL MSTAR train split (17-deg depression, the standard MSTAR protocol),
evaluated against the REAL test split (15-deg) - not the 32-chip curated
UI catalog, the full local collection (utils/mrf/mstar_scenes.MSTAR_DIR),
so accuracy numbers are real, not cherry-picked from a handful of chips.

Method: simple, classic, and disclosed - a correlation-filter-style
nearest-template match (the same family of approach as the MACH filter
literature above), not a deep model. Real limitation, stated plainly: this
is far simpler than modern MSTAR ATR classifiers (which reach ~95%+ with
CNNs); this is deliberately a transparent, auditable baseline, not a
claim of state-of-the-art accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from utils.mrf.mstar_scenes import MSTAR_DIR
from utils.mstar.reader import read_mstar_chip

# The three real vehicle classes - SLICY is a calibration target, not a
# vehicle, so it's excluded from recognition (nothing to "recognize" it
# against - it's always correctly identifiable by its very different,
# much smaller chip size, 54x54 vs 128x128).
TARGET_CLASSES = ["bmp2_tank", "btr70_transport", "t72_tank"]
_CLASS_DIR_NAME = {"bmp2_tank": "BMP2", "btr70_transport": "BTR70", "t72_tank": "T72"}

DEFAULT_PATCH_SIZE = 32


def _normalize(m: np.ndarray) -> np.ndarray:
    log_m = np.log1p(m.astype(np.float64))
    lo, hi = log_m.min(), log_m.max()
    if hi <= lo:
        return np.zeros_like(log_m)
    return (log_m - lo) / (hi - lo)


def extract_target_patch(magnitude: np.ndarray, patch_size: int) -> np.ndarray | None:
    """Same real target-localization logic as
    utils.mrf.mstar_scenes._bright_center: connected-component analysis on
    a bright mask isolates the real target blob (not scattered speckle),
    then crops a patch_size x patch_size window centered on the blob's
    peak pixel. Returns None if no blob is found or the chip is too small
    for the requested patch (e.g. a corrupt/degenerate file)."""
    m = _normalize(magnitude)
    h, w = m.shape
    if h < patch_size or w < patch_size:
        return None
    mask = m >= np.percentile(m, 97)
    labeled, n_components = ndimage.label(mask)
    if n_components == 0:
        return None
    sizes = ndimage.sum(mask, labeled, range(1, n_components + 1))
    largest_label = int(np.argmax(sizes)) + 1
    blob_only = np.where(labeled == largest_label, m, -1.0)
    peak_row, peak_col = np.unravel_index(np.argmax(blob_only), blob_only.shape)

    half = patch_size // 2
    r0 = min(max(0, int(peak_row) - half), h - patch_size)
    c0 = min(max(0, int(peak_col) - half), w - patch_size)
    return m[r0 : r0 + patch_size, c0 : c0 + patch_size]


def _scan_split_files(split: str) -> dict[str, list[Path]]:
    """split: 'TRAIN' (17_DEG) or 'TEST' (15_DEG) - the real, standard
    MSTAR protocol depression angles. Scans the actual downloaded
    collection directly (not the curated 32-chip UI catalog)."""
    depression = "17_DEG" if split == "TRAIN" else "15_DEG"
    base = MSTAR_DIR / "TARGETS" / split / depression
    out: dict[str, list[Path]] = {}
    for class_name in TARGET_CLASSES:
        class_dir = base / _CLASS_DIR_NAME[class_name]
        out[class_name] = sorted(class_dir.rglob("*")) if class_dir.is_dir() else []
        out[class_name] = [p for p in out[class_name] if p.is_file()]
    return out


@dataclass
class Templates:
    patch_size: int
    templates: dict[str, np.ndarray]
    n_train_samples: dict[str, int]


_templates_cache: dict[int, Templates] = {}


def build_templates(patch_size: int = DEFAULT_PATCH_SIZE, max_per_class: int | None = None) -> Templates:
    """Builds one real reference signature per class: the pixel-wise mean
    of every real train chip's extracted target patch. Cached per
    patch_size (idempotent, cheap to call repeatedly - see
    apps/api/routers/mstar.py's /pattern endpoint)."""
    cache_key = patch_size if max_per_class is None else -patch_size * 100000 - max_per_class
    if cache_key in _templates_cache:
        return _templates_cache[cache_key]

    files_by_class = _scan_split_files("TRAIN")
    templates: dict[str, np.ndarray] = {}
    n_samples: dict[str, int] = {}
    for class_name, files in files_by_class.items():
        if max_per_class is not None:
            files = files[:max_per_class]
        patches = []
        for f in files:
            try:
                chip = read_mstar_chip(f)
            except Exception:
                continue  # a real corrupt/unreadable file - skip rather than fail the whole build
            patch = extract_target_patch(chip.magnitude, patch_size)
            if patch is not None:
                patches.append(patch)
        if not patches:
            raise RuntimeError(f"no usable real train chips found for class '{class_name}' - check {MSTAR_DIR}")
        templates[class_name] = np.mean(patches, axis=0)
        n_samples[class_name] = len(patches)

    result = Templates(patch_size=patch_size, templates=templates, n_train_samples=n_samples)
    _templates_cache[cache_key] = result
    return result


def classify_patch(patch: np.ndarray, templates: Templates) -> tuple[str, dict[str, float]]:
    """Pearson correlation of the query patch against each class's real
    template - the classic correlation-filter matching approach. Returns
    (predicted_class, {class: correlation_score})."""
    flat_query = patch.flatten()
    scores: dict[str, float] = {}
    for class_name, template in templates.templates.items():
        flat_template = template.flatten()
        if np.std(flat_query) < 1e-9 or np.std(flat_template) < 1e-9:
            scores[class_name] = 0.0
        else:
            scores[class_name] = float(np.corrcoef(flat_query, flat_template)[0, 1])
    predicted = max(scores, key=lambda k: scores[k])
    return predicted, scores


@dataclass
class EvaluationResult:
    accuracy: float
    n_test_samples: int
    confusion: dict[str, dict[str, int]]  # confusion[true_class][predicted_class] = count
    n_train_samples: dict[str, int]


_evaluation_cache: dict[int, EvaluationResult] = {}


def evaluate(patch_size: int = DEFAULT_PATCH_SIZE, max_per_class: int | None = None) -> EvaluationResult:
    """Real accuracy on the real MSTAR test split (15-deg depression) -
    every test chip usable for patch extraction, not a cherry-picked
    handful. Cached (this is expensive enough - ~1300+ real chip decodes -
    that recomputing per request would be wasteful, not that the result
    changes)."""
    cache_key = patch_size if max_per_class is None else -patch_size * 100000 - max_per_class
    if cache_key in _evaluation_cache:
        return _evaluation_cache[cache_key]

    templates = build_templates(patch_size, max_per_class)
    files_by_class = _scan_split_files("TEST")

    confusion: dict[str, dict[str, int]] = {c: {c2: 0 for c2 in TARGET_CLASSES} for c in TARGET_CLASSES}
    n_correct = 0
    n_total = 0
    for true_class, files in files_by_class.items():
        if max_per_class is not None:
            files = files[:max_per_class]
        for f in files:
            try:
                chip = read_mstar_chip(f)
            except Exception:
                continue
            patch = extract_target_patch(chip.magnitude, patch_size)
            if patch is None:
                continue
            predicted, _ = classify_patch(patch, templates)
            confusion[true_class][predicted] += 1
            n_total += 1
            if predicted == true_class:
                n_correct += 1

    result = EvaluationResult(
        accuracy=round(n_correct / n_total, 4) if n_total else 0.0,
        n_test_samples=n_total,
        confusion=confusion,
        n_train_samples=templates.n_train_samples,
    )
    _evaluation_cache[cache_key] = result
    return result
