"""SARBake reference masks and the scoring built on them (utils/mrf/sarbake.py).

The scoring tests use tiny arrays. The data tests run on the real chip and
mask files and are skipped when the SARBake masks are not on this machine.
"""

import numpy as np
import pytest

from utils.mrf import sarbake
from utils.mrf.mstar_scenes import CATALOG, MSTAR_DIR, _load, calibrate_magnitude

_CHIP_ID = "t72_132_test_1"
_needs_masks = pytest.mark.skipif(
    sarbake.load_mask(_CHIP_ID) is None or not (MSTAR_DIR / CATALOG[_CHIP_ID]).exists(),
    reason="SARBake masks or MSTAR chips not available on this machine",
)


def test_score_binary_counts_only_the_target_class():
    mask = np.array([[1, 1, 0], [2, 2, 0], [0, 0, 0]])
    perfect = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]])
    scores = sarbake.score_binary(perfect, mask)
    assert scores["iou"] == 1.0 and scores["dice"] == 1.0 and scores["accuracy"] == 1.0

    half = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    scores = sarbake.score_binary(half, mask)
    assert scores["iou"] == pytest.approx(0.5)
    assert scores["precision"] == 1.0
    assert scores["recall"] == pytest.approx(0.5)


def test_score_three_class_scores_each_class():
    mask = np.array([[0, 1], [2, 2]])
    assert sarbake.score_three_class(mask, mask)["mean_iou"] == 1.0
    wrong_shadow = np.array([[0, 1], [0, 0]])
    scores = sarbake.score_three_class(wrong_shadow, mask)
    assert scores["iou_target"] == 1.0
    assert scores["iou_shadow"] == 0.0
    assert scores["accuracy"] == pytest.approx(0.5)


def test_missing_mask_returns_none():
    assert sarbake.load_mask("slicy_15deg_1") is None  # SLICY has no SARBake masks
    assert sarbake.load_mask("not-a-chip") is None


@_needs_masks
def test_mask_matches_the_chip_it_belongs_to():
    mask = sarbake.load_mask(_CHIP_ID)
    assert mask.shape == (128, 128)
    assert set(np.unique(mask)) <= {0, 1, 2}
    assert (mask == sarbake.TARGET).sum() > 0 and (mask == sarbake.SHADOW).sum() > 0

    # the reference only means something if it lines up with the real image:
    # vehicle returns well above the background, shadow below it
    db = 20 * np.log10(_load(_CHIP_ID).magnitude.astype(np.float64) + 1e-6)
    background = np.median(db[mask == sarbake.BACKGROUND])
    assert np.median(db[mask == sarbake.TARGET]) - background > 5.0
    assert np.median(db[mask == sarbake.SHADOW]) - background < -3.0


@_needs_masks
def test_calibrated_pipeline_segments_the_vehicle_far_better_than_raw_brightness():
    from utils.mrf.energy import build_patch_energy, solve_graph_cut
    from utils.mrf.mstar_scenes import target_probability

    mask = sarbake.load_mask(_CHIP_ID)
    magnitude = _load(_CHIP_ID).magnitude
    sl = (slice(40, 88), slice(40, 88))
    target = mask[sl]

    calibrated = calibrate_magnitude(magnitude)[sl]
    labels = solve_graph_cut(build_patch_energy(target_probability(calibrated), calibrated)).reshape(target.shape)
    calibrated_iou = sarbake.score_binary(labels, target)["iou"]

    log_m = np.log1p(magnitude.astype(np.float64))
    raw = ((log_m - log_m.min()) / (log_m.max() - log_m.min()))[sl]
    raw_labels = solve_graph_cut(build_patch_energy(raw, raw)).reshape(target.shape)
    raw_iou = sarbake.score_binary(raw_labels, target)["iou"]

    assert calibrated_iou > 0.4
    assert calibrated_iou > raw_iou + 0.3
