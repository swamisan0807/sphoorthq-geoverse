"""Pluggable segmentation objective + metric definitions.

Satisfies "definition of segmentation objectives and performance criteria" -
new targets (flood, land-cover, built-up...) register here with their class
schema and metric set, instead of being hardcoded into the model or the API.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion


@dataclass
class Objective:
    name: str
    classes: dict[int, str]  # class id -> label
    primary_metric: str
    metrics: list[str]


OBJECTIVES: dict[str, Objective] = {
    "flood-segmentation": Objective(
        name="flood-segmentation",
        classes={0: "not_water", 1: "water"},
        primary_metric="iou",
        metrics=["iou", "f1", "precision", "recall", "boundary_f1"],
    ),
    "landcover-segmentation": Objective(
        name="landcover-segmentation",
        classes={
            0: "tree_cover",
            1: "shrubland",
            2: "grassland",
            3: "cropland",
            4: "built_up",
            5: "bare_sparse_vegetation",
            6: "water",
            7: "wetland",
        },
        primary_metric="mean_iou",
        metrics=["mean_iou", "per_class_iou", "kappa", "overall_accuracy"],
    ),
}


def get_objective(name: str) -> Objective:
    if name not in OBJECTIVES:
        raise KeyError(f"unknown objective '{name}', available: {list(OBJECTIVES)}")
    return OBJECTIVES[name]


def _confusion_counts(pred: np.ndarray, target: np.ndarray, positive: int = 1) -> tuple[int, int, int, int]:
    tp = int(np.sum((pred == positive) & (target == positive)))
    fp = int(np.sum((pred == positive) & (target != positive)))
    fn = int(np.sum((pred != positive) & (target == positive)))
    tn = int(np.sum((pred != positive) & (target != positive)))
    return tp, fp, fn, tn


def iou(pred: np.ndarray, target: np.ndarray, positive: int = 1) -> float:
    tp, fp, fn, _ = _confusion_counts(pred, target, positive)
    denom = tp + fp + fn
    return tp / denom if denom else 1.0


def f1(pred: np.ndarray, target: np.ndarray, positive: int = 1) -> float:
    tp, fp, fn, _ = _confusion_counts(pred, target, positive)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def precision(pred: np.ndarray, target: np.ndarray, positive: int = 1) -> float:
    tp, fp, _, _ = _confusion_counts(pred, target, positive)
    return tp / (tp + fp) if (tp + fp) else 1.0


def recall(pred: np.ndarray, target: np.ndarray, positive: int = 1) -> float:
    tp, _, fn, _ = _confusion_counts(pred, target, positive)
    return tp / (tp + fn) if (tp + fn) else 1.0


def boundary_f1(pred: np.ndarray, target: np.ndarray, positive: int = 1, tolerance_px: int = 2) -> float:
    """F1 on object boundaries only (dilated ring), rewards edge accuracy that
    pixel-wise IoU can hide."""

    def boundary(mask: np.ndarray) -> np.ndarray:
        m = mask == positive
        return m & ~binary_erosion(m, iterations=1)

    pred_b = binary_dilation(boundary(pred), iterations=tolerance_px)
    target_b = binary_dilation(boundary(target), iterations=tolerance_px)

    tp = int(np.sum(boundary(pred) & target_b))
    fp = int(np.sum(boundary(pred) & ~target_b))
    fn = int(np.sum(boundary(target) & ~pred_b))

    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def mean_iou(pred: np.ndarray, target: np.ndarray, num_classes: int) -> float:
    ious = [iou(pred, target, positive=c) for c in range(num_classes)]
    return float(np.mean(ious))


METRIC_FNS: dict[str, Callable] = {
    "iou": iou,
    "f1": f1,
    "precision": precision,
    "recall": recall,
    "boundary_f1": boundary_f1,
}


def evaluate(objective_name: str, pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    objective = get_objective(objective_name)
    results = {}
    for metric_name in objective.metrics:
        if metric_name in METRIC_FNS:
            results[metric_name] = METRIC_FNS[metric_name](pred, target)
        elif metric_name == "mean_iou":
            results[metric_name] = mean_iou(pred, target, len(objective.classes))
    return results
