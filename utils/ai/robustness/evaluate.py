"""Robustness benchmark harness: runs a model against clean + perturbed inputs,
writes a comparison report to datasets/reports/."""

import json
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from utils.ai.objectives.registry import evaluate as evaluate_objective
from utils.ai.robustness.perturbations import PERTURBATIONS
from utils.core.paths import REPORTS_DIR


def run_robustness_suite(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    cube: np.ndarray,
    label: np.ndarray,
    objective_name: str,
    sar_channel_indices: list[int],
    optical_channel_indices: list[int],
    model_id: str,
    valid_mask: np.ndarray | None = None,
) -> dict:
    """valid_mask: optional boolean array, same shape as label - pixels
    where this is False are excluded from every condition's evaluation.
    Defaults to label != -1 if not given, but callers whose input data can
    have its own nodata sentinel independent of the label (e.g. sen1floods11
    S1 chips can have NaN pixels that do NOT always coincide with a -1
    label - confirmed by direct inspection) should pass a mask that also
    accounts for that, or every condition's IoU/F1 silently treats no-data
    pixels as a confirmed negative class."""
    if valid_mask is None:
        valid_mask = label != -1

    results = {"model_id": model_id, "objective": objective_name, "conditions": {}}

    def eval_masked(pred: np.ndarray) -> dict:
        return evaluate_objective(objective_name, pred[valid_mask], label[valid_mask])

    clean_pred = predict_fn(cube)
    results["conditions"]["clean"] = eval_masked(clean_pred)

    for name, perturb_fn in PERTURBATIONS.items():
        idx = sar_channel_indices if "speckle" in name or "incidence" in name else optical_channel_indices
        try:
            perturbed_cube = perturb_fn(cube, idx)
            pred = predict_fn(perturbed_cube)
            results["conditions"][name] = eval_masked(pred)
        except Exception as e:  # noqa: BLE001 - record failure, keep the suite running
            results["conditions"][name] = {"error": str(e)}

    return results


def degradation_summary(results: dict, primary_metric: str) -> dict:
    """% drop in the objective's primary metric relative to clean, per condition."""
    clean_score = results["conditions"]["clean"].get(primary_metric)
    if clean_score is None:
        return {}
    summary = {}
    for condition, metrics in results["conditions"].items():
        if condition == "clean" or "error" in metrics:
            continue
        score = metrics.get(primary_metric)
        if score is not None and clean_score:
            summary[condition] = (clean_score - score) / clean_score
    return summary


def save_report(results: dict, report_dir: Path = REPORTS_DIR) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    results["timestamp"] = stamp
    out_path = report_dir / f"robustness_{results['model_id']}_{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return out_path
