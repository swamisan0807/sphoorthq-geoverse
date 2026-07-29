"""Robustness benchmark harness: runs a model against clean + perturbed inputs,
writes a comparison report to datasets/reports/."""

import json
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from src.ai.objectives.registry import evaluate as evaluate_objective
from src.ai.robustness.perturbations import PERTURBATIONS
from src.core.paths import REPORTS_DIR


def run_robustness_suite(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    cube: np.ndarray,
    label: np.ndarray,
    objective_name: str,
    sar_channel_indices: list[int],
    optical_channel_indices: list[int],
    model_id: str,
) -> dict:
    results = {"model_id": model_id, "objective": objective_name, "conditions": {}}

    clean_pred = predict_fn(cube)
    results["conditions"]["clean"] = evaluate_objective(objective_name, clean_pred, label)

    for name, perturb_fn in PERTURBATIONS.items():
        idx = sar_channel_indices if "speckle" in name or "incidence" in name else optical_channel_indices
        try:
            perturbed_cube = perturb_fn(cube, idx)
            pred = predict_fn(perturbed_cube)
            results["conditions"][name] = evaluate_objective(objective_name, pred, label)
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
