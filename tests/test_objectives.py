import numpy as np

from src.ai.objectives.registry import evaluate, f1, get_objective, iou


def test_iou_perfect_match():
    pred = np.array([[1, 1], [0, 0]])
    target = np.array([[1, 1], [0, 0]])
    assert iou(pred, target) == 1.0


def test_iou_no_overlap():
    pred = np.array([[1, 0], [0, 0]])
    target = np.array([[0, 1], [0, 0]])
    assert iou(pred, target) == 0.0


def test_f1_known_value():
    # 1 TP, 1 FP, 1 FN -> precision=0.5, recall=0.5, f1=0.5
    pred = np.array([1, 1, 0])
    target = np.array([1, 0, 1])
    assert f1(pred, target) == 0.5


def test_get_objective_unknown_raises():
    import pytest

    with pytest.raises(KeyError):
        get_objective("not-a-real-objective")


def test_evaluate_flood_segmentation_returns_all_metrics():
    pred = np.array([[1, 0], [0, 1]])
    target = np.array([[1, 0], [1, 1]])
    objective = get_objective("flood-segmentation")
    results = evaluate("flood-segmentation", pred, target)
    assert set(results.keys()) == set(objective.metrics)
