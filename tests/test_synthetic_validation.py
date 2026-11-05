"""Correctness tests for utils/mrf/metrics.py, utils/mrf/synthetic_validation.py,
and POST /api/mstar/synthetic-validation - the one place real Accuracy/IoU/Dice
exist in this app, because it's the one place real ground truth exists (never
a real MSTAR chip, which has no per-pixel mask - see every other mstar
endpoint's `note`)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from utils.mrf.metrics import compute_metrics
from utils.mrf.solvers import run_classical
from utils.mrf.synthetic_validation import build_validation_cases, problem_for_case


def _auth_headers() -> dict[str, str]:
    from utils.auth.tokens import issue_token

    return {"Authorization": f"Bearer {issue_token('synthetic-validation-test-user')}"}


def test_compute_metrics_perfect_prediction_is_all_ones():
    true = np.array([[1, 0], [0, 1]])
    m = compute_metrics(true.copy(), true)
    assert m["accuracy"] == 1.0
    assert m["iou"] == 1.0
    assert m["dice"] == 1.0


def test_compute_metrics_all_wrong_binary_prediction():
    true = np.array([[1, 0], [1, 0]])
    pred = np.array([[0, 1], [0, 1]])
    m = compute_metrics(pred, true)
    assert m["accuracy"] == 0.0
    assert m["iou"] == 0.0  # no overlap between predicted-target and true-target pixels
    assert m["dice"] == 0.0


def test_compute_metrics_partial_overlap_matches_hand_computed_values():
    # true target = {(0,0),(0,1)}, pred target = {(0,0),(1,0)} -> intersection=1, union=3
    true = np.array([[1, 1], [0, 0]])
    pred = np.array([[1, 0], [1, 0]])
    m = compute_metrics(pred, true)
    assert m["accuracy"] == 0.5  # 2/4 pixels match
    assert m["iou"] == pytest.approx(1 / 3)
    assert m["dice"] == pytest.approx(2 * 1 / (2 + 2))


def test_classical_solver_reproduces_thales_worked_example_exactly():
    # The strongest validation this project can offer: run the real
    # classical solver on THALES's own worked example (quoted directly from
    # their brief) and check it reproduces THALES's own stated answer.
    cases = build_validation_cases()
    thales_case = cases[0]
    assert thales_case.name.startswith("THALES worked example")

    problem = problem_for_case(thales_case)
    labels, _energy, _dur = run_classical(problem)
    assert np.array_equal(labels, thales_case.true_labels)

    m = compute_metrics(labels, thales_case.true_labels)
    assert m["accuracy"] == 1.0
    assert m["iou"] == 1.0
    assert m["dice"] == 1.0


def test_classical_solver_reproduces_all_generated_cases_exactly():
    for case in build_validation_cases()[1:]:
        problem = problem_for_case(case)
        labels, _energy, _dur = run_classical(problem)
        m = compute_metrics(labels, case.true_labels)
        assert m["accuracy"] == 1.0, f"{case.name} failed: predicted {labels.tolist()} vs true {case.true_labels.tolist()}"


def test_api_synthetic_validation_classical_only_returns_real_metrics():
    from apps.api.main import app

    client = TestClient(app)
    resp = client.post("/api/mstar/synthetic-validation", json={"run_quantum": False}, headers=_auth_headers())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["cases"]) == 4
    for case in body["cases"]:
        assert case["classical"]["metrics"] is not None
        assert case["classical"]["metrics"]["accuracy"] == 1.0
        assert case["quantum"] is None  # run_quantum was false
    assert "None of the cases is an MSTAR chip" in body["note"]
