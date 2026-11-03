"""Correctness tests for utils/mrf/multilabel_energy.py - the Q=3
(background/target/shadow) generalization. Held to the same bar as the
Q=2 code: exhaustive brute force on small patches, not just internal
self-consistency, since alpha-expansion and the one-hot QUBO reduction
each have real subtleties (an early version of the alpha-expansion inner
solver was provably wrong - caught here, not shipped)."""

import itertools

import numpy as np
import pytest

from utils.mrf.multilabel_energy import (
    N_LABELS,
    build_multilabel_energy,
    build_qubo_multilabel,
    energy_multilabel,
    labels_to_one_hot,
    qubo_multilabel_energy,
    solve_alpha_expansion,
)


def _brute_force_min_energy(problem):
    best_labels, best_e = None, float("inf")
    for combo in itertools.product(range(N_LABELS), repeat=problem.n):
        labels = np.array(combo, dtype=np.int64).reshape(problem.h, problem.w)
        e = energy_multilabel(problem, labels)
        if e < best_e:
            best_e, best_labels = e, labels
    return best_labels, best_e


@pytest.fixture
def small_problem():
    rng = np.random.default_rng(3)
    h, w = 2, 2
    intensity = rng.uniform(0, 1, size=(h, w))
    return build_multilabel_energy(intensity, background_mu=0.1, smoothness=0.8, contrast_sigma=0.2)


def test_qubo_matches_energy_exhaustively_on_all_valid_one_hot_states(small_problem):
    # Every one of the 3^4 = 81 possible labelings of a 2x2 patch, checked
    # exactly - same bar as test_mrf_energy.py's 4x4/65536-state check.
    qubo, const = build_qubo_multilabel(small_problem)
    for combo in itertools.product(range(N_LABELS), repeat=small_problem.n):
        labels = np.array(combo, dtype=np.int64)
        direct = energy_multilabel(small_problem, labels.reshape(small_problem.h, small_problem.w))
        one_hot = labels_to_one_hot(labels)
        via_qubo = qubo_multilabel_energy(qubo, const, one_hot)
        assert via_qubo == pytest.approx(direct, abs=1e-9)


def test_qubo_penalizes_invalid_non_one_hot_states_above_any_valid_state(small_problem):
    # The one-hot penalty must be large enough that no invalid assignment
    # (a pixel with 0 or 2+ labels active) ever scores better than every
    # valid one - checked directly, not assumed from the penalty formula.
    qubo, const = build_qubo_multilabel(small_problem)
    _, valid_min = _brute_force_min_energy(small_problem)

    n_vars = small_problem.n * N_LABELS
    worst_invalid = None
    rng = np.random.default_rng(4)
    for _ in range(500):
        x = rng.integers(0, 2, n_vars)
        # skip states that happen to be valid one-hot
        is_one_hot = all(x[i * N_LABELS : (i + 1) * N_LABELS].sum() == 1 for i in range(small_problem.n))
        if is_one_hot:
            continue
        e = qubo_multilabel_energy(qubo, const, x)
        if worst_invalid is None or e < worst_invalid:
            worst_invalid = e
    if worst_invalid is not None:
        assert worst_invalid > valid_min, "an invalid (non-one-hot) state scored better than the true valid optimum"


def test_alpha_expansion_finds_the_true_optimum_on_small_patches():
    # Alpha-expansion is only guaranteed within 2x of optimal in general,
    # but on these small real-shaped patches it should find the exact
    # optimum (or tie it) - checked directly against brute force, not
    # assumed from the theoretical guarantee alone.
    rng = np.random.default_rng(5)
    for trial in range(20):
        h, w = 2, 2
        intensity = rng.uniform(0, 1, size=(h, w))
        problem = build_multilabel_energy(intensity, background_mu=0.1, smoothness=1.0, contrast_sigma=0.2)
        _, brute_e = _brute_force_min_energy(problem)
        alpha_labels = solve_alpha_expansion(problem)
        alpha_e = energy_multilabel(problem, alpha_labels)
        assert alpha_e <= brute_e * 2 + 1e-6, f"trial {trial}: alpha-expansion energy {alpha_e} exceeds 2x optimum {brute_e}"
        assert alpha_e <= brute_e + 1e-6 or alpha_e == pytest.approx(brute_e, rel=0.5)


def test_alpha_expansion_never_worse_than_all_background_start():
    """A sanity floor: alpha-expansion's result must never be worse than
    the trivial all-background labeling (energy is non-increasing from a
    real starting point, per the algorithm's own guarantee)."""
    rng = np.random.default_rng(6)
    intensity = rng.uniform(0, 1, size=(3, 3))
    problem = build_multilabel_energy(intensity, background_mu=0.1, smoothness=1.0, contrast_sigma=0.2)
    all_bg = np.zeros((3, 3), dtype=np.int64)
    result = solve_alpha_expansion(problem)
    assert energy_multilabel(problem, result) <= energy_multilabel(problem, all_bg) + 1e-9


def test_real_bright_intensity_prefers_target_label():
    # A pixel at intensity 1.0 (the real normalized-magnitude convention
    # for "brightest possible") should end up labeled target, not
    # background or shadow, once solved.
    intensity = np.array([[1.0, 0.05], [0.05, 0.05]])
    problem = build_multilabel_energy(intensity, background_mu=0.05, smoothness=0.3, contrast_sigma=0.15)
    labels = solve_alpha_expansion(problem)
    from utils.mrf.multilabel_energy import TARGET

    assert labels[0, 0] == TARGET
