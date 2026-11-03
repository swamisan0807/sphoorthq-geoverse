"""Correctness tests for utils/mrf/energy.py - the classical graph-cut
solver, the QUBO reformulation, and the QUBO->Ising map all have to agree on
the SAME energy for apps/api/routers/mrf.py's classical-vs-QAOA comparison
to mean anything. Verified against brute force on small patches, not just
internal self-consistency.
"""

import itertools

import numpy as np
import pytest

from utils.mrf.energy import (
    build_patch_energy,
    build_qubo,
    energy,
    inject_gaussian_noise,
    qubo_energy,
    qubo_to_ising,
    solve_graph_cut,
)
from utils.mrf.verification import exhaustive_equivalence_check


def _brute_force_min_energy(problem):
    best_labels, best_e = None, float("inf")
    for bits in itertools.product([0, 1], repeat=problem.n):
        labels = np.array(bits, dtype=np.int64)
        e = energy(problem, labels)
        if e < best_e:
            best_e, best_labels = e, labels
    return best_labels.reshape(problem.h, problem.w), best_e


@pytest.fixture
def small_problem():
    rng = np.random.default_rng(0)
    h, w = 3, 3
    prob = rng.uniform(0.05, 0.95, size=(h, w))
    intensity = rng.uniform(0, 1, size=(h, w))
    return build_patch_energy(prob, intensity, smoothness=0.8, contrast_sigma=0.2)


def test_graph_cut_matches_brute_force_minimum(small_problem):
    _, brute_e = _brute_force_min_energy(small_problem)
    cut_labels = solve_graph_cut(small_problem)
    cut_e = energy(small_problem, cut_labels)
    assert cut_e == pytest.approx(brute_e, abs=1e-9)


def test_graph_cut_matches_brute_force_minimum_on_confident_patch():
    # A patch with a clear water/land split and strong smoothness - graph
    # cut should recover the obviously-correct segmentation, not just tie
    # brute force numerically by accident.
    prob = np.array([[0.95, 0.9, 0.1], [0.92, 0.15, 0.05], [0.3, 0.1, 0.05]])
    intensity = np.array([[0.9, 0.85, 0.1], [0.88, 0.15, 0.1], [0.2, 0.1, 0.05]])
    problem = build_patch_energy(prob, intensity, smoothness=2.0, contrast_sigma=0.3)
    brute_labels, brute_e = _brute_force_min_energy(problem)
    cut_labels = solve_graph_cut(problem)
    assert energy(problem, cut_labels) == pytest.approx(brute_e, abs=1e-9)
    np.testing.assert_array_equal(cut_labels, brute_labels)


def test_qubo_matches_energy_on_random_assignments(small_problem):
    qubo, const = build_qubo(small_problem)
    rng = np.random.default_rng(1)
    for _ in range(20):
        labels = rng.integers(0, 2, size=small_problem.n)
        direct = energy(small_problem, labels)
        via_qubo = qubo_energy(qubo, const, labels)
        assert via_qubo == pytest.approx(direct, abs=1e-9)


def test_ising_matches_qubo_on_random_assignments(small_problem):
    qubo, const = build_qubo(small_problem)
    h, J, ising_const = qubo_to_ising(qubo, const, small_problem.n)
    rng = np.random.default_rng(2)
    for _ in range(20):
        labels = rng.integers(0, 2, size=small_problem.n)
        x = labels.astype(float)
        z = 1 - 2 * x  # x=0 -> z=+1, x=1 -> z=-1, matching x_i=(1-z_i)/2

        ising_e = ising_const
        for i, coeff in h.items():
            ising_e += coeff * z[i]
        for (i, j), coeff in J.items():
            ising_e += coeff * z[i] * z[j]

        assert ising_e == pytest.approx(qubo_energy(qubo, const, labels), abs=1e-9)


def test_qubo_matches_energy_exhaustively_on_thales_style_4x4_patch():
    # Not a random sample: EVERY one of the 2**16 = 65,536 possible
    # labelings of a 4x4 patch, checked exactly - the evidence behind
    # "build_qubo() is an exact reformulation of energy(), not merely a
    # close approximation of it". Also confirms solve_graph_cut() lands on
    # the true global optimum, not just a local one.
    intensity = (
        np.array(
            [
                [100, 10, 10, 10],
                [100, 10, 10, 10],
                [100, 100, 100, 10],
                [100, 100, 100, 10],
            ],
            dtype=float,
        )
        / 100.0
    )
    rng = np.random.default_rng(0)
    prob = np.clip(intensity + rng.normal(0, 0.12, size=intensity.shape), 0.02, 0.98)
    problem = build_patch_energy(prob, intensity, smoothness=1.5, contrast_sigma=0.25)

    result = exhaustive_equivalence_check(problem)

    assert result.n == 16
    assert result.n_states == 65536
    assert result.max_abs_qubo_energy_diff < 1e-9
    assert result.graph_cut_matches_brute_force_optimum


def test_exhaustive_check_matches_brute_force_helper_on_small_problem(small_problem):
    # Cross-check exhaustive_equivalence_check's vectorized brute force
    # against this file's own itertools-based _brute_force_min_energy on
    # the same fixture used by the graph-cut test above - two independently
    # written brute-force implementations agreeing is stronger evidence
    # than trusting either alone.
    _, expected_min = _brute_force_min_energy(small_problem)
    result = exhaustive_equivalence_check(small_problem)
    assert result.brute_force_min_energy == pytest.approx(expected_min, abs=1e-9)


def test_inject_gaussian_noise_is_reproducible_and_bounded():
    # Same (intensity, sigma, seed) must give the same noisy array every
    # time - the robustness sweep on apps/web's About page has to be
    # honestly re-runnable, not a one-off random result.
    intensity = np.full((5, 5), 0.5)
    noisy_a = inject_gaussian_noise(intensity, sigma=0.2, seed=0)
    noisy_b = inject_gaussian_noise(intensity, sigma=0.2, seed=0)
    assert np.array_equal(noisy_a, noisy_b)
    assert noisy_a.min() >= 0.0 and noisy_a.max() <= 1.0
    assert not np.array_equal(noisy_a, intensity)  # sigma > 0 actually perturbs it


def test_inject_gaussian_noise_sigma_zero_is_a_no_op():
    intensity = np.array([[0.1, 0.9], [0.4, 0.6]])
    assert np.array_equal(inject_gaussian_noise(intensity, sigma=0.0), intensity)


def test_inject_gaussian_noise_increases_energy_of_the_clean_optimum():
    # A clean, easy problem (sharp 0/1 split) with sigma=0 solves to a low
    # energy; adding real noise to the same intensity should make that same
    # clean-optimal labeling cost more (the unary evidence is now less
    # confident) - the actual mechanism the robustness sweep measures.
    intensity = np.array([[0.05, 0.05], [0.95, 0.95]])
    clean = build_patch_energy(intensity, intensity)
    clean_optimum = solve_graph_cut(clean)
    clean_energy = energy(clean, clean_optimum)

    noisy_intensity = inject_gaussian_noise(intensity, sigma=0.3, seed=1)
    noisy = build_patch_energy(noisy_intensity, noisy_intensity)
    noisy_energy_at_clean_optimum = energy(noisy, clean_optimum)

    assert noisy_energy_at_clean_optimum > clean_energy


def test_qubo_omits_zero_linear_terms_but_stays_correct():
    # A single pixel, no neighbors, unary equally split -> linear coeff is
    # exactly 0 and should not appear as an explicit (i, i) key, but the
    # energy must still come out right.
    prob = np.array([[0.5]])
    intensity = np.array([[0.5]])
    problem = build_patch_energy(prob, intensity)
    qubo, const = build_qubo(problem)
    assert (0, 0) not in qubo
    for labels in ([0], [1]):
        assert qubo_energy(qubo, const, np.array(labels)) == pytest.approx(
            energy(problem, np.array(labels)), abs=1e-9
        )
