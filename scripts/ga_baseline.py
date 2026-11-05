"""Genetic-algorithm baseline for the QAOA scaling table.

The THALES brief names a MATLAB genetic algorithm (Global Optimization Toolbox)
as its classical baseline. MATLAB is not part of this project, so this script
implements a genetic algorithm in NumPy instead and runs it on the same tiles
and the same energy as the scaling sweep. It is not MATLAB's ga: the operators
and settings below follow ga's usual defaults for binary problems, but they are
this script's own implementation.

Settings: population 200 (50 when the tile has 5 pixels or fewer), the best 5%
kept unchanged each generation, 80% of the remaining children made by uniform
crossover of two tournament-selected parents (tournament of 4) and 20% by
mutating one selected parent (each bit flips with probability 1/n). The run
stops when it has evaluated as many labelings as QAOA samples (15 steps x 512
shots + a final 2,048-shot sample = 9,728), so the comparison is on equal
evaluation budgets. Each tile is repeated over 30 seeds.

Usage: .venv/Scripts/python.exe scripts/ga_baseline.py
No server or Classiq session needed.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mrf import sarbake  # noqa: E402
from utils.mrf.energy import build_patch_energy, energy, solve_graph_cut  # noqa: E402
from utils.mrf.mstar_scenes import read_calibrated_window, target_probability  # noqa: E402

CHIPS = ["t72_132_test_1", "bmp2_9563_test_1"]
SIZES = [3, 4, 5]
SEEDS = 30
BUDGET = 15 * 512 + 2048
ELITE_SHARE = 0.05
CROSSOVER_SHARE = 0.8
TOURNAMENT = 4


def make_fitness(problem):
    edges = np.array(problem.edges)
    e0, e1 = edges[:, 0], edges[:, 1]

    def fitness(x: np.ndarray) -> np.ndarray:
        unary = (x * problem.unary1 + (1 - x) * problem.unary0).sum(axis=1)
        return unary + ((x[:, e0] != x[:, e1]) * problem.weights).sum(axis=1)

    return fitness


def run_ga(fitness, n: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    pop_size = 50 if n <= 5 else 200
    n_elite = int(np.ceil(ELITE_SHARE * pop_size))
    n_children = pop_size - n_elite
    n_cross = int(round(CROSSOVER_SHARE * n_children))

    pop = rng.integers(0, 2, size=(pop_size, n), dtype=np.int8)
    fit = fitness(pop)
    evaluated = pop_size
    best = float(fit.min())

    def tournament(count: int) -> np.ndarray:
        idx = rng.integers(0, pop_size, size=(count, TOURNAMENT))
        return pop[idx[np.arange(count), np.argmin(fit[idx], axis=1)]]

    while evaluated + n_children <= BUDGET:
        elite = pop[np.argsort(fit)[:n_elite]]
        mother, father = tournament(n_cross), tournament(n_cross)
        take_mother = rng.random((n_cross, n)) < 0.5
        crossed = np.where(take_mother, mother, father)
        mutated = tournament(n_children - n_cross).copy()
        flips = rng.random(mutated.shape) < (1.0 / n)
        mutated = np.where(flips, 1 - mutated, mutated).astype(np.int8)
        pop = np.vstack([elite, crossed, mutated]).astype(np.int8)
        fit = fitness(pop)
        evaluated += n_children
        best = min(best, float(fit.min()))
    return best


def main() -> None:
    rows = []
    for chip in CHIPS:
        for size in SIZES:
            row0, col0 = sarbake.boundary_window(chip, size)
            intensity = read_calibrated_window(chip, row0, col0, size)
            problem = build_patch_energy(target_probability(intensity), intensity)
            optimum = float(energy(problem, solve_graph_cut(problem)))
            fitness = make_fitness(problem)
            best = [run_ga(fitness, size * size, seed) for seed in range(SEEDS)]
            row = {
                "chip": chip,
                "size": size,
                "qubits": size * size,
                "optimum": round(optimum, 4),
                "ga_best_mean": round(float(np.mean(best)), 4),
                "ga_best_min": round(float(np.min(best)), 4),
                "ga_best_max": round(float(np.max(best)), 4),
                "seeds_reaching_optimum": int(sum(abs(b - optimum) < 1e-9 for b in best)),
            }
            rows.append(row)
            print(row, flush=True)
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
