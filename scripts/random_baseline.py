"""Random-sampling baseline for the QAOA scaling table.

For each tile used by scripts/scaling_sweep.py, draw uniformly random labelings
with the same sample budget QAOA gets (15 optimizer steps x 512 shots + a final
2,048-shot sample), keep the lowest-energy one, and repeat over 30 seeds. If
random sampling reaches the classical optimum as often as QAOA does, a QAOA
match on that tile shows the pipeline is correct, not that QAOA is doing
better than chance.

Usage: .venv/Scripts/python.exe scripts/random_baseline.py
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
N_SAMPLES = 15 * 512 + 2048


def main() -> None:
    rows = []
    for chip in CHIPS:
        for size in SIZES:
            row0, col0 = sarbake.boundary_window(chip, size)
            intensity = read_calibrated_window(chip, row0, col0, size)
            problem = build_patch_energy(target_probability(intensity), intensity)
            optimum = energy(problem, solve_graph_cut(problem))
            n = size * size
            edges = np.array(problem.edges)
            best = []
            for seed in range(SEEDS):
                x = np.random.default_rng(seed).integers(0, 2, size=(N_SAMPLES, n), dtype=np.int8)
                e = (x * problem.unary1 + (1 - x) * problem.unary0).sum(1)
                e = e + ((x[:, edges[:, 0]] != x[:, edges[:, 1]]) * problem.weights).sum(1)
                best.append(float(e.min()))
            row = {
                "chip": chip,
                "size": size,
                "qubits": n,
                "optimum": round(float(optimum), 4),
                "random_best_mean": round(float(np.mean(best)), 4),
                "random_best_min": round(float(np.min(best)), 4),
                "seeds_reaching_optimum": int(sum(abs(b - optimum) < 1e-9 for b in best)),
            }
            rows.append(row)
            print(row, flush=True)
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
