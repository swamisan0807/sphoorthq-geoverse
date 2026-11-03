"""Synthetic validation set with real, known ground truth - the only honest
way to compute Accuracy/IoU/Dice in this project, since real MSTAR chips
have no per-pixel mask (see utils/mrf/mstar_scenes.py and every
apps/api/routers/mstar.py response's `note`). Two sources, both real:

1. THALES's own worked example from its SAR-segmentation brief (page 3,
   "TEST MODEL & DATA DESCRIPTION") - quoted directly, not paraphrased: a
   5x5 intensity matrix and its label matrix, exactly as printed. Running
   this project's real solvers against THALES's own example and comparing
   to THALES's own stated answer is about as strong a validation as this
   project can offer.
2. A few small generated cases (a known bright rectangular "target" on a
   dark background, optionally perturbed by
   utils.mrf.energy.inject_gaussian_noise) - ground truth known by
   construction, not measured or assumed.

Every case here is clearly synthetic, not real MSTAR. utils/mrf/metrics.py's
Accuracy/IoU/Dice are computed only against cases from this module - never
applied to an actual MSTAR chip result anywhere in this codebase.
"""

from dataclasses import dataclass

import numpy as np

from utils.mrf.energy import PatchEnergy, build_patch_energy, inject_gaussian_noise


@dataclass
class ValidationCase:
    name: str
    source: str  # where the ground truth comes from - never "assumed" or "estimated"
    intensity: np.ndarray  # normalized [0, 1]
    true_labels: np.ndarray  # int labels, same shape


def _thales_worked_example() -> ValidationCase:
    # Quoted directly from THALES's SAR Image Segmentation brief, page 3
    # ("TEST MODEL & DATA DESCRIPTION"): the intensity matrix (left) and its
    # label matrix (right), class 0 = background, class 1 = object.
    raw = np.array(
        [
            [100, 10, 10, 10, 10],
            [100, 10, 10, 10, 10],
            [100, 100, 100, 10, 10],
            [100, 100, 100, 10, 10],
            [100, 100, 100, 10, 10],
        ],
        dtype=np.float64,
    )
    true_labels = np.array(
        [
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
        ],
        dtype=np.int64,
    )
    intensity = raw / 100.0  # THALES's own values are exactly {10, 100} -> {0.1, 1.0}
    return ValidationCase(
        name="THALES worked example (5x5)",
        source="THALES SAR-segmentation brief, page 3 - quoted directly",
        intensity=intensity,
        true_labels=true_labels,
    )


def _generated_case(name: str, size: int, target_box: tuple[int, int, int, int], noise_sigma: float = 0.0, seed: int = 0) -> ValidationCase:
    """target_box = (r0, r1, c0, c1), inclusive - the known bright region.
    Ground truth is exactly this box, by construction, not inferred."""
    r0, r1, c0, c1 = target_box
    intensity = np.full((size, size), 0.1)
    true_labels = np.zeros((size, size), dtype=np.int64)
    intensity[r0 : r1 + 1, c0 : c1 + 1] = 1.0
    true_labels[r0 : r1 + 1, c0 : c1 + 1] = 1
    if noise_sigma > 0:
        intensity = inject_gaussian_noise(intensity, noise_sigma, seed=seed)
    return ValidationCase(
        name=name,
        source=f"generated: known {r1 - r0 + 1}x{c1 - c0 + 1} target box" + (f", synthetic noise sigma={noise_sigma}" if noise_sigma > 0 else ""),
        intensity=intensity,
        true_labels=true_labels,
    )


def build_validation_cases() -> list[ValidationCase]:
    return [
        _thales_worked_example(),
        _generated_case("clean 3x3, corner target", size=3, target_box=(0, 1, 0, 1)),
        _generated_case("clean 4x4, block target", size=4, target_box=(1, 2, 1, 2)),
        _generated_case("noisy 3x3, corner target (sigma=0.15)", size=3, target_box=(0, 1, 0, 1), noise_sigma=0.15, seed=1),
    ]


def problem_for_case(case: ValidationCase, smoothness: float = 1.0, contrast_sigma: float = 0.15) -> PatchEnergy:
    return build_patch_energy(case.intensity, case.intensity, smoothness=smoothness, contrast_sigma=contrast_sigma)
