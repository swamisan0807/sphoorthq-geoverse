"""Synthetic data-condition perturbations for robustness testing.

Each function simulates a real acquisition/processing failure mode so a
trained model's degradation can be measured systematically rather than
anecdotally from whatever scenes happen to be in the test set.
"""

import numpy as np


def add_speckle_noise(sar_db: np.ndarray, looks: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
    """Simulate under-multilooked SAR (more speckle than the training data)."""
    rng = rng or np.random.default_rng()
    sigma0 = 10 ** (sar_db / 10.0)
    noise = rng.gamma(shape=looks, scale=1.0 / looks, size=sar_db.shape)
    return 10 * np.log10(np.clip(sigma0 * noise, 1e-6, None))


def shift_incidence_angle(gamma0_db: np.ndarray, delta_deg: float) -> np.ndarray:
    """Simulate a scene acquired at a different incidence angle than training data -
    backscatter changes roughly as cos(theta), approximate the shift in dB."""
    return gamma0_db + 10 * np.log10(np.cos(np.radians(delta_deg)) + 1e-6)


def band_dropout(cube: np.ndarray, channel_indices: list[int], fill_value: float = 0.0) -> np.ndarray:
    """Simulate a missing modality (e.g. cloud-obscured optical, DEM gap)."""
    out = cube.copy()
    out[channel_indices] = fill_value
    return out


def temporal_gap(cube: np.ndarray, optical_channel_indices: list[int], staleness_factor: float = 0.7) -> np.ndarray:
    """Simulate stale optical imagery (old acquisition vs current SAR) by blending
    toward the channel mean - crude proxy for temporal decorrelation."""
    out = cube.copy()
    for idx in optical_channel_indices:
        channel_mean = np.nanmean(cube[idx])
        out[idx] = staleness_factor * cube[idx] + (1 - staleness_factor) * channel_mean
    return out


def add_gaussian_noise(cube: np.ndarray, std: float, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return cube + rng.normal(0, std, size=cube.shape)


PERTURBATIONS = {
    "speckle_heavy": lambda cube, sar_idx: _apply_to_channels(cube, sar_idx, add_speckle_noise, looks=1),
    "incidence_shift_10deg": lambda cube, sar_idx: _apply_to_channels(
        cube, sar_idx, shift_incidence_angle, delta_deg=10.0
    ),
    "optical_dropout": lambda cube, opt_idx: band_dropout(cube, opt_idx),
    "gaussian_noise_low": lambda cube, _idx: add_gaussian_noise(cube, std=0.05),
    "gaussian_noise_high": lambda cube, _idx: add_gaussian_noise(cube, std=0.2),
}


def _apply_to_channels(cube: np.ndarray, indices: list[int], fn, **kwargs) -> np.ndarray:
    out = cube.copy()
    for idx in indices:
        out[idx] = fn(cube[idx], **kwargs)
    return out
