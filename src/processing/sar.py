"""Sentinel-1 GRD preprocessing: calibration, speckle filtering, terrain correction.

Reference chain (standard SAR preprocessing order):
  1. radiometric calibration (DN -> sigma0)
  2. speckle filtering
  3. terrain (radiometric slope) correction using a DEM
  4. conversion to dB for model input
"""

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter


def calibrate_to_sigma0(dn: np.ndarray, calibration_lut: np.ndarray) -> np.ndarray:
    """Apply per-pixel calibration: sigma0 = DN^2 / calibration_constant^2.

    calibration_lut must already be resampled/interpolated to the raster's
    full resolution (see Sentinel-1 annotation/calibration/*.xml sigmaNought LUT).
    """
    dn = dn.astype(np.float64)
    return (dn**2) / (calibration_lut**2)


def lee_filter(image: np.ndarray, window: int = 7) -> np.ndarray:
    """Refined Lee-style speckle filter using local mean/variance."""
    img = image.astype(np.float64)
    mean = uniform_filter(img, window)
    mean_sq = uniform_filter(img**2, window)
    variance = mean_sq - mean**2

    overall_variance = np.var(img)
    weights = variance / (variance + overall_variance + 1e-12)
    return mean + weights * (img - mean)


def to_db(sigma0: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    return 10.0 * np.log10(np.clip(sigma0, floor, None))


def terrain_flatten(sigma0: np.ndarray, local_incidence_angle_deg: np.ndarray) -> np.ndarray:
    """Radiometric terrain correction: gamma0 = sigma0 / cos(local_incidence_angle)."""
    theta = np.deg2rad(local_incidence_angle_deg)
    return sigma0 / np.clip(np.cos(theta), 1e-3, None)


def preprocess_grd(
    src_path: str,
    calibration_lut: np.ndarray,
    local_incidence_angle_deg: np.ndarray,
    speckle_window: int = 7,
) -> tuple[np.ndarray, dict]:
    """Full chain: read -> calibrate -> despeckle -> terrain-flatten -> dB."""
    with rasterio.open(src_path) as src:
        dn = src.read(1)
        profile = src.profile

    sigma0 = calibrate_to_sigma0(dn, calibration_lut)
    despeckled = lee_filter(sigma0, window=speckle_window)
    gamma0 = terrain_flatten(despeckled, local_incidence_angle_deg)
    db = to_db(gamma0)

    profile.update(dtype="float32", count=1)
    return db.astype(np.float32), profile
