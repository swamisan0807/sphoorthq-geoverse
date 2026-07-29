"""SAR speckle filtering and dB conversion.

sen1floods11 chips (the only SAR data in this repo - see docs/architecture.md
for why the other raw scenes were removed) arrive already radiometrically
calibrated and terrain-corrected by the dataset publisher, so only speckle
filtering and dB handling are needed here. The full raw-GRD calibration
chain (DN -> sigma0 via a calibration LUT, terrain flattening via a DEM's
local incidence angle) is a real, separate preprocessing stage for
uncalibrated Sentinel-1 products - out of scope until such a product is
actually part of this pipeline.
"""

import numpy as np
from scipy.ndimage import uniform_filter


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
