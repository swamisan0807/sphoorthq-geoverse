"""Sentinel-2 L2A preprocessing: cloud masking and spectral indices.

SCL (Scene Classification Layer) class codes used for cloud/shadow masking:
  0 no data, 1 saturated/defective, 3 cloud shadow, 8/9/10 cloud (med/high/cirrus), 11 snow
"""

import numpy as np
import rasterio

CLOUD_SHADOW_SCL_CLASSES = {0, 1, 3, 8, 9, 10, 11}


def read_band(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def cloud_mask_from_scl(scl: np.ndarray) -> np.ndarray:
    """Returns a boolean mask, True = valid (clear) pixel."""
    invalid = np.isin(scl, list(CLOUD_SHADOW_SCL_CLASSES))
    return ~invalid


def _normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = a + b
    return np.where(denom == 0, 0.0, (a - b) / np.where(denom == 0, 1, denom))


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return _normalized_difference(nir, red)


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """McFeeters NDWI - open water."""
    return _normalized_difference(green, nir)


def compute_mndwi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """Modified NDWI (Xu 2006) - better water detection in built-up areas."""
    return _normalized_difference(green, swir1)


def apply_mask(array: np.ndarray, valid_mask: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    return np.where(valid_mask, array, fill_value)
