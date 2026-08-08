"""Hand-crafted spatial/statistical features - feeds classic ML baselines and
augments the deep model input (satisfies "modelling of spatial and statistical
relationships within image data")."""

import numpy as np
from scipy.ndimage import uniform_filter
from skimage.feature import graycomatrix, graycoprops


def polarimetric_ratio(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    """VV/VH ratio - discriminates surface roughness/scattering type."""
    return vv / (vh + 1e-8)


def polarimetric_difference(vv_db: np.ndarray, vh_db: np.ndarray) -> np.ndarray:
    return vv_db - vh_db


def local_statistics(array: np.ndarray, window: int = 9) -> dict[str, np.ndarray]:
    """Sliding-window mean/std/coefficient-of-variation - texture proxy, cheaper than GLCM."""
    mean = uniform_filter(array, window)
    mean_sq = uniform_filter(array**2, window)
    var = np.clip(mean_sq - mean**2, 0, None)
    std = np.sqrt(var)
    cov = std / (np.abs(mean) + 1e-8)
    return {"mean": mean, "std": std, "cov": cov}


def glcm_texture(
    array_uint8: np.ndarray,
    distances: list[int] = [1],
    angles: list[float] = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    patch_size: int = 32,
) -> dict[str, np.ndarray]:
    """GLCM contrast/homogeneity/energy/correlation, computed per non-overlapping patch
    and upsampled back to full resolution. array_uint8 must be quantized to uint8."""
    h, w = array_uint8.shape
    props = ["contrast", "homogeneity", "energy", "correlation"]
    out = {p: np.zeros((h // patch_size, w // patch_size), dtype=np.float32) for p in props}

    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            patch = array_uint8[i : i + patch_size, j : j + patch_size]
            glcm = graycomatrix(patch, distances=distances, angles=angles, symmetric=True, normed=True)
            for p in props:
                out[p][i // patch_size, j // patch_size] = graycoprops(glcm, p).mean()

    return {p: np.kron(v, np.ones((patch_size, patch_size))) for p, v in out.items()}


def quantize_to_uint8(array: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    lo = lo if lo is not None else np.nanpercentile(array, 2)
    hi = hi if hi is not None else np.nanpercentile(array, 98)
    clipped = np.clip(array, lo, hi)
    return (255 * (clipped - lo) / (hi - lo + 1e-8)).astype(np.uint8)
