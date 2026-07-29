"""Resample every modality onto one common CRS/grid/resolution for an AOI."""

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

from src.core.types import BBox


def target_grid(bbox: BBox, dst_crs: str, resolution_m: float):
    """Compute a transform/shape for the AOI at a fixed resolution."""
    width = int((bbox.maxx - bbox.minx) / resolution_m)
    height = int((bbox.maxy - bbox.miny) / resolution_m)
    transform = rasterio.transform.from_bounds(
        bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, width, height
    )
    return transform, (height, width)


def reproject_match(
    src_path: str,
    dst_crs: str,
    dst_transform,
    dst_shape: tuple[int, int],
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    with rasterio.open(src_path) as src:
        dst = np.zeros(dst_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
        )
    return dst


def stack_to_common_grid(
    source_paths: dict[str, str],
    bbox: BBox,
    dst_crs: str,
    resolution_m: float,
    categorical_bands: set[str] | None = None,
) -> tuple[np.ndarray, list[str], dict]:
    """Reproject every band path onto one grid and stack into (band, H, W)."""
    categorical_bands = categorical_bands or set()
    transform, shape = target_grid(bbox, dst_crs, resolution_m)

    bands = []
    names = []
    for name, path in source_paths.items():
        resampling = Resampling.nearest if name in categorical_bands else Resampling.bilinear
        bands.append(reproject_match(path, dst_crs, transform, shape, resampling))
        names.append(name)

    profile = {
        "driver": "GTiff",
        "height": shape[0],
        "width": shape[1],
        "count": len(bands),
        "dtype": "float32",
        "crs": dst_crs,
        "transform": transform,
    }
    return np.stack(bands, axis=0), names, profile
