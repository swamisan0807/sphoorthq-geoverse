"""ESA WorldCover tiles: mosaic over an AOI and remap to project taxonomy."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject

# ESA WorldCover 10m v200 class codes -> project taxonomy
WORLDCOVER_CLASS_MAP = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse_vegetation",
    70: "snow_ice",
    80: "water",
    90: "wetland_herbaceous",
    95: "mangroves",
    100: "moss_lichen",
}

WATER_CLASS_CODES = {80, 90, 95}  # used as weak flood/water labels


def find_tiles_for_bbox(worldcover_dir: Path, bbox: tuple[float, float, float, float]) -> list[Path]:
    """Return WorldCover tile paths whose 3-degree grid cell intersects bbox.

    Tile naming: ESA_WorldCover_10m_2021_v200_{N|S}{lat:02d}{E|W}{lon:03d}_Map.tif,
    each tile covering a 3x3 degree cell anchored at its SW corner.
    """
    minx, miny, maxx, maxy = bbox
    matches = []
    for tif in worldcover_dir.glob("ESA_WorldCover_*_Map.tif"):
        parts = tif.stem.split("_")
        grid = next((p for p in parts if len(p) == 7 and p[0] in "NS"), None)
        if grid is None:
            continue
        lat = int(grid[1:3]) * (1 if grid[0] == "N" else -1)
        lon = int(grid[4:7]) * (1 if grid[3] == "E" else -1)
        tile_minx, tile_miny = lon, lat
        tile_maxx, tile_maxy = lon + 3, lat + 3
        if not (tile_maxx < minx or tile_minx > maxx or tile_maxy < miny or tile_miny > maxy):
            matches.append(tif)
    return matches


def mosaic_tiles(tile_paths: list[Path]) -> tuple[np.ndarray, dict]:
    srcs = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, transform = merge(srcs)
    finally:
        for s in srcs:
            s.close()
    profile = rasterio.open(tile_paths[0]).profile
    profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform)
    return mosaic[0], profile


def reproject_to_grid(
    array: np.ndarray, src_profile: dict, dst_crs: str, dst_transform, dst_shape: tuple[int, int]
) -> np.ndarray:
    dst = np.zeros(dst_shape, dtype=array.dtype)
    reproject(
        source=array,
        destination=dst,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,  # categorical data - never interpolate
    )
    return dst


def water_mask(landcover: np.ndarray) -> np.ndarray:
    return np.isin(landcover, list(WATER_CLASS_CODES))
