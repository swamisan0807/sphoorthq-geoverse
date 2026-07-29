"""Copernicus DEM derivatives: slope, aspect, hillshade, SAR layover/shadow mask."""

import numpy as np
import rasterio


def read_dem(path: str) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32), src.profile


def compute_slope_aspect(dem: np.ndarray, pixel_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Slope (degrees) and aspect (degrees, 0=north, clockwise) via Horn's method."""
    dzdx = np.gradient(dem, axis=1) / pixel_size_m
    dzdy = np.gradient(dem, axis=0) / pixel_size_m

    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    aspect = np.degrees(np.arctan2(dzdy, -dzdx))
    aspect = (90.0 - aspect) % 360.0
    return slope, aspect


def compute_hillshade(
    dem: np.ndarray,
    pixel_size_m: float,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
) -> np.ndarray:
    slope, aspect = compute_slope_aspect(dem, pixel_size_m)
    slope_rad = np.radians(slope)
    aspect_rad = np.radians(aspect)
    azimuth_rad = np.radians(360.0 - azimuth_deg + 90.0)
    altitude_rad = np.radians(altitude_deg)

    shaded = np.sin(altitude_rad) * np.cos(slope_rad) + np.cos(altitude_rad) * np.sin(
        slope_rad
    ) * np.cos(azimuth_rad - aspect_rad)
    return np.clip(shaded * 255, 0, 255).astype(np.uint8)


def layover_shadow_mask(
    slope_deg: np.ndarray, aspect_deg: np.ndarray, sar_look_angle_deg: float, sar_heading_deg: float
) -> np.ndarray:
    """Coarse SAR layover/shadow mask from terrain slope facing the sensor.

    Returns a uint8 mask: 0=normal, 1=layover, 2=shadow.
    This is a simplified geometric approximation, not a full range-Doppler model -
    good enough to flag AOI regions that need masking before training.
    """
    relative_aspect = (aspect_deg - sar_heading_deg + 360.0) % 360.0
    facing_sensor = (relative_aspect < 90.0) | (relative_aspect > 270.0)

    mask = np.zeros_like(slope_deg, dtype=np.uint8)
    layover = facing_sensor & (slope_deg > (90.0 - sar_look_angle_deg))
    shadow = (~facing_sensor) & (slope_deg > sar_look_angle_deg)
    mask[layover] = 1
    mask[shadow] = 2
    return mask
