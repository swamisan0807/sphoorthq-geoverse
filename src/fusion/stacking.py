"""Early fusion: stack co-registered bands from different sources into one tensor."""

import numpy as np

# canonical channel order for the fused input cube - keep stable, model configs
# reference channels by index derived from this list.
DEFAULT_CHANNEL_ORDER = [
    "s1_vv_db",
    "s1_vh_db",
    "s2_b02",  # blue
    "s2_b03",  # green
    "s2_b04",  # red
    "s2_b08",  # nir
    "ndvi",
    "ndwi",
    "dem",
    "slope",
    "worldcover",
]


def stack_bands(band_arrays: dict[str, np.ndarray], channel_order: list[str] | None = None) -> np.ndarray:
    order = channel_order or DEFAULT_CHANNEL_ORDER
    missing = [c for c in order if c not in band_arrays]
    if missing:
        raise KeyError(f"missing channels for fusion stack: {missing}")
    return np.stack([band_arrays[c] for c in order], axis=0)


def normalize_channel(array: np.ndarray, method: str = "zscore", stats: dict | None = None) -> np.ndarray:
    if method == "zscore":
        mean = stats["mean"] if stats else np.nanmean(array)
        std = stats["std"] if stats else np.nanstd(array)
        return (array - mean) / (std + 1e-8)
    if method == "minmax":
        lo = stats["min"] if stats else np.nanmin(array)
        hi = stats["max"] if stats else np.nanmax(array)
        return (array - lo) / (hi - lo + 1e-8)
    raise ValueError(f"unknown normalization method: {method}")


def normalize_stack(cube: np.ndarray, channel_order: list[str], per_channel_stats: dict[str, dict]) -> np.ndarray:
    out = np.empty_like(cube, dtype=np.float32)
    for i, name in enumerate(channel_order):
        stats = per_channel_stats.get(name)
        out[i] = normalize_channel(cube[i], stats=stats) if stats else cube[i]
    return out
