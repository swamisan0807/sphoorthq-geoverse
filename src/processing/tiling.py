"""Chip a co-registered multi-band cube into fixed-size training patches."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Patch:
    row: int
    col: int
    array: np.ndarray  # (bands, patch_size, patch_size)
    label: np.ndarray | None = None


def chip(
    cube: np.ndarray,
    patch_size: int = 256,
    overlap: int = 32,
    label: np.ndarray | None = None,
    drop_incomplete: bool = True,
) -> list[Patch]:
    """cube: (bands, H, W). Returns overlapping patches for tiling large scenes."""
    _, h, w = cube.shape
    stride = patch_size - overlap
    patches = []

    for row in range(0, h, stride):
        for col in range(0, w, stride):
            r_end, c_end = row + patch_size, col + patch_size
            if r_end > h or c_end > w:
                if drop_incomplete:
                    continue
                r_end, c_end = min(r_end, h), min(c_end, w)

            patch_arr = cube[:, row:r_end, col:c_end]
            patch_label = label[row:r_end, col:c_end] if label is not None else None
            patches.append(Patch(row=row, col=col, array=patch_arr, label=patch_label))

    return patches


def split_patches(
    patches: list[Patch], train: float = 0.7, val: float = 0.15, seed: int = 42
) -> dict[str, list[Patch]]:
    """Deterministic spatial-block split (by patch index, not random pixels -
    avoids leaking overlapping neighborhoods across train/val/test)."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(patches))

    n_train = int(len(patches) * train)
    n_val = int(len(patches) * val)

    return {
        "train": [patches[i] for i in indices[:n_train]],
        "val": [patches[i] for i in indices[n_train : n_train + n_val]],
        "test": [patches[i] for i in indices[n_train + n_val :]],
    }
