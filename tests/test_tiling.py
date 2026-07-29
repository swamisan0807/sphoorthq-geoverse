import numpy as np

from src.processing.tiling import chip, split_patches


def test_chip_covers_exact_grid():
    cube = np.zeros((3, 512, 512))
    patches = chip(cube, patch_size=256, overlap=0)
    assert len(patches) == 4  # 512/256 = 2, 2x2 grid
    assert all(p.array.shape == (3, 256, 256) for p in patches)


def test_chip_drops_incomplete_by_default():
    cube = np.zeros((3, 300, 300))
    patches = chip(cube, patch_size=256, overlap=0, drop_incomplete=True)
    assert len(patches) == 1  # only the top-left full 256x256 fits


def test_chip_keeps_incomplete_when_requested():
    cube = np.zeros((3, 300, 300))
    patches = chip(cube, patch_size=256, overlap=0, drop_incomplete=False)
    assert len(patches) == 4  # partial patches along the bottom/right edges


def test_split_patches_partitions_all():
    cube = np.zeros((1, 256, 2560))
    patches = chip(cube, patch_size=256, overlap=0)
    splits = split_patches(patches, train=0.7, val=0.15)
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == len(patches)
