"""Correctness tests for utils/mstar/reader.py - the MSTAR Phoenix-header
chip parser. Verified against the real, locally-downloaded MSTAR public
target-chip release (datasets/raw/mstar/), not a synthetic fixture, since
the whole point is to prove the real file format is decoded correctly (byte
counts, per-chip header metadata) rather than just exercising the code path.
"""

import pytest

from utils.mrf.mstar_scenes import CATALOG, MSTAR_DIR
from utils.mstar.reader import read_mstar_chip

pytestmark = pytest.mark.skipif(
    not (MSTAR_DIR / CATALOG["t72_132_test_1"]).exists(),
    reason="real MSTAR chip files not present locally - this test exercises the real parser, not a mock",
)


def test_reads_real_t72_chip_header_and_dimensions():
    chip = read_mstar_chip(MSTAR_DIR / CATALOG["t72_132_test_1"])
    assert chip.target_type == "t72_tank"
    assert chip.target_serial == "132"
    assert chip.depression_deg == 15.0
    assert chip.magnitude.shape == (128, 128)
    assert chip.phase.shape == (128, 128)


def test_reads_real_slicy_chip_with_nonstandard_size():
    # SLICY calibration chips are 54x54, not the 128x128 of the vehicle
    # targets - dimensions must come from the file's own header, not an
    # assumed constant.
    chip = read_mstar_chip(MSTAR_DIR / CATALOG["slicy_15deg_1"])
    assert chip.magnitude.shape == (54, 54)


def test_magnitude_is_real_nonnegative_sar_data():
    chip = read_mstar_chip(MSTAR_DIR / CATALOG["bmp2_9563_test_1"])
    m = chip.magnitude
    assert m.min() >= 0.0
    assert m.max() > 0.0
    # a real chip has both bright (target/speckle) and near-zero (shadow) pixels
    assert m.std() > 0.0


def test_rejects_a_non_mstar_file(tmp_path):
    bogus = tmp_path / "not_mstar.bin"
    bogus.write_bytes(b"\x00" * 100)
    with pytest.raises(ValueError, match="not a Phoenix-format"):
        read_mstar_chip(bogus)
