"""Reader for the real MSTAR Public Targets chip format (T72/BMP2/BTR70/SLICY).

Each chip file is a Phoenix-format ASCII header followed by a binary data
block: a NumberOfRows x NumberOfColumns magnitude plane, then an equally
sized phase plane, both big-endian ("Sun") float32 - per
datasets/raw/mstar/.../README.TXT. Header fields are real per-chip ground
truth (target type/serial, azimuth, depression) parsed from the file itself,
never inferred from the path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_HEADER_END_MARKER = b"[EndofPhoenixHeader]"


@dataclass(frozen=True)
class MstarChip:
    path: Path
    header: dict[str, str]
    magnitude: np.ndarray  # (rows, cols) float32
    phase: np.ndarray  # (rows, cols) float32

    @property
    def target_type(self) -> str:
        return self.header.get("TargetType", "unknown")

    @property
    def target_serial(self) -> str:
        return self.header.get("TargetSerNum", "unknown")

    @property
    def depression_deg(self) -> float:
        return float(self.header.get("DesiredDepression", "nan"))

    @property
    def azimuth_deg(self) -> float:
        return float(self.header.get("TargetAz", "nan"))


def _parse_header(raw: bytes) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in raw.decode("ascii", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        fields[key.strip()] = value.strip()
    return fields


def read_mstar_chip(path: str | Path) -> MstarChip:
    """Parse one real MSTAR chip file into its header and magnitude/phase planes."""
    path = Path(path)
    raw = path.read_bytes()

    marker_end = raw.find(_HEADER_END_MARKER)
    if marker_end == -1:
        raise ValueError(f"{path}: not a Phoenix-format MSTAR chip (no header marker found)")
    header_end = marker_end + len(_HEADER_END_MARKER) + 1  # + trailing newline
    header = _parse_header(raw[:header_end])

    rows = int(header["NumberOfRows"])
    cols = int(header["NumberOfColumns"])
    plane_size = rows * cols
    plane_bytes = plane_size * 4

    data = raw[header_end:]
    expected = plane_bytes * 2
    if len(data) != expected:
        raise ValueError(
            f"{path}: expected {expected} bytes of image data ({rows}x{cols}, "
            f"magnitude+phase float32), found {len(data)}"
        )

    magnitude = np.frombuffer(data[:plane_bytes], dtype=">f4").reshape(rows, cols).astype(np.float32)
    phase = np.frombuffer(data[plane_bytes:], dtype=">f4").reshape(rows, cols).astype(np.float32)

    return MstarChip(path=path, header=header, magnitude=magnitude, phase=phase)
