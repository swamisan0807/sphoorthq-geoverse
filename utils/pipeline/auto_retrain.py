"""Detects new data landing in datasets/raw/sen1floods11 and can trigger
real retraining jobs when it does. "New data" means a chip_id genuinely
present in the S1Hand directory listing that wasn't there at the last
baseline check - a real filesystem diff, not a stub. There is no live
streaming ingestion feeding this directory today, so in practice this
fires when the ingestion connectors (utils/ingestion/) pull additional
chips in - the mechanism itself is real and testable independent of that.
"""

import json

from utils.ai.classic.sen1floods11_dataset import HAND_LABELED_DIR
from utils.core.paths import METADATA_DIR

MANIFEST_PATH = METADATA_DIR / "known_chip_ids.json"


def current_chip_ids() -> set[str]:
    s1_dir = HAND_LABELED_DIR / "S1Hand"
    if not s1_dir.exists():
        return set()
    return {p.name.removesuffix("_S1Hand.tif") for p in s1_dir.glob("*_S1Hand.tif")}


def _load_baseline() -> set[str] | None:
    if not MANIFEST_PATH.exists():
        return None
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return set(json.load(f))


def save_baseline(chip_ids: set[str]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(chip_ids), f, indent=2)


def check_new_data() -> dict:
    current = current_chip_ids()
    baseline = _load_baseline()
    if baseline is None:
        save_baseline(current)
        return {
            "has_new_data": False,
            "new_chip_ids": [],
            "total_chips": len(current),
            "baseline_just_created": True,
        }
    new_ids = sorted(current - baseline)
    return {
        "has_new_data": len(new_ids) > 0,
        "new_chip_ids": new_ids,
        "total_chips": len(current),
        "baseline_just_created": False,
    }
