"""Catalog endpoints over the real sen1floods11 splits - no separate
metadata store, this reads the same CSV splits the notebooks train on."""

import io
from collections import defaultdict

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image

from src.ai.classic.sen1floods11_dataset import (
    chip_id_from_s1_filename,
    load_split,
    normalize_s1,
    read_label,
    read_s1,
)

router = APIRouter(prefix="/api/catalog", tags=["catalog"])

SPLITS = ("train", "valid", "test")


def _chip_ids(split: str) -> list[str]:
    try:
        pairs = load_split(split)
    except FileNotFoundError:
        raise HTTPException(404, f"unknown split '{split}'")
    return [chip_id_from_s1_filename(s1_filename) for s1_filename, _ in pairs]


@router.get("/events")
def list_events():
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {s: 0 for s in SPLITS})
    for split in SPLITS:
        for chip_id in _chip_ids(split):
            event = chip_id.split("_")[0]
            counts[event][split] += 1
    return [
        {"event": event, **split_counts, "total": sum(split_counts.values())}
        for event, split_counts in sorted(counts.items())
    ]


@router.get("/chips")
def list_chips(split: str = "train", event: str | None = None):
    chip_ids = _chip_ids(split)
    if event:
        chip_ids = [c for c in chip_ids if c.split("_")[0] == event]
    return {"split": split, "event": event, "chip_ids": chip_ids}


@router.get("/chips/{chip_id}/preview.png")
def chip_preview(chip_id: str):
    """False-color VV/VH render alongside the hand-labeled mask, side by side."""
    try:
        s1 = read_s1(chip_id)
        label = read_label(chip_id)
    except Exception:
        raise HTTPException(404, f"chip '{chip_id}' not found")

    s1n = normalize_s1(s1)
    vv, vh = s1n[0], s1n[1]
    rgb = (np.stack([vv, vh, (vv + vh) / 2], axis=-1) * 255).astype(np.uint8)

    labeled = rgb.copy()
    labeled[label == 1] = [30, 120, 255]
    labeled[label == -1] = [140, 140, 140]

    combined = np.concatenate([rgb, labeled], axis=1)
    buf = io.BytesIO()
    Image.fromarray(combined).save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
