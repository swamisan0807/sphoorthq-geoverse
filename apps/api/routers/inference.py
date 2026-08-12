"""Runs the real trained models (Decision Tree from notebook 04, U-Net from
notebook 09) against a chosen chip - same code paths the notebooks use,
just invoked from a request instead of a cell."""

import io
import pickle
from pathlib import Path

import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel

from utils.ai.classic.sen1floods11_dataset import (
    normalize_s1,
    read_label,
    read_s1,
    s1_nodata_mask,
)
from utils.ai.classic.unet import UNet
from utils.ai.objectives.registry import evaluate
from utils.core.paths import PROCESSED_DIR
from utils.fusion.pixel_features import build_feature_cube
from utils.registry import model_registry

router = APIRouter(prefix="/api/inference", tags=["inference"])

# fallback paths for models trained before the registry existed (or if a
# notebook was run directly, outside the Jobs engine, and never registered)
_FALLBACK_RF_PATH = PROCESSED_DIR / "models" / "classical_rf_v1.pkl"
_FALLBACK_UNET_PATH = PROCESSED_DIR / "models" / "patch_unet_v1.pt"

# (resolved path, loaded object) - reloaded whenever the registry's
# "current" pointer resolves to a different path than what's cached, so a
# restore() takes effect on the next inference call without a restart
_rf_cache: tuple[str, object] | None = None
_unet_cache: tuple[str, object] | None = None


def _rf_path() -> Path:
    return model_registry.get_current_path("classical_rf") or _FALLBACK_RF_PATH


def _unet_path() -> Path:
    return model_registry.get_current_path("patch_unet") or _FALLBACK_UNET_PATH


def _load_rf():
    global _rf_cache
    path = _rf_path()
    if not path.exists():
        raise HTTPException(404, "Decision Tree model not found - run notebooks/04_classical_ml.ipynb first")
    if _rf_cache is None or _rf_cache[0] != str(path):
        with open(path, "rb") as f:
            _rf_cache = (str(path), pickle.load(f))
    return _rf_cache[1]


def _load_unet():
    global _unet_cache
    path = _unet_path()
    if not path.exists():
        raise HTTPException(404, "U-Net model not found - run notebooks/09_patch_unet.ipynb first")
    if _unet_cache is None or _unet_cache[0] != str(path):
        model = UNet(in_channels=2, num_classes=1, base_channels=16)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        _unet_cache = (str(path), model)
    return _unet_cache[1]


def _run_model(chip_id: str, model_name: str, s1: np.ndarray) -> np.ndarray:
    if model_name == "rf":
        rf = _load_rf()
        cube = build_feature_cube(s1)
        channels, h, w = cube.shape
        flat = cube.reshape(channels, h * w).T
        return rf.predict(flat).reshape(h, w).astype(np.int64)
    if model_name == "unet":
        unet = _load_unet()
        x = torch.from_numpy(normalize_s1(s1)).unsqueeze(0).float()
        with torch.no_grad():
            logits = unet(x)
            probs = torch.sigmoid(logits)[0, 0]
        return (probs > 0.5).numpy().astype(np.int64)
    raise HTTPException(400, f"unknown model '{model_name}', expected 'rf' or 'unet'")


def _predict(chip_id: str, model_name: str):
    try:
        s1 = read_s1(chip_id)
        label = read_label(chip_id)
    except Exception:
        raise HTTPException(404, f"chip '{chip_id}' not found")
    valid_mask = (label != -1) & ~s1_nodata_mask(s1)
    pred = _run_model(chip_id, model_name, s1)
    return s1, label, valid_mask, pred


class InferRequest(BaseModel):
    chip_id: str
    model: str


class InferResponse(BaseModel):
    chip_id: str
    model: str
    metrics: dict[str, float] | None
    n_valid_pixels: int
    n_nodata_pixels: int
    note: str | None = None


@router.post("", response_model=InferResponse)
def run_inference(req: InferRequest):
    s1, label, valid_mask, pred = _predict(req.chip_id, req.model)
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        # evaluate()'s per-metric formulas default to a "perfect" 1.0 when
        # there's nothing to compare (tp+fp+fn == 0) - real for a chip with
        # actual overlap, meaningless for a chip that's 100% nodata. Report
        # that honestly instead of a fabricated perfect score.
        return InferResponse(
            chip_id=req.chip_id,
            model=req.model,
            metrics=None,
            n_valid_pixels=0,
            n_nodata_pixels=int(s1_nodata_mask(s1).sum()),
            note="chip has zero valid pixels after masking (fully nodata) - no metric is meaningful here",
        )
    metrics = evaluate("flood-segmentation", pred[valid_mask], label[valid_mask])
    return InferResponse(
        chip_id=req.chip_id,
        model=req.model,
        metrics={k: round(v, 4) for k, v in metrics.items()},
        n_valid_pixels=n_valid,
        n_nodata_pixels=int(s1_nodata_mask(s1).sum()),
    )


@router.get("/{chip_id}/preview.png")
def inference_preview(chip_id: str, model: str = "rf"):
    s1, label, valid_mask, pred = _predict(chip_id, model)
    s1n = normalize_s1(s1)
    vv, vh = s1n[0], s1n[1]
    rgb = (np.stack([vv, vh, (vv + vh) / 2], axis=-1) * 255).astype(np.uint8)

    gt_img = rgb.copy()
    gt_img[label == 1] = [30, 120, 255]
    gt_img[label == -1] = [140, 140, 140]

    pred_img = rgb.copy()
    pred_img[pred == 1] = [30, 120, 255]
    pred_img[~valid_mask] = [140, 140, 140]

    combined = np.concatenate([rgb, gt_img, pred_img], axis=1)
    buf = io.BytesIO()
    Image.fromarray(combined).save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
