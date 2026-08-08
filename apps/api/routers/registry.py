"""Model registry endpoints: version history + restore ("roll back
inference to a prior version without retraining")."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from utils.registry import model_registry

router = APIRouter(prefix="/api/registry", tags=["registry"])

KNOWN_MODELS = ["classical_rf", "patch_unet"]


class RestoreRequest(BaseModel):
    version: int


@router.get("")
def list_models():
    out = []
    for name in KNOWN_MODELS:
        versions = model_registry.list_versions(name)
        out.append(
            {
                "model_name": name,
                "current_version": model_registry.get_current_version(name),
                "n_versions": len(versions),
            }
        )
    return out


@router.get("/{model_name}/versions")
def versions(model_name: str):
    if model_name not in KNOWN_MODELS:
        raise HTTPException(404, f"unknown model '{model_name}', expected one of {KNOWN_MODELS}")
    return {
        "model_name": model_name,
        "current_version": model_registry.get_current_version(model_name),
        "versions": model_registry.list_versions(model_name),
    }


@router.post("/{model_name}/restore")
def restore(model_name: str, req: RestoreRequest):
    if model_name not in KNOWN_MODELS:
        raise HTTPException(404, f"unknown model '{model_name}', expected one of {KNOWN_MODELS}")
    try:
        manifest = model_registry.restore_version(model_name, req.version)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return manifest
