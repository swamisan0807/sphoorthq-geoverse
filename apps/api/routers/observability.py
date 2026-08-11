"""Run history, saved models, and the architecture diagram - the same
sources notebook 07's dashboard reads, exposed over HTTP."""

from fastapi import APIRouter

from utils.observability.diagram import ARCHITECTURE_DIAGRAM
from utils.observability.run_logger import load_recent_runs
from utils.registry import model_registry
from utils.registry.model_registry import KNOWN_MODELS

router = APIRouter(prefix="/api", tags=["observability"])


@router.get("/runs")
def list_runs(limit: int = 20):
    return load_recent_runs(limit=limit)


@router.get("/models")
def list_models():
    """The registry's current version of each known model - same source of
    truth apps/api/routers/registry.py uses (S3-backed when STATE_S3_BUCKET
    is set). This used to list datasets/processed/models/*.pkl directly off
    local disk - a flat, unversioned convenience copy each training
    notebook happens to also save there, not the actual registry - so it
    went stale/empty the moment local disk wasn't the source of truth
    anymore (e.g. right after a fresh clone, or datasets/ getting cleared)
    even though real registered versions existed in S3 the whole time."""
    out = []
    for name in KNOWN_MODELS:
        manifest = model_registry.get_current_manifest(name)
        if manifest is None:
            continue
        out.append(
            {
                "name": f"{name} (v{manifest['version']})",
                "size_bytes": manifest["size_bytes"],
                "modified_at": manifest["created_at"],
            }
        )
    return out


@router.get("/architecture")
def architecture():
    return {"mermaid": ARCHITECTURE_DIAGRAM}
