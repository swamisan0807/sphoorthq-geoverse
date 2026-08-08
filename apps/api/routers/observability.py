"""Run history, saved models, and the architecture diagram - the same
sources notebook 07's dashboard reads, exposed over HTTP."""

from fastapi import APIRouter

from utils.core.paths import PROCESSED_DIR
from utils.observability.diagram import ARCHITECTURE_DIAGRAM
from utils.observability.run_logger import load_recent_runs

router = APIRouter(prefix="/api", tags=["observability"])


@router.get("/runs")
def list_runs(limit: int = 20):
    return load_recent_runs(limit=limit)


@router.get("/models")
def list_models():
    models_dir = PROCESSED_DIR / "models"
    if not models_dir.exists():
        return []
    return [
        {
            "name": f.name,
            "size_bytes": f.stat().st_size,
            "modified_at": f.stat().st_mtime,
        }
        for f in sorted(models_dir.iterdir())
        if f.is_file()
    ]


@router.get("/architecture")
def architecture():
    return {"mermaid": ARCHITECTURE_DIAGRAM}
