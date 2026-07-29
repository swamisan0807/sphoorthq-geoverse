from fastapi import APIRouter

from src.api.schemas import ExperimentSummary

router = APIRouter()

# TODO: replace with real training run logs once src/ai/classic and
# src/ai/quantum training loops write results to datasets/reports.
_STUB_EXPERIMENTS = [
    ExperimentSummary(
        id="classic-unet-v1",
        model_type="classic",
        objective="flood-segmentation",
        metrics={"iou": 0.0, "f1": 0.0, "boundary_f1": 0.0},
        created_at="2026-07-18T00:00:00Z",
    ),
    ExperimentSummary(
        id="quantum-hybrid-v1",
        model_type="quantum",
        objective="flood-segmentation",
        metrics={"iou": 0.0, "f1": 0.0, "boundary_f1": 0.0},
        created_at="2026-07-18T00:00:00Z",
    ),
]


@router.get("", response_model=list[ExperimentSummary])
def list_experiments():
    return _STUB_EXPERIMENTS
