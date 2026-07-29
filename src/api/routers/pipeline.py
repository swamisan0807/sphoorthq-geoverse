"""Auto-retrain-on-new-data: check whether the on-disk chip set has grown
since the last check, and optionally trigger real retraining jobs."""

from fastapi import APIRouter, Depends

from src.api.routers.auth import get_current_user
from src.jobs import engine as jobs_engine
from src.pipeline import auto_retrain

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


@router.get("/check-new-data")
def check_new_data():
    return auto_retrain.check_new_data()


@router.post("/retrain-if-new")
def retrain_if_new(username: str = Depends(get_current_user)):
    status = auto_retrain.check_new_data()
    if not status["has_new_data"]:
        return {"triggered": False, "job_ids": [], **status}

    auto_retrain.save_baseline(auto_retrain.current_chip_ids())
    job_ids = [
        jobs_engine.start_job("04_classical_ml", triggered_by=username).job_id,
        jobs_engine.start_job("09_patch_unet", triggered_by=username).job_id,
    ]
    return {"triggered": True, "job_ids": job_ids, **status}
