import uuid

from fastapi import APIRouter, HTTPException

from src.api.schemas import InferenceJob, InferenceRequest

router = APIRouter()

# TODO: replace in-memory dict with a real job queue once src/ai pipeline exists.
_JOBS: dict[str, InferenceJob] = {}


@router.post("/run", response_model=InferenceJob)
def run_inference(req: InferenceRequest):
    job_id = str(uuid.uuid4())
    job = InferenceJob(job_id=job_id, status="queued")
    _JOBS[job_id] = job
    return job


@router.get("/{job_id}", response_model=InferenceJob)
def get_job(job_id: str):
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job
