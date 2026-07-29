"""Databricks-style "run this notebook" over HTTP - triggers a real
notebook execution job and lets you poll its status."""

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.routers.auth import get_current_user
from src.jobs import engine

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class RunRequest(BaseModel):
    notebook: str


@router.get("/notebooks")
def notebooks():
    return engine.list_notebooks()


@router.get("")
def jobs(limit: int = 30):
    return [asdict(j) for j in engine.list_jobs(limit=limit)]


@router.post("")
def run_notebook(req: RunRequest, username: str = Depends(get_current_user)):
    try:
        job = engine.start_job(req.notebook, triggered_by=username)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return asdict(job)


@router.get("/{job_id}")
def job_detail(job_id: str):
    job = engine.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"job '{job_id}' not found")
    return asdict(job)
