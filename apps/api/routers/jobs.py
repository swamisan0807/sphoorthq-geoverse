"""Databricks-style "run this notebook" over HTTP - triggers a real
notebook execution job and lets you poll its status."""

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from apps.api.routers.auth import get_current_user
from utils.jobs import engine

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class RunRequest(BaseModel):
    notebook: str


@router.get("/notebooks")
def notebooks():
    return engine.list_notebooks()


@router.get("")
def jobs(limit: int = 30, mine: bool = False, username: str = Depends(get_current_user)):
    """mine=true filters to only the requesting user's own jobs (user_id ==
    triggered_by) - off by default, preserving the existing shared job board
    (everyone's jobs, `triggered_by` column) rather than silently hiding
    other users' runs from a page that's always shown that column."""
    triggered_by = username if mine else None
    return [asdict(j) for j in engine.list_jobs(limit=limit, triggered_by=triggered_by)]


@router.post("")
def run_notebook(req: RunRequest, username: str = Depends(get_current_user)):
    try:
        job = engine.start_job(req.notebook, triggered_by=username)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return asdict(job)


@router.get("/{job_id}")
def job_detail(job_id: str, username: str = Depends(get_current_user)):
    job = engine.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"job '{job_id}' not found")
    return asdict(job)
