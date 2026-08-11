"""Notebook execution as a service - "run these notebooks like Databricks".

Triggers a real `jupyter nbconvert --execute --inplace` subprocess per job,
tracked in-memory plus persisted to datasets/reports/jobs/*.json (local) or
"reports/jobs/" in STATE_S3_BUCKET when configured (utils/core/cloud_state.py)
so job history survives a backend restart *and* a stateless redeploy. No
task queue (Celery/etc) - a background thread per job is enough for a
single-process demo backend.

Honest limitation: nbconvert only rewrites the .ipynb file (and this
process only captures nbconvert's own stdout/stderr) once the whole
notebook finishes - there is no live cell-by-cell output stream here, only
queued -> running -> success/failed with a final log tail.
"""

import json
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

from utils.core import cloud_state
from utils.core.paths import JOBS_DIR, NOTEBOOKS_DIR, PROCESSED_DIR, PROJECT_ROOT

_LOG_TAIL_CHARS = 8000

# notebooks that produce a model file worth registering, once their job succeeds
_TRAINING_NOTEBOOKS = {
    "04_classical_ml": {
        "model_name": "classical_rf",
        "source": PROCESSED_DIR / "models" / "classical_rf_v1.pkl",
        "extension": "pkl",
    },
    "09_patch_unet": {
        "model_name": "patch_unet",
        "source": PROCESSED_DIR / "models" / "patch_unet_v1.pt",
        "extension": "pt",
    },
}


def list_notebooks() -> list[str]:
    return sorted(p.stem for p in NOTEBOOKS_DIR.glob("*.ipynb"))


@dataclass
class JobRecord:
    job_id: str
    notebook: str  # display label - a notebook stem for kind="notebook", or a free-form label otherwise
    status: str  # queued | running | success | failed
    triggered_by: str
    started_at: float
    ended_at: float | None = None
    returncode: int | None = None
    log_tail: str = ""
    kind: str = "notebook"  # "notebook" | "quantum"
    extra: dict = field(default_factory=dict)


_jobs: dict[str, JobRecord] = {}
_lock = threading.Lock()

# Read-through cache for job records fetched from S3, keyed by job_id - only
# ever populated with *terminal* records (success/failed), which can never
# change again, so caching them forever is safe. Without this, list_jobs()
# re-fetches every job individually from S3 (one GET each, no batching) on
# every single call - fine for a handful of jobs, but real end-to-end
# testing quickly racks up dozens of job records, and every Jobs-page poll
# or /api/compare/data call was re-paying that full N-GET cost from a cold
# process (no warm _jobs cache) every time. queued/running jobs are
# deliberately never cached here so a status change is still picked up.
_TERMINAL_STATUSES = {"success", "failed"}
_s3_job_cache: dict[str, JobRecord] = {}


def _persist(job: JobRecord) -> None:
    if cloud_state.enabled():
        cloud_state.write_json(f"reports/jobs/{job.job_id}.json", asdict(job))
        return
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    with open(JOBS_DIR / f"{job.job_id}.json", "w", encoding="utf-8") as f:
        json.dump(asdict(job), f, indent=2)


def _run(job: JobRecord) -> None:
    notebook_path = NOTEBOOKS_DIR / f"{job.notebook}.ipynb"
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        str(notebook_path),
    ]
    job.status = "running"
    _persist(job)

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=1800,
        )
        combined = (proc.stdout or "") + (proc.stderr or "")
        job.log_tail = combined[-_LOG_TAIL_CHARS:]
        job.returncode = proc.returncode
        job.status = "success" if proc.returncode == 0 else "failed"
        if job.status == "success":
            _maybe_register_model(job)
    except subprocess.TimeoutExpired as e:
        job.log_tail = f"job exceeded 1800s timeout and was killed\n{(e.stdout or '')[-_LOG_TAIL_CHARS:]}"
        job.status = "failed"
        job.returncode = -1
    except Exception as e:  # noqa: BLE001 - record failure, don't crash the thread
        job.log_tail = f"{type(e).__name__}: {e}"
        job.status = "failed"
        job.returncode = -1
    finally:
        job.ended_at = time.time()
        with _lock:
            _persist(job)


def _maybe_register_model(job: JobRecord) -> None:
    spec = _TRAINING_NOTEBOOKS.get(job.notebook)
    if spec is None:
        return
    from utils.observability.run_logger import load_recent_runs
    from utils.registry import model_registry

    metrics = {}
    for run in load_recent_runs(limit=10):
        if run["run_name"] == job.notebook and run["started_at"] >= job.started_at:
            metrics = {k: v for k, v in run.get("metrics", {}).items() if isinstance(v, (int, float))}
            break

    model_registry.register_version(
        model_name=spec["model_name"],
        source_path=spec["source"],
        metrics=metrics,
        notebook=job.notebook,
        registered_by=job.triggered_by,
        extension=spec["extension"],
    )


def start_job(notebook: str, triggered_by: str) -> JobRecord:
    if notebook not in list_notebooks():
        raise ValueError(f"unknown notebook '{notebook}', available: {list_notebooks()}")

    job = JobRecord(
        job_id=str(uuid.uuid4()),
        notebook=notebook,
        status="queued",
        triggered_by=triggered_by,
        started_at=time.time(),
    )
    with _lock:
        _jobs[job.job_id] = job
        _persist(job)

    thread = threading.Thread(target=_run, args=(job,), daemon=True)
    thread.start()
    return job


def record_finished_job(
    label: str,
    triggered_by: str,
    started_at: float,
    ended_at: float,
    status: str,
    extra: dict,
    kind: str = "quantum",
) -> JobRecord:
    """For work that already ran synchronously in the request handler
    (e.g. a quantum kernel SVM call) - logs it into the same job history
    the notebook Jobs engine uses, so it shows up in one unified Jobs view."""
    job = JobRecord(
        job_id=str(uuid.uuid4()),
        notebook=label,
        status=status,
        triggered_by=triggered_by,
        started_at=started_at,
        ended_at=ended_at,
        returncode=0 if status == "success" else -1,
        kind=kind,
        extra=extra,
    )
    with _lock:
        _jobs[job.job_id] = job
        _persist(job)
    return job


def get_job(job_id: str) -> JobRecord | None:
    with _lock:
        if job_id in _jobs:
            return _jobs[job_id]
    if job_id in _s3_job_cache:
        return _s3_job_cache[job_id]
    if cloud_state.enabled():
        data = cloud_state.read_json(f"reports/jobs/{job_id}.json")
        if not data:
            return None
        record = JobRecord(**data)
        if record.status in _TERMINAL_STATUSES:
            _s3_job_cache[job_id] = record
        return record
    path = JOBS_DIR / f"{job_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return JobRecord(**json.load(f))


def list_jobs(limit: int = 30, triggered_by: str | None = None) -> list[JobRecord]:
    """triggered_by filters to jobs started by that user_id (the same value
    every job record already logs - see start_job()/trigger_quantum_job()) -
    filter first, then apply limit, so "my jobs" always returns up to
    `limit` of *mine*, not `limit` overall with mine sparsely mixed in."""
    with _lock:
        in_memory = {j.job_id: j for j in _jobs.values()}

    if cloud_state.enabled():
        to_fetch = []
        for obj in cloud_state.list_objects("reports/jobs/"):
            if not obj["key"].endswith(".json"):
                continue
            job_id = obj["key"].removeprefix("reports/jobs/").removesuffix(".json")
            if job_id in in_memory:
                continue
            if job_id in _s3_job_cache:
                in_memory[job_id] = _s3_job_cache[job_id]
                continue
            to_fetch.append((job_id, obj["key"]))

        # Real, separate network round trips (no batch-get in S3) - fetch
        # concurrently rather than one-by-one, since a page load waiting on
        # dozens of sequential GETs is the whole problem this cache exists
        # to avoid on a cold process (empty cache) or after many new jobs.
        if to_fetch:
            with ThreadPoolExecutor(max_workers=min(16, len(to_fetch))) as pool:
                fetched = pool.map(lambda t: (t[0], cloud_state.read_json(t[1])), to_fetch)
            for job_id, data in fetched:
                if not data:
                    continue
                record = JobRecord(**data)
                in_memory[job_id] = record
                if record.status in _TERMINAL_STATUSES:
                    _s3_job_cache[job_id] = record
    elif JOBS_DIR.exists():
        for path in JOBS_DIR.glob("*.json"):
            if path.stem in in_memory:
                continue
            with open(path, encoding="utf-8") as f:
                record = JobRecord(**json.load(f))
            in_memory[record.job_id] = record

    jobs = in_memory.values()
    if triggered_by is not None:
        jobs = (j for j in jobs if j.triggered_by == triggered_by)
    return sorted(jobs, key=lambda j: j.started_at, reverse=True)[:limit]
