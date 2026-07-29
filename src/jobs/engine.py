"""Notebook execution as a service - "run these notebooks like Databricks".

Triggers a real `jupyter nbconvert --execute --inplace` subprocess per job,
tracked in-memory plus persisted to datasets/reports/jobs/*.json so job
history survives a backend restart. No task queue (Celery/etc) - a
background thread per job is enough for a single-process demo backend.

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
from dataclasses import asdict, dataclass, field

from src.core.paths import JOBS_DIR, NOTEBOOKS_DIR, PROCESSED_DIR, PROJECT_ROOT

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


def _persist(job: JobRecord) -> None:
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
    from src.observability.run_logger import load_recent_runs
    from src.registry import model_registry

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
    path = JOBS_DIR / f"{job_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return JobRecord(**json.load(f))


def list_jobs(limit: int = 30) -> list[JobRecord]:
    with _lock:
        in_memory = {j.job_id: j for j in _jobs.values()}
    if JOBS_DIR.exists():
        for path in JOBS_DIR.glob("*.json"):
            if path.stem in in_memory:
                continue
            with open(path, encoding="utf-8") as f:
                record = JobRecord(**json.load(f))
            in_memory[record.job_id] = record
    return sorted(in_memory.values(), key=lambda j: j.started_at, reverse=True)[:limit]
