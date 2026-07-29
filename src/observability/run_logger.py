"""Structured pipeline observability: every notebook stage (ingest, process,
feature-engineer, train classical, train QML-IBM, train QML-Braket,
evaluate) reports through the same RunLogger, producing one JSON record per
run under datasets/reports/runs/. This is what lets you answer "what
actually happened in this pipeline run, how long did each stage take, what
did it operate on, and what came out" after the fact - the observability
requirement - without bolting on a separate monitoring stack.
"""

import json
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

from src.core.paths import RUNS_DIR


@dataclass
class StageRecord:
    name: str
    started_at: float
    ended_at: float | None = None
    duration_s: float | None = None
    status: str = "running"  # running | ok | failed
    metrics: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class RunRecord:
    run_id: str
    run_name: str
    started_at: float
    ended_at: float | None = None
    stages: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


class RunLogger:
    def __init__(self, run_name: str):
        self.run = RunRecord(run_id=str(uuid.uuid4()), run_name=run_name, started_at=time.time())

    @contextmanager
    def stage(self, name: str):
        record = StageRecord(name=name, started_at=time.time())
        self.run.stages.append(record)
        print(f"[{self.run.run_name}] -> {name} ...")
        try:
            yield record
            record.status = "ok"
        except Exception as e:  # noqa: BLE001 - re-raised after logging
            record.status = "failed"
            record.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            raise
        finally:
            record.ended_at = time.time()
            record.duration_s = round(record.ended_at - record.started_at, 3)
            symbol = "OK" if record.status == "ok" else "FAILED"
            print(f"[{self.run.run_name}] <- {name} [{symbol}] {record.duration_s}s {record.metrics}")

    def log_metric(self, key: str, value) -> None:
        self.run.metrics[key] = value

    def log_metrics(self, metrics: dict) -> None:
        self.run.metrics.update(metrics)

    def finalize(self) -> str:
        self.run.ended_at = time.time()
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RUNS_DIR / f"{self.run.run_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.run), f, indent=2)
        total = round(self.run.ended_at - self.run.started_at, 3)
        print(f"[{self.run.run_name}] run complete in {total}s -> {out_path}")
        return str(out_path)

    def summary_table(self) -> list[dict]:
        return [
            {"stage": s.name, "status": s.status, "duration_s": s.duration_s, **s.metrics}
            for s in self.run.stages
        ]


def load_recent_runs(limit: int = 20) -> list[dict]:
    """Reads back run logs for the observability dashboard notebook."""
    if not RUNS_DIR.exists():
        return []
    files = sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    runs = []
    for f in files[:limit]:
        with open(f, encoding="utf-8") as fh:
            runs.append(json.load(fh))
    return runs
