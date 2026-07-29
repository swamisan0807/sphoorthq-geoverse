from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS_DIR = PROJECT_ROOT / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
PROCESSED_DIR = DATASETS_DIR / "processed"
METADATA_DIR = DATASETS_DIR / "metadata"
EXPORTS_DIR = DATASETS_DIR / "exports"
REPORTS_DIR = DATASETS_DIR / "reports"
RUNS_DIR = REPORTS_DIR / "runs"
JOBS_DIR = REPORTS_DIR / "jobs"
REGISTRY_DIR = PROCESSED_DIR / "models" / "registry"

CONFIG_DIR = PROJECT_ROOT / "config"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
