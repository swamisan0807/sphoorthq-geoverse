from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS_DIR = PROJECT_ROOT / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
PROCESSED_DIR = DATASETS_DIR / "processed"
METADATA_DIR = DATASETS_DIR / "metadata"
EXPORTS_DIR = DATASETS_DIR / "exports"
REPORTS_DIR = DATASETS_DIR / "reports"
THUMBNAILS_DIR = DATASETS_DIR / "thumbnails"

CONFIG_DIR = PROJECT_ROOT / "config"
