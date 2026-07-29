from functools import lru_cache
from pathlib import Path

import yaml

from src.core.paths import CONFIG_DIR


@lru_cache(maxsize=None)
def load_config(name: str) -> dict:
    """Load a YAML file from config/ by name, e.g. load_config('aoi')."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no such config file: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config_path(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
