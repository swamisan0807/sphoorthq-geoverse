"""Versioned model registry: every successful training run gets a new
immutable version (metrics + file snapshot), and a "current" pointer says
which version inference actually loads - restorable to any prior version
without retraining. This is what "hybrid version every time" / "restore in
model for inference" map to concretely.

Storage layout under REGISTRY_DIR:
  <model_name>/v<N>/model.<ext>
  <model_name>/v<N>/manifest.json
  <model_name>/current.json          -> {"version": N}
"""

import json
import shutil
import time
from pathlib import Path

from src.core.paths import REGISTRY_DIR

# model_name -> (source file the notebook actually saves, file extension)
MODEL_SOURCES: dict[str, tuple[Path, str]] = {}


def _model_dir(model_name: str) -> Path:
    return REGISTRY_DIR / model_name


def _current_pointer_path(model_name: str) -> Path:
    return _model_dir(model_name) / "current.json"


def register_version(
    model_name: str,
    source_path: Path,
    metrics: dict,
    notebook: str,
    registered_by: str,
    extension: str,
) -> dict:
    """Copies source_path into a new immutable version slot, writes its
    manifest, and makes it the new current version."""
    if not source_path.exists():
        raise FileNotFoundError(f"model source file not found: {source_path}")

    versions = list_versions(model_name)
    next_version = (versions[-1]["version"] + 1) if versions else 1

    version_dir = _model_dir(model_name) / f"v{next_version}"
    version_dir.mkdir(parents=True, exist_ok=True)
    dest = version_dir / f"model.{extension}"
    shutil.copy2(source_path, dest)

    manifest = {
        "model_name": model_name,
        "version": next_version,
        "created_at": time.time(),
        "notebook": notebook,
        "registered_by": registered_by,
        "metrics": metrics,
        "file": dest.name,
        "size_bytes": dest.stat().st_size,
    }
    with open(version_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(_current_pointer_path(model_name), "w", encoding="utf-8") as f:
        json.dump({"version": next_version}, f)

    return manifest


def list_versions(model_name: str) -> list[dict]:
    model_dir = _model_dir(model_name)
    if not model_dir.exists():
        return []
    manifests = []
    for version_dir in sorted(model_dir.glob("v*")):
        manifest_path = version_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                manifests.append(json.load(f))
    return sorted(manifests, key=lambda m: m["version"])


def get_current_version(model_name: str) -> int | None:
    pointer_path = _current_pointer_path(model_name)
    if not pointer_path.exists():
        return None
    with open(pointer_path, encoding="utf-8") as f:
        return json.load(f)["version"]


def get_current_manifest(model_name: str) -> dict | None:
    version = get_current_version(model_name)
    if version is None:
        return None
    for manifest in list_versions(model_name):
        if manifest["version"] == version:
            return manifest
    return None


def get_current_path(model_name: str) -> Path | None:
    manifest = get_current_manifest(model_name)
    if manifest is None:
        return None
    return _model_dir(model_name) / f"v{manifest['version']}" / manifest["file"]


def restore_version(model_name: str, version: int) -> dict:
    """Points 'current' at an already-registered version - no retraining,
    inference immediately starts using it."""
    versions = {m["version"]: m for m in list_versions(model_name)}
    if version not in versions:
        raise ValueError(f"model '{model_name}' has no version {version} (has: {sorted(versions)})")
    with open(_current_pointer_path(model_name), "w", encoding="utf-8") as f:
        json.dump({"version": version}, f)
    return versions[version]
