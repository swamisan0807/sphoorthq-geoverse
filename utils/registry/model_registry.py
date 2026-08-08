"""Versioned model registry: every successful training run gets a new
immutable version (metrics + file snapshot), and a "current" pointer says
which version inference actually loads - restorable to any prior version
without retraining. This is what "hybrid version every time" / "restore in
model for inference" map to concretely.

Backed by S3 when STATE_S3_BUCKET is configured (utils/core/cloud_state.py) -
survives a stateless redeploy (e.g. Render with no persistent disk), which
local disk under REGISTRY_DIR cannot. Falls back to local disk when unset,
explicit and logged like every other cloud/local fallback in this codebase.

Storage layout (same shape either way - local paths relative to
REGISTRY_DIR, S3 keys relative to "processed/models/registry/", the same
relative path REGISTRY_DIR has under datasets/ - so S3 keys line up with
STATE_S3_PREFIX="datasets" and with datasets/processed/models/registry/ as
already laid out on local disk / in the project's S3 dataset mirror):
  <model_name>/v<N>/model.<ext>
  <model_name>/v<N>/manifest.json
  <model_name>/current.json          -> {"version": N}
"""

import json
import shutil
import time
from pathlib import Path

from utils.core import cloud_state
from utils.core.paths import REGISTRY_DIR

# model_name -> (source file the notebook actually saves, file extension)
MODEL_SOURCES: dict[str, tuple[Path, str]] = {}


def _s3_key(*parts: str) -> str:
    return "processed/models/registry/" + "/".join(parts)


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
    file_name = f"model.{extension}"

    manifest = {
        "model_name": model_name,
        "version": next_version,
        "created_at": time.time(),
        "notebook": notebook,
        "registered_by": registered_by,
        "metrics": metrics,
        "file": file_name,
        "size_bytes": source_path.stat().st_size,
    }

    if cloud_state.enabled():
        cloud_state.upload_file(source_path, _s3_key(model_name, f"v{next_version}", file_name))
        cloud_state.write_json(_s3_key(model_name, f"v{next_version}", "manifest.json"), manifest)
        cloud_state.write_json(_s3_key(model_name, "current.json"), {"version": next_version})
    else:
        version_dir = _model_dir(model_name) / f"v{next_version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        dest = version_dir / file_name
        shutil.copy2(source_path, dest)
        with open(version_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        with open(_current_pointer_path(model_name), "w", encoding="utf-8") as f:
            json.dump({"version": next_version}, f)

    return manifest


def list_versions(model_name: str) -> list[dict]:
    if cloud_state.enabled():
        manifests = []
        for key in cloud_state.list_keys(_s3_key(model_name) + "/"):
            if key.endswith("manifest.json"):
                m = cloud_state.read_json(key)
                if m:
                    manifests.append(m)
        return sorted(manifests, key=lambda m: m["version"])

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
    if cloud_state.enabled():
        data = cloud_state.read_json(_s3_key(model_name, "current.json"))
        return data["version"] if data else None

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
    """Returns a local, loadable path to the current version's model file.
    S3-backed: downloads into a local cache on first use (or if the cached
    copy's size doesn't match the manifest) - S3 objects aren't directly
    openable by pickle.load/torch.load, so local disk here is just working
    scratch space, not the source of truth. Local-backed: the file is
    already the source of truth, return it directly."""
    manifest = get_current_manifest(model_name)
    if manifest is None:
        return None

    if cloud_state.enabled():
        cache_path = REGISTRY_DIR / "_s3_cache" / model_name / f"v{manifest['version']}" / manifest["file"]
        if not cache_path.exists() or cache_path.stat().st_size != manifest["size_bytes"]:
            cloud_state.download_file(_s3_key(model_name, f"v{manifest['version']}", manifest["file"]), cache_path)
        return cache_path

    return _model_dir(model_name) / f"v{manifest['version']}" / manifest["file"]


def restore_version(model_name: str, version: int) -> dict:
    """Points 'current' at an already-registered version - no retraining,
    inference immediately starts using it."""
    versions = {m["version"]: m for m in list_versions(model_name)}
    if version not in versions:
        raise ValueError(f"model '{model_name}' has no version {version} (has: {sorted(versions)})")

    if cloud_state.enabled():
        cloud_state.write_json(_s3_key(model_name, "current.json"), {"version": version})
    else:
        with open(_current_pointer_path(model_name), "w", encoding="utf-8") as f:
            json.dump({"version": version}, f)
    return versions[version]
