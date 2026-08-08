"""S3-backed state store for anything that must survive a stateless
redeploy (model registry, job/run history) - a Render deploy with no
persistent disk (see README "Persistence") wipes local files on every
restart, so those need a real external store, not just local JSON files.

Enabled by setting STATE_S3_BUCKET - every module that uses this checks
enabled() first and falls back to local disk under datasets/ when unset,
explicit and logged, same honesty pattern as utils/qml/ibm_quantum.py's
simulator fallback: a local-disk result is not the same durability claim
as an S3-backed one, so nothing here pretends otherwise.

Credentials: standard boto3 chain (env vars, ~/.aws/credentials, or an IAM
role) - nothing hardcoded, same as utils/ingestion/s3_connector.py.
"""

import json
import os
from pathlib import Path

_BUCKET = os.environ.get("STATE_S3_BUCKET")
_PREFIX = os.environ.get("STATE_S3_PREFIX", "").strip("/")
_REGION = os.environ.get("STATE_S3_REGION")

_client = None


def enabled() -> bool:
    return bool(_BUCKET)


def bucket_name() -> str | None:
    return _BUCKET


def _get_client():
    global _client
    if _client is None:
        import boto3

        _client = boto3.client("s3", region_name=_REGION) if _REGION else boto3.client("s3")
    return _client


def _full_key(key: str) -> str:
    return f"{_PREFIX}/{key}" if _PREFIX else key


def _strip_prefix(key: str) -> str:
    if _PREFIX and key.startswith(_PREFIX + "/"):
        return key[len(_PREFIX) + 1 :]
    return key


def read_json(key: str) -> dict | list | None:
    """Returns None for a missing key (not found is a normal, expected
    outcome here - e.g. no "current version" pointer registered yet), lets
    any other error (bad credentials, network) propagate for real. Also
    returns None for an existing-but-empty object - S3 "folder" placeholder
    keys (zero-byte objects some sync tools create for each path segment,
    e.g. a bare "reports/jobs/" key) read back as empty, not missing, so
    without this a listing that includes one would crash json.loads on an
    empty body instead of being skipped like a real missing key."""
    client = _get_client()
    try:
        obj = client.get_object(Bucket=_BUCKET, Key=_full_key(key))
    except client.exceptions.NoSuchKey:
        return None
    except client.exceptions.ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        raise
    body = obj["Body"].read()
    return json.loads(body) if body else None


def write_json(key: str, data) -> None:
    body = json.dumps(data, indent=2).encode("utf-8")
    _get_client().put_object(Bucket=_BUCKET, Key=_full_key(key), Body=body, ContentType="application/json")


def list_objects(prefix: str) -> list[dict]:
    """Real S3 listing under prefix - [{"key": <relative to STATE_S3_PREFIX>,
    "last_modified": epoch_float, "size_bytes": int}, ...], newest-last (S3
    doesn't sort for you) - callers that want recency-order sort themselves,
    the same as the local-disk path already does with st_mtime."""
    paginator = _get_client().get_paginator("list_objects_v2")
    out = []
    for page in paginator.paginate(Bucket=_BUCKET, Prefix=_full_key(prefix)):
        for obj in page.get("Contents", []):
            out.append(
                {
                    "key": _strip_prefix(obj["Key"]),
                    "last_modified": obj["LastModified"].timestamp(),
                    "size_bytes": obj["Size"],
                }
            )
    return out


def list_keys(prefix: str) -> list[str]:
    return [o["key"] for o in list_objects(prefix)]


def upload_file(local_path: Path, key: str) -> None:
    _get_client().upload_file(str(local_path), _BUCKET, _full_key(key))


def download_file(key: str, local_path: Path) -> Path:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _get_client().download_file(_BUCKET, _full_key(key), str(local_path))
    return local_path
