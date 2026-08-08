"""AWS S3 ingestion connector.

Credentials resolved the standard boto3 way (env vars AWS_ACCESS_KEY_ID /
AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN, ~/.aws/credentials, or an IAM
role) - nothing is hardcoded here. Pass anonymous=True for public buckets
(e.g. open Earth-observation buckets) that don't need credentials at all.
"""

from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

from utils.ingestion.base import IngestionConnector, RemoteObject


class S3Connector(IngestionConnector):
    def __init__(self, bucket: str, anonymous: bool = False, **client_kwargs):
        self.bucket = bucket
        if anonymous:
            client_kwargs.setdefault("config", Config(signature_version=UNSIGNED))
        self.client = boto3.client("s3", **client_kwargs)

    def list_objects(self, prefix: str = "") -> list[RemoteObject]:
        paginator = self.client.get_paginator("list_objects_v2")
        objects = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith("/"):
                    # Zero-byte "folder" placeholder key (created by some
                    # sync tools for every path segment, S3 itself has no
                    # real directories) - not a real object to download.
                    continue
                objects.append(RemoteObject(key=obj["Key"], size_bytes=obj["Size"]))
        return objects

    def download_object(self, key: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(local_path))
        return local_path

    def upload_object(self, local_path: Path, key: str) -> str:
        """Uploads one local file to `key`. S3 has no real directories - the
        key's "/" segments are created implicitly, no separate mkdir-style
        step needed, unlike a local or SFTP filesystem."""
        self.client.upload_file(str(local_path), self.bucket, key)
        return key

    def upload_prefix(self, local_dir: Path, prefix: str, skip_existing: bool = True) -> list[str]:
        """Uploads every file under local_dir, mirroring its structure under
        `prefix` in the bucket - the upload-direction mirror of
        download_prefix(). skip_existing compares remote size (a real
        HEAD-equivalent list, not a guess) so a partial upload can resume
        without re-uploading files that already made it across."""
        prefix = prefix if prefix.endswith("/") or not prefix else prefix + "/"
        existing = {o.key: o.size_bytes for o in self.list_objects(prefix)} if skip_existing else {}
        uploaded = []
        for local_path in sorted(local_dir.rglob("*")):
            if not local_path.is_file():
                continue
            rel_key = prefix + str(local_path.relative_to(local_dir)).replace("\\", "/")
            if skip_existing and existing.get(rel_key) == local_path.stat().st_size:
                continue
            self.upload_object(local_path, rel_key)
            uploaded.append(rel_key)
        return uploaded
