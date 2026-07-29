from pathlib import Path

import boto3

from src.storage.base import StorageBackend


class S3Storage(StorageBackend):
    def __init__(self, bucket: str, prefix: str = "", **client_kwargs):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.client = boto3.client("s3", **client_kwargs)

    def _key(self, remote_key: str) -> str:
        return f"{self.prefix}/{remote_key}" if self.prefix else remote_key

    def upload(self, local_path: Path, remote_key: str) -> str:
        key = self._key(remote_key)
        self.client.upload_file(str(local_path), self.bucket, key)
        return f"s3://{self.bucket}/{key}"

    def download(self, remote_key: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, self._key(remote_key), str(local_path))
        return local_path

    def exists(self, remote_key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(remote_key))
            return True
        except ClientError:
            return False

    def list_keys(self, prefix: str) -> list[str]:
        paginator = self.client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self._key(prefix)):
            keys.extend(obj["Key"] for obj in page.get("Contents", []))
        return keys
