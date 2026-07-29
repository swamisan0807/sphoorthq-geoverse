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

from src.ingestion.base import IngestionConnector, RemoteObject


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
                objects.append(RemoteObject(key=obj["Key"], size_bytes=obj["Size"]))
        return objects

    def download_object(self, key: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(local_path))
        return local_path
