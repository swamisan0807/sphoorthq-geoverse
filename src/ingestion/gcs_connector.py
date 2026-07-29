"""Google Cloud Storage ingestion connector.

anonymous=True uses an unauthenticated client for public buckets (this is
how datasets/raw/sen1floods11 was actually pulled - gs://sen1floods11 is
public, no GCP credentials needed). Otherwise falls back to Application
Default Credentials (gcloud auth, service account key via
GOOGLE_APPLICATION_CREDENTIALS, or workload identity).
"""

from pathlib import Path

from google.cloud import storage

from src.ingestion.base import IngestionConnector, RemoteObject


class GCSConnector(IngestionConnector):
    def __init__(self, bucket: str, anonymous: bool = False, project: str | None = None):
        client = storage.Client.create_anonymous_client() if anonymous else storage.Client(project=project)
        self.bucket = client.bucket(bucket)

    def list_objects(self, prefix: str = "") -> list[RemoteObject]:
        return [
            RemoteObject(key=blob.name, size_bytes=blob.size or 0)
            for blob in self.bucket.list_blobs(prefix=prefix)
        ]

    def download_object(self, key: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.bucket.blob(key).download_to_filename(str(local_path))
        return local_path
