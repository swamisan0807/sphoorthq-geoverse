from pathlib import Path

from google.cloud import storage as gcs

from src.storage.base import StorageBackend


class GCSStorage(StorageBackend):
    def __init__(self, bucket_name: str):
        self.bucket = gcs.Client().bucket(bucket_name)

    def upload(self, local_path: Path, remote_key: str) -> str:
        blob = self.bucket.blob(remote_key)
        blob.upload_from_filename(str(local_path))
        return f"gs://{self.bucket.name}/{remote_key}"

    def download(self, remote_key: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.bucket.blob(remote_key).download_to_filename(str(local_path))
        return local_path

    def exists(self, remote_key: str) -> bool:
        return self.bucket.blob(remote_key).exists()

    def list_keys(self, prefix: str) -> list[str]:
        return [b.name for b in self.bucket.list_blobs(prefix=prefix)]
