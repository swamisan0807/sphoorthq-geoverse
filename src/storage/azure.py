from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

from src.storage.base import StorageBackend


class AzureBlobStorage(StorageBackend):
    def __init__(self, account_url: str, container: str):
        self.container = container
        self.client = BlobServiceClient(
            account_url=account_url, credential=DefaultAzureCredential()
        ).get_container_client(container)

    def upload(self, local_path: Path, remote_key: str) -> str:
        with open(local_path, "rb") as f:
            self.client.upload_blob(name=remote_key, data=f, overwrite=True)
        return f"{self.client.url}/{remote_key}"

    def download(self, remote_key: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(self.client.download_blob(remote_key).readall())
        return local_path

    def exists(self, remote_key: str) -> bool:
        return self.client.get_blob_client(remote_key).exists()

    def list_keys(self, prefix: str) -> list[str]:
        return [b.name for b in self.client.list_blobs(name_starts_with=prefix)]
