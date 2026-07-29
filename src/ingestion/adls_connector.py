"""Azure Data Lake Storage Gen2 ingestion connector.

Uses the dedicated ADLS Gen2 SDK (azure-storage-file-datalake), not the
plain Blob API - this gets real directory semantics (hierarchical
namespace) rather than treating ADLS as flat blob storage with '/' in
key names. Auth via DefaultAzureCredential (env vars, managed identity,
az CLI login, etc. - see Azure SDK docs for the full chain).
"""

from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.storage.filedatalake import DataLakeServiceClient

from src.ingestion.base import IngestionConnector, RemoteObject


class ADLSConnector(IngestionConnector):
    def __init__(self, account_url: str, filesystem: str):
        self.service_client = DataLakeServiceClient(
            account_url=account_url, credential=DefaultAzureCredential()
        )
        self.filesystem_client = self.service_client.get_file_system_client(filesystem)

    def list_objects(self, prefix: str = "") -> list[RemoteObject]:
        objects = []
        for path in self.filesystem_client.get_paths(path=prefix, recursive=True):
            if not path.is_directory:
                objects.append(RemoteObject(key=path.name, size_bytes=path.content_length or 0))
        return objects

    def download_object(self, key: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        file_client = self.filesystem_client.get_file_client(key)
        with open(local_path, "wb") as f:
            download = file_client.download_file()
            download.readinto(f)
        return local_path
