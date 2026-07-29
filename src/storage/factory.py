from pathlib import Path

from src.core.paths import DATASETS_DIR
from src.storage.base import StorageBackend
from src.storage.local import LocalStorage


def get_storage(backend: str = "local", **kwargs) -> StorageBackend:
    if backend == "local":
        return LocalStorage(root=kwargs.get("root", DATASETS_DIR))
    if backend == "s3":
        from src.storage.s3 import S3Storage

        return S3Storage(**kwargs)
    if backend == "azure":
        from src.storage.azure import AzureBlobStorage

        return AzureBlobStorage(**kwargs)
    if backend == "gcs":
        from src.storage.gcs import GCSStorage

        return GCSStorage(**kwargs)
    raise ValueError(f"unknown storage backend: {backend}")
