from abc import ABC, abstractmethod
from pathlib import Path


class StorageBackend(ABC):
    @abstractmethod
    def upload(self, local_path: Path, remote_key: str) -> str:
        """Returns the remote URI."""

    @abstractmethod
    def download(self, remote_key: str, local_path: Path) -> Path:
        ...

    @abstractmethod
    def exists(self, remote_key: str) -> bool:
        ...

    @abstractmethod
    def list_keys(self, prefix: str) -> list[str]:
        ...
