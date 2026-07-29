"""Cloud-agnostic ingestion interface.

Every connector pulls remote objects down to local disk (datasets/raw/...)
so the rest of the platform - notebooks, feature engineering, ML/QML - only
ever deals with local files and never needs to know which cloud a given
dataset originally came from.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RemoteObject:
    key: str
    size_bytes: int


class IngestionConnector(ABC):
    @abstractmethod
    def list_objects(self, prefix: str = "") -> list[RemoteObject]:
        """List objects under a prefix, recursively."""

    @abstractmethod
    def download_object(self, key: str, local_path: Path) -> Path:
        """Download a single object to local_path, creating parent dirs."""

    def download_prefix(
        self, prefix: str, local_dir: Path, skip_existing: bool = True
    ) -> list[Path]:
        """Download every object under prefix into local_dir, mirroring the
        remote key structure relative to prefix. Resumable via skip_existing."""
        downloaded = []
        for obj in self.list_objects(prefix):
            rel_key = obj.key[len(prefix):].lstrip("/") if prefix else obj.key
            local_path = local_dir / rel_key
            if skip_existing and local_path.exists() and local_path.stat().st_size == obj.size_bytes:
                downloaded.append(local_path)
                continue
            downloaded.append(self.download_object(obj.key, local_path))
        return downloaded
