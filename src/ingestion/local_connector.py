"""Local filesystem 'connector' - lets the pipeline treat an already-local
directory (or a network share mounted as a drive) through the same
IngestionConnector interface as a real cloud source."""

import shutil
from pathlib import Path

from src.ingestion.base import IngestionConnector, RemoteObject


class LocalConnector(IngestionConnector):
    def __init__(self, root: Path):
        self.root = Path(root)

    def list_objects(self, prefix: str = "") -> list[RemoteObject]:
        base = self.root / prefix
        if not base.exists():
            return []
        return [
            RemoteObject(key=str(p.relative_to(self.root)).replace("\\", "/"), size_bytes=p.stat().st_size)
            for p in base.rglob("*")
            if p.is_file()
        ]

    def download_object(self, key: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.root / key, local_path)
        return local_path
