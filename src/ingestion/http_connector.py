"""Generic public-bucket connector via the S3/GCS-compatible XML listing API.

No cloud SDK, no credentials - works against any publicly readable bucket
that exposes the standard ListBucketResult XML (both AWS S3 and GCS do).
This is exactly how datasets/raw/sen1floods11 was pulled from
gs://sen1floods11 in this project: plain HTTPS GET + XML pagination.
Prefer S3Connector/GCSConnector when you actually have credentials and
want their richer APIs - this one is for "public bucket, no auth, no SDK".
"""

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from src.ingestion.base import IngestionConnector, RemoteObject

_NS = {"s3": "http://doc.s3.amazonaws.com/2006-03-01"}


class HTTPBucketConnector(IngestionConnector):
    def __init__(self, base_url: str):
        self.base_url = base_url if base_url.endswith("/") else base_url + "/"

    def list_objects(self, prefix: str = "") -> list[RemoteObject]:
        objects: list[RemoteObject] = []
        marker = ""
        while True:
            url = f"{self.base_url}?prefix={urllib.parse.quote(prefix)}"
            if marker:
                url += f"&marker={urllib.parse.quote(marker)}"
            with urllib.request.urlopen(url, timeout=30) as resp:
                root = ET.fromstring(resp.read())

            contents = root.findall("s3:Contents", _NS)
            for c in contents:
                key = c.find("s3:Key", _NS).text
                size = int(c.find("s3:Size", _NS).text)
                objects.append(RemoteObject(key=key, size_bytes=size))

            truncated = root.find("s3:IsTruncated", _NS).text == "true"
            if not truncated or not contents:
                break
            marker = contents[-1].find("s3:Key", _NS).text
        return objects

    def download_object(self, key: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        url = self.base_url + urllib.parse.quote(key)
        tmp = local_path.with_suffix(local_path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(local_path)
        return local_path
