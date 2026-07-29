from src.ingestion.base import IngestionConnector


def get_connector(kind: str, **kwargs) -> IngestionConnector:
    """kind: 's3' | 'adls' | 'gcs' | 'http' | 'local'. kwargs are passed to
    the connector's constructor - see each module for its required fields."""
    if kind == "s3":
        from src.ingestion.s3_connector import S3Connector

        return S3Connector(**kwargs)
    if kind == "adls":
        from src.ingestion.adls_connector import ADLSConnector

        return ADLSConnector(**kwargs)
    if kind == "gcs":
        from src.ingestion.gcs_connector import GCSConnector

        return GCSConnector(**kwargs)
    if kind == "http":
        from src.ingestion.http_connector import HTTPBucketConnector

        return HTTPBucketConnector(**kwargs)
    if kind == "local":
        from src.ingestion.local_connector import LocalConnector

        return LocalConnector(**kwargs)
    raise ValueError(f"unknown ingestion connector kind: {kind}")
