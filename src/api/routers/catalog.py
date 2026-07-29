from fastapi import APIRouter, HTTPException

from src.api.schemas import DatasetSummary
from src.catalog.scanner import mark_processed, scan_raw_datasets

router = APIRouter()


def _to_summary(record) -> DatasetSummary:
    return DatasetSummary(
        id=record.id,
        source=record.source.value,
        acquisition_date=record.acquisition_date.isoformat() if record.acquisition_date else None,
        footprint_bbox=record.bbox.as_list() if record.bbox else None,
        crs=record.crs,
        status=record.status.value,
    )


@router.get("", response_model=list[DatasetSummary])
def list_datasets():
    records = mark_processed(scan_raw_datasets())
    return [_to_summary(r) for r in records]


@router.get("/{dataset_id}", response_model=DatasetSummary)
def get_dataset(dataset_id: str):
    records = mark_processed(scan_raw_datasets())
    for r in records:
        if r.id == dataset_id:
            return _to_summary(r)
    raise HTTPException(status_code=404, detail="dataset not found")
