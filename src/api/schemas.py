from typing import Literal, Optional

from pydantic import BaseModel


class DatasetSummary(BaseModel):
    id: str
    source: str  # sentinel1, sentinel2, dem, worldcover, osm, sen1floods11
    acquisition_date: Optional[str] = None
    footprint_bbox: Optional[list[float]] = None  # [minx, miny, maxx, maxy]
    crs: Optional[str] = None
    thumbnail_url: Optional[str] = None
    status: Literal["raw", "processed", "missing"] = "raw"


class InferenceRequest(BaseModel):
    aoi_bbox: list[float]  # [minx, miny, maxx, maxy]
    date: Optional[str] = None
    model: Literal["classic-unet", "quantum-hybrid"] = "classic-unet"
    objective: str = "flood-segmentation"


class InferenceJob(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    result_tile_url: Optional[str] = None
    metrics: Optional[dict] = None


class ExperimentSummary(BaseModel):
    id: str
    model_type: Literal["classic", "quantum"]
    objective: str
    metrics: dict
    created_at: str
