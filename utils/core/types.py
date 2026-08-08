from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class DataSource(str, Enum):
    SENTINEL1 = "sentinel1"
    SENTINEL2 = "sentinel2"
    DEM = "dem"
    WORLDCOVER = "worldcover"
    OSM = "osm"
    SEN1FLOODS11 = "sen1floods11"


class DatasetStatus(str, Enum):
    RAW = "raw"
    PROCESSED = "processed"
    MISSING = "missing"


@dataclass(frozen=True)
class BBox:
    minx: float
    miny: float
    maxx: float
    maxy: float

    def as_list(self) -> list[float]:
        return [self.minx, self.miny, self.maxx, self.maxy]

    def intersects(self, other: "BBox") -> bool:
        return not (
            self.maxx < other.minx
            or self.minx > other.maxx
            or self.maxy < other.miny
            or self.miny > other.maxy
        )


@dataclass
class AOI:
    id: str
    bbox: BBox
    crs: str = "EPSG:4326"


@dataclass
class DatasetRecord:
    id: str
    source: DataSource
    path: str
    status: DatasetStatus = DatasetStatus.RAW
    acquisition_date: date | None = None
    bbox: BBox | None = None
    crs: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class Tile:
    aoi_id: str
    date: str
    bbox: BBox
    band_paths: dict[str, str] = field(default_factory=dict)
    label_path: str | None = None
