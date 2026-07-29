"""Scans datasets/raw and builds DatasetRecord entries.

No GDAL/rasterio dependency here on purpose - this only needs to run fast
over folder layout + small XML/manifest files to produce a catalog. Actual
raster metadata (real bbox/CRS) gets filled in by src/processing during
co-registration, once a dataset has a processed/ counterpart.
"""

import re
from datetime import date
from pathlib import Path
from xml.etree import ElementTree as ET

from src.core.paths import PROCESSED_DIR, RAW_DIR
from src.core.types import DataSource, DatasetRecord, DatasetStatus

_SAFE_DATE_RE = re.compile(r"(\d{4})(\d{2})(\d{2})[Tt]\d{6}")


def _parse_date_from_name(name: str) -> date | None:
    m = _SAFE_DATE_RE.search(name)
    if not m:
        return None
    y, mo, d = (int(g) for g in m.groups())
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def _scan_sentinel1(root: Path) -> DatasetRecord | None:
    manifest = root / "manifest.safe"
    if not manifest.exists():
        return None
    measurement_dir = root / "measurement"
    tiffs = sorted(measurement_dir.glob("*.tiff")) if measurement_dir.exists() else []
    acq_date = None
    pols: list[str] = []
    try:
        ns = {"s1sarl1": "http://www.esa.int/safe/sentinel-1.0/sentinel-1/sar/level-1"}
        tree = ET.parse(manifest)
        pol_el = tree.getroot().findall(".//s1sarl1:transmitterReceiverPolarisation", ns)
        pols = [e.text for e in pol_el if e.text]
    except ET.ParseError:
        pass
    for f in tiffs:
        acq_date = acq_date or _parse_date_from_name(f.name)
    return DatasetRecord(
        id="sentinel1-" + (acq_date.isoformat() if acq_date else "unknown"),
        source=DataSource.SENTINEL1,
        path=str(root),
        status=DatasetStatus.RAW,
        acquisition_date=acq_date,
        extra={"polarisations": pols, "measurement_files": [f.name for f in tiffs]},
    )


def _scan_sentinel2(root: Path) -> DatasetRecord | None:
    mtd = root / "MTD_MSIL2A.xml"
    if not mtd.exists():
        return None
    acq_date = None
    tile_id = None
    for jp2 in (root / "GRANULE").rglob("*_B02_10m.jp2") if (root / "GRANULE").exists() else []:
        acq_date = acq_date or _parse_date_from_name(jp2.name)
        tile_id = jp2.name.split("_")[0]
    return DatasetRecord(
        id="sentinel2-" + (tile_id or "unknown") + "-" + (acq_date.isoformat() if acq_date else "unknown"),
        source=DataSource.SENTINEL2,
        path=str(root),
        status=DatasetStatus.RAW,
        acquisition_date=acq_date,
        extra={"tile_id": tile_id},
    )


def _scan_dem(root: Path) -> list[DatasetRecord]:
    records = []
    for xml_file in root.glob("Copernicus_DSM_*"):
        if xml_file.is_dir():
            tile_name = xml_file.name
            records.append(
                DatasetRecord(
                    id=f"dem-{tile_name}",
                    source=DataSource.DEM,
                    path=str(xml_file),
                    status=DatasetStatus.RAW,
                    extra={"tile": tile_name},
                )
            )
    return records


def _scan_worldcover(root: Path) -> DatasetRecord | None:
    tiles = sorted(root.glob("ESA_WorldCover_*_Map.tif"))
    if not tiles:
        return None
    return DatasetRecord(
        id="worldcover-2021",
        source=DataSource.WORLDCOVER,
        path=str(root),
        status=DatasetStatus.RAW,
        extra={"tile_count": len(tiles)},
    )


def _scan_osm(root: Path) -> DatasetRecord | None:
    pbf_files = sorted(root.glob("*.osm.pbf"))
    if not pbf_files:
        return None
    name = pbf_files[0].name.removesuffix(".osm.pbf")
    return DatasetRecord(
        id=f"osm-{name}",
        source=DataSource.OSM,
        path=str(pbf_files[0]),
        status=DatasetStatus.RAW,
        extra={"file": pbf_files[0].name},
    )


def _scan_sen1floods11(root: Path) -> DatasetRecord:
    s1_dir = root / "data" / "flood_events" / "HandLabeled" / "S1Hand"
    label_dir = root / "data" / "flood_events" / "HandLabeled" / "LabelHand"
    splits_dir = root / "splits" / "flood_handlabeled"

    if not s1_dir.exists():
        return DatasetRecord(
            id="sen1floods11", source=DataSource.SEN1FLOODS11, path=str(root), status=DatasetStatus.MISSING
        )

    chips = sorted(s1_dir.glob("*_S1Hand.tif"))
    events = sorted({c.name.split("_")[0] for c in chips})
    label_count = len(list(label_dir.glob("*_LabelHand.tif"))) if label_dir.exists() else 0
    splits_present = {
        split: (splits_dir / f"flood_{split}_data.csv").exists()
        for split in ("train", "valid", "test")
    }

    return DatasetRecord(
        id="sen1floods11",
        source=DataSource.SEN1FLOODS11,
        path=str(root),
        status=DatasetStatus.RAW if chips else DatasetStatus.MISSING,
        extra={
            "chip_count": len(chips),
            "label_count": label_count,
            "events": events,
            "splits_present": splits_present,
        },
    )


def scan_raw_datasets(raw_dir: Path = RAW_DIR) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []

    s1 = raw_dir / "sentinel1"
    if s1.exists():
        rec = _scan_sentinel1(s1)
        if rec:
            records.append(rec)

    s2 = raw_dir / "sentinel2"
    if s2.exists():
        rec = _scan_sentinel2(s2)
        if rec:
            records.append(rec)

    dem = raw_dir / "dem"
    if dem.exists():
        records.extend(_scan_dem(dem))

    wc = raw_dir / "worldcover"
    if wc.exists():
        rec = _scan_worldcover(wc)
        if rec:
            records.append(rec)

    osm = raw_dir / "osm"
    if osm.exists():
        rec = _scan_osm(osm)
        if rec:
            records.append(rec)

    s1f11 = raw_dir / "sen1floods11"
    records.append(_scan_sen1floods11(s1f11))

    return records


def mark_processed(records: list[DatasetRecord], processed_dir: Path = PROCESSED_DIR) -> list[DatasetRecord]:
    """Flip status to PROCESSED for records that have output under datasets/processed."""
    if not processed_dir.exists():
        return records
    processed_ids = {p.stem for p in processed_dir.rglob("*") if p.is_file()}
    for r in records:
        if any(r.id in pid for pid in processed_ids):
            r.status = DatasetStatus.PROCESSED
    return records
