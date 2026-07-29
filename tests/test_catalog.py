from src.catalog.scanner import _scan_sen1floods11, scan_raw_datasets
from src.core.types import DataSource, DatasetStatus


def test_scan_finds_all_known_sources():
    records = scan_raw_datasets()
    sources = {r.source for r in records}
    assert sources == set(DataSource)


def test_sentinel1_polarisations_detected():
    records = scan_raw_datasets()
    s1 = next(r for r in records if r.source == DataSource.SENTINEL1)
    assert set(s1.extra["polarisations"]) == {"VV", "VH"}
    assert s1.acquisition_date is not None


def test_sen1floods11_reported_missing_when_empty(tmp_path):
    record = _scan_sen1floods11(tmp_path)
    assert record.status == DatasetStatus.MISSING


def test_sen1floods11_detects_chips_and_events(tmp_path):
    s1_dir = tmp_path / "data" / "flood_events" / "HandLabeled" / "S1Hand"
    label_dir = tmp_path / "data" / "flood_events" / "HandLabeled" / "LabelHand"
    splits_dir = tmp_path / "splits" / "flood_handlabeled"
    s1_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    splits_dir.mkdir(parents=True)

    (s1_dir / "Bolivia_103757_S1Hand.tif").write_bytes(b"")
    (s1_dir / "Ghana_24858_S1Hand.tif").write_bytes(b"")
    (label_dir / "Bolivia_103757_LabelHand.tif").write_bytes(b"")
    (splits_dir / "flood_train_data.csv").write_text("a,b\n")

    record = _scan_sen1floods11(tmp_path)
    assert record.status == DatasetStatus.RAW
    assert record.extra["chip_count"] == 2
    assert record.extra["events"] == ["Bolivia", "Ghana"]
    assert record.extra["label_count"] == 1
    assert record.extra["splits_present"] == {"train": True, "valid": False, "test": False}


def test_worldcover_tile_count():
    records = scan_raw_datasets()
    wc = next(r for r in records if r.source == DataSource.WORLDCOVER)
    assert wc.extra["tile_count"] > 0
