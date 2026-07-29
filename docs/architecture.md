# geoverse architecture

SAR-centric multi-modal segmentation pipeline: Sentinel-1 SAR fused with Sentinel-2 optical,
Copernicus DEM, ESA WorldCover, and OSM, feeding both classical and quantum-hybrid
segmentation models, with a web UI for cataloging data, running inference, and comparing
model results.

## Data inventory (`datasets/raw/`)

| Source | Content | Role |
|---|---|---|
| `sentinel1/` | 1 scene, IW GRD, dual-pol VV+VH, 2026-06-28 | primary input |
| `sentinel2/` | 1 tile (T09XWK), L2A, full 10/20/60m bands | auxiliary optical + weak labels (NDVI/NDWI) |
| `dem/` | 1 Copernicus DEM GLO-30 tile | terrain correction + slope/aspect feature |
| `worldcover/` | 333 tiles, ESA WorldCover 10m 2021, covering N30-N81/E60-E117 | weak/reference labels |
| `osm/` | 1 Asia extract (.osm.pbf) | vector ground truth (water/buildings/roads) |
| `sen1floods11/` | 446 hand-labeled chips, 11 flood events, 1.7GB | **primary label source for flood segmentation** |

**Known issue (standalone scenes only):** `sentinel1/`, `sentinel2/`, `dem/`, and `worldcover/` do not
spatially overlap - their footprints were checked directly: S1 is North Africa (~34N/6E), S2 is the
Canadian Arctic (~80N/-123W), DEM is Somalia (~11S/47E), WorldCover covers Central Asia/Siberia
(N30-N81/E60-E117). `src/processing/coregistration.py` will fail or produce an empty stack if pointed at
these four together - they're unrelated sample downloads, not a coherent AOI. `config/aoi.yaml` is
intentionally empty pending a real overlapping AOI (or use each modality independently to exercise its
own processing module).

**`sen1floods11` does not have this problem.** Downloaded from the public GCS bucket
(`gs://sen1floods11`, source: [cloudtostreet/Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11)) -
scoped to the `HandLabeled` subset (~1.7GB of the ~34.3GB full bucket; `WeaklyLabeled`/`perm_water`/
`checkpoints` excluded). Each of its 446 chips is a self-contained, pre-aligned 512x512px @ 10m AOI
(S1 VV/VH + hand-labeled water mask), so it needs no coregistration at all. Verified end-to-end: the
catalog scanner correctly reports all 446 chips across 11 events with all three official splits present;
a smoke test read a real chip pair (`Ghana_103272`), ran it through `src/processing/tiling.chip()`, and
scored the dataset's own Otsu-threshold baseline against the hand labels via
`src/ai/objectives/registry.evaluate()` - IoU 0.582, F1 0.736, precision 0.602, recall 0.947, which is a
believable number (Otsu over-detects water, hence high recall / lower precision). This is the fastest
path to a real trained model - see `config/datasets.yaml` for the full field-by-field breakdown.

## Pipeline flow

```
datasets/raw/  --[src/catalog]-->  catalog (real bbox/date/status per source)
               --[src/processing]--> datasets/processed/ (co-registered, per-AOI tiles)
               --[src/fusion]-----> stacked multi-modal tensor (or late-fusion encoders)
               --[src/processing/tiling]--> datasets/exports/ (train/val/test patches)
               --[src/ai/classic | src/ai/quantum]--> trained model
               --[src/ai/robustness]--> datasets/reports/robustness_*.json
```

## src/ layout

- **core/** - shared types (`AOI`, `BBox`, `DatasetRecord`, `Tile`), path constants, YAML config loader.
- **catalog/** - `scanner.py` walks `datasets/raw/`, parses manifests (Sentinel-1/2 XML, WorldCover tile
  grid, OSM pbf), returns `DatasetRecord` objects. No GDAL dependency - fast enough to run on every API call.
- **processing/** - one module per modality (`sar.py`, `optical.py`, `dem.py`, `vector.py`, `landcover.py`),
  plus `coregistration.py` (resample everything to one CRS/grid/resolution) and `tiling.py` (chip + split).
- **fusion/** - `stacking.py` (early fusion - channel stack), `feature_engineering.py` (GLCM texture,
  polarimetric ratios - satisfies "spatial and statistical relationships" objective), `late_fusion.py`
  (per-modality encoders merged before the segmentation head, PyTorch).
- **ai/**
  - `classic/` - U-Net, Dice/Tversky/Combo losses, Random Forest baseline, train/infer loops.
  - `quantum/` - `qcnn.py` (PennyLane quantum conv layer), `quantum_kernel.py` (quantum-kernel SVM),
    `hybrid_unet.py` (classical U-Net with a quantum bottleneck block - the quantum layer only touches
    the smallest feature map, since simulators can't scale to full-resolution qubit counts).
  - `objectives/registry.py` - pluggable objective definitions (`flood-segmentation`,
    `landcover-segmentation`) with their class schema and metric set (IoU, F1, boundary-F1, kappa) -
    this is what "definition of segmentation objectives and performance criteria" resolves to in code.
  - `robustness/` - perturbation functions (speckle injection, incidence-angle shift, band dropout,
    Gaussian noise) plus a harness that runs a model against clean + perturbed inputs and reports
    % degradation per condition - answers "robustness under varying data conditions".
- **storage/** - backend-agnostic I/O (`local`, `s3`, `azure`, `gcs`) behind one `StorageBackend` interface.
- **cli/** - `python -m src.cli.main {ingest,preprocess,train,evaluate,predict}`.
- **api/** - FastAPI service backing the web UI (`apps/web/`): catalog, tiles, inference, experiments routers.

## Web UI (`apps/web/`)

React + Vite + TypeScript, MapLibre for the map view, recharts for the experiment dashboard. Talks to
`src/api` over `/api/*` (Vite dev proxy to `localhost:8000`). Four pages: Map, Dataset Catalog, Run
Inference, Experiments - see the running instructions in the repo root README.

## Running

```powershell
# backend
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -m uvicorn src.api.main:app --reload --port 8000

# frontend
cd apps\web && npm install && npm run dev

# CLI
.\.venv\Scripts\python -m src.cli.main ingest

# tests
.\.venv\Scripts\python -m pytest tests/ -q

# quantum stack (only needed for src/ai/quantum)
.\.venv\Scripts\pip install -r requirements-quantum.txt
```

## What's real vs. stubbed today

Real and tested against actual data: the catalog scanner (all 6 sources, including the populated
`sen1floods11`), the objectives/metrics registry, `tiling.chip()`, and the config loader - all verified
against real `sen1floods11` GeoTIFFs, not just synthetic arrays. `processing/sar.py` /`optical.py`/
`dem.py`/`landcover.py`/`vector.py` and `coregistration.py` are implemented but not yet exercised
against real rasters - not needed for `sen1floods11` (already co-registered), still blocked for the four
standalone scenes by the AOI-overlap issue above. `ai/classic/*` and `ai/quantum/*` (U-Net, hybrid
quantum model, training loops) are implemented but not yet trained against real data - `sen1floods11` is
now unblocked for this, next step is wiring `PatchDataset` to it directly. The API's `catalog` router is
wired to the real scanner; `tiles`, `inference`, and `experiments` routers still return stub/placeholder
data pending a trained model.
