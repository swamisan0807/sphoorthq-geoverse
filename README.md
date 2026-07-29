# sphoorthq-geoverse

SAR-centric multi-modal segmentation: Sentinel-1 SAR fused with Sentinel-2 optical, Copernicus DEM,
ESA WorldCover, and OSM, with classical and quantum-hybrid segmentation models, served through a
web UI for cataloging data, running inference, and comparing model results.

See [docs/architecture.md](docs/architecture.md) for the full pipeline, directory layout, and how
to run the backend, frontend, CLI, and tests.

```powershell
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -m src.cli.main ingest
```
