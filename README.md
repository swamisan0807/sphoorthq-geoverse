# sphoorthq-geoverse

SAR flood-segmentation platform: cloud-agnostic ingestion (S3 / Azure ADLS Gen2 / GCS / public HTTPS /
local), notebook-driven processing and feature engineering, a classical ML (Random Forest + U-Net) +
quantum ML hybrid (real IBM Quantum + AWS Braket connectivity, local-simulator by default), a
systematic robustness sweep, and per-stage observability logging.

Built for Thales' **SAR Image Analysis** track - Quantum Innovation Summit 2026, Algorithm Design
Competition.

See [docs/architecture.md](docs/architecture.md) for the full pipeline, directory layout, and the
architecture/dataflow diagram.

## Quickstart

```powershell
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\pip install -r requirements-quantum.txt   # needed for notebooks 05/06 (IBM Quantum + Braket)

.\.venv\Scripts\jupyter lab notebooks/
```

Run `notebooks/01_ingest.ipynb` through `09_patch_unet.ipynb` in order. They operate on
`datasets/raw/sen1floods11/` (446 real hand-labeled SAR flood chips, already downloaded).

- `04_classical_ml.ipynb` - Random Forest (pixel-wise).
- `05_qml_ibm_braket.ipynb` / `06_hybrid_ensemble_evaluation.ipynb` - quantum kernel SVM + hybrid vote.
  Default to `FORCE_SIMULATION = True` - local simulators only, no real hardware, no queue, no cost.
  Set `FORCE_SIMULATION = False` and supply credentials (see
  [config/platform.yaml](config/platform.yaml)) to route the same code through real IBM Quantum / AWS
  Braket hardware.
- `08_robustness_sweep.ipynb` - 5 perturbation types x 5 flood events, systematic degradation matrix.
- `09_patch_unet.ipynb` - patch-based U-Net training.

## Web UI

`src/api/` (FastAPI) serves the notebook pipeline as a real platform - self-service login, notebook jobs,
a versioned model registry, classical-vs-quantum comparison, and a knowledge graph, on top of everything
in the "Web UI" section above. `apps/web/` (React + Vite) is the UI on top of it. Nothing here is mocked:
inference loads the same `.pkl`/`.pt` files the notebooks save (via the registry's current-version
pointer), the catalog reads the same `sen1floods11` splits, the Quantum tab submits real circuits through
the same `src/qml/` code the notebooks use, and the Jobs tab runs real `jupyter nbconvert --execute`
subprocesses.

**Auth** (`src/auth/`) - self-service signup/login, any username creates a new account, no invite/approval
step. Passwords hashed with `bcrypt` directly (not passlib - see the comment in `src/auth/store.py` for
why), sessions are JWT bearer tokens. Every `/api/*` route except `/api/auth/*` and `/api/health` requires
a valid session.

**Jobs** (`src/jobs/`) - trigger any notebook (`01`-`09`) to run as a real background subprocess from the
UI, like a Databricks job run. No cell-by-cell live stream (nbconvert only rewrites the notebook once it
finishes) - status goes queued -> running -> success/failed with a captured log tail. Quantum kernel SVM
runs also log into this same job history (`kind: "quantum"`), so real hardware vs. simulation usage is
visible in one place.

**Model registry** (`src/registry/`) - every successful `04_classical_ml` / `09_patch_unet` job
auto-registers a new immutable version (real file snapshot + real metrics). A "current" pointer says
which version inference actually loads; "restore" to a prior version takes effect on the next inference
call, no retraining needed.

**Auto-retrain on new data** (`src/pipeline/auto_retrain.py`) - diffs the on-disk chip set against a
baseline manifest; if new chips appear, triggers real retraining jobs for both classical models.

**Compare** (`src/api/routers/compare.py`) - server-side Python/matplotlib bar chart comparing the
current registry-pointed RF/U-Net versions against the latest quantum kernel SVM run.

**Knowledge graph** (`src/graph/`) - built only from relationships the platform actually recorded:
dataset -> flood events (real catalog), notebook -> registered model version (real registry manifests),
event -> quantum run (real, since every quantum job logs its exact chip). No inferred/fabricated edges.

Run both (separate terminals):

```powershell
.\.venv\Scripts\uvicorn src.api.main:app --reload --port 8000

cd apps\web
npm install   # first time only
npm run dev
```

Open `http://localhost:5173`, sign up (any username/password, 8+ chars), then log in. Run
`04_classical_ml` and `09_patch_unet` from the Jobs tab first (or from Jupyter) so the Inference tab's
Random Forest / U-Net options have a model to load.

## Docker / CI

`Dockerfile.api` and `apps/web/Dockerfile` build real, runnable images (`datasets/raw/` - the actual
satellite imagery - is intentionally not baked in; mount it as a volume). `.github/workflows/ci.yml` lints
(`ruff`), runs the smoke test suite (`tests/`), and builds both images on every push/PR. There's no deploy
step: this repo has no configured container registry or deployment target, so "CI/CD" here honestly means
"lint, test, and build the images" - not a fabricated deploy to nowhere.
