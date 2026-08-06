# SphoorthiQ — SAR Flood Segmentation, Classical + Quantum ML Platform

Cloud-agnostic ingestion (S3 / Azure ADLS Gen2 / GCS / public HTTPS / local), notebook-driven processing
and feature engineering, a classical ML (Random Forest + U-Net) + quantum ML hybrid (real IBM Quantum
connectivity, local-simulator by default), a systematic robustness sweep, per-stage observability
logging, and a full web platform (self-service auth, job engine, model registry, quantum inference) on
top.

Built for Thales' **SAR Image Analysis** track - Quantum Innovation Summit 2026, Algorithm Design
Competition.

See [docs/architecture.md](docs/architecture.md) for the full pipeline, directory layout, and the
architecture/dataflow diagram.

## Contents

- [Quickstart](#quickstart) - run the notebooks
- [Web UI](#web-ui) - what the platform layer does, run it locally
- [Deploy](#deploy) - run it as a real, browser-reachable service (incl. Render)
- [Checks](#checks) - lint/test manually (no CI configured)

## Quickstart

```powershell
.\.venv\Scripts\pip install -r requirements.txt   # one file - everything, including qiskit for notebooks 05/06

.\.venv\Scripts\jupyter lab notebooks/
```

Run `notebooks/01_ingest.ipynb` through `09_patch_unet.ipynb` in order. They operate on
`datasets/raw/sen1floods11/` (446 real hand-labeled SAR flood chips - see [Data](#data) below for how
that gets there).

- `04_classical_ml.ipynb` - Random Forest (pixel-wise).
- `05_qml_ibm.ipynb` / `06_hybrid_ensemble_evaluation.ipynb` - quantum kernel SVM + hybrid vote.
  Default to `FORCE_SIMULATION = True` - local simulator only, no real hardware, no queue, no cost.
  Set `FORCE_SIMULATION = False` and supply credentials (see
  [config/platform.yaml](config/platform.yaml)) to route the same code through real IBM Quantum hardware.
- `08_robustness_sweep.ipynb` - 5 perturbation types x 5 flood events, systematic degradation matrix.
- `09_patch_unet.ipynb` - patch-based U-Net training.

## Web UI

`src/api/` (FastAPI) serves the notebook pipeline as a real platform - self-service login, notebook jobs,
a versioned model registry, classical-vs-quantum comparison, and a knowledge graph, on top of everything
in [Quickstart](#quickstart) above. `apps/web/` (React + Vite) is the UI on top of it. Nothing here is
mocked: inference loads the same `.pkl`/`.pt` files the notebooks save (via the registry's current-version
pointer), the catalog reads the same `sen1floods11` splits, the Quantum tab submits real circuits through
the same `src/qml/` code the notebooks use, and the Jobs tab runs real `jupyter nbconvert --execute`
subprocesses.

**Auth** (`src/auth/`) - self-service signup/login, any username creates a new account, no invite/approval
step. Real SQLite database (`datasets/metadata/users.db`), passwords hashed with `bcrypt` directly (not
passlib - see the comment in `src/auth/store.py` for why), sessions are JWT bearer tokens. Every `/api/*`
route except `/api/auth/*` and `/api/health` requires a valid session. Forgot-password sends a real,
single-use, 15-minute reset link (real SMTP if configured, honest local fallback otherwise - see
[Environment variables](#environment-variables)).

**Jobs** (`src/jobs/`) - trigger any notebook (`01`-`09`) to run as a real background subprocess from the
UI, like a Databricks job run. No cell-by-cell live stream (nbconvert only rewrites the notebook once it
finishes) - status goes queued -> running -> success/failed with a captured log tail. Quantum kernel SVM
runs also log into this same job history (`kind: "quantum"`), so real hardware vs. simulation usage is
visible in one place.

**Quantum** (`src/api/routers/quantum.py`, `src/qml/`) - runs a real quantum-kernel SVM (Havlicek et al.
2019) fresh on a small balanced pixel sample. The UI picks between **Qiskit Simulation** (local
`AerSimulator`, no network call) and **Real IBM Hardware** - the latter lets you type in an IBM Quantum
API token/instance/channel for just that one run (never written to disk, never logged) instead of
requiring server-side env vars. Picking real hardware first calls a fast `/api/quantum/connect`
pre-flight check (auth + list backends only, no per-backend queue-status query - typically a few seconds)
before submitting any circuits, so the UI shows "connected" almost immediately rather than waiting on the
whole job. Every IBM Cloud call (auth, backend listing, least-busy lookup, job result) is timeout-bounded
(`src/qml/ibm_quantum.py`: `CONNECT_TIMEOUT_S`, `LEAST_BUSY_TIMEOUT_S`, `JOB_RESULT_TIMEOUT_S`), so a
slow or rate-limited real account fails with a clear error instead of hanging the request. AWS Braket
support was removed - IBM Quantum only.

**Model registry** (`src/registry/`) - every successful `04_classical_ml` / `09_patch_unet` job
auto-registers a new immutable version (real file snapshot + real metrics). A "current" pointer says
which version inference actually loads; "restore" to a prior version takes effect on the next inference
call, no retraining needed.

**Auto-retrain on new data** (`src/pipeline/auto_retrain.py`) - diffs the on-disk chip set against a
baseline manifest; if new chips appear, triggers real retraining jobs for both classical models.

**Compare** (`src/api/routers/compare.py`) - the current registry-pointed RF/U-Net versions against the
most recent successful quantum kernel SVM run *of each kind* - a simulator run and a real-hardware run
are tracked as separate series and never averaged together. Client-side interactive chart (hover for
exact values, keyboard-reachable) plus a server-rendered PNG download; both use the same validated
CVD-safe 4-color palette. The page polls every 5s (with a live/paused toggle) so a job finishing anywhere
else in the app shows up here without a manual reload.

**Knowledge graph** (`src/graph/`) - built only from relationships the platform actually recorded:
dataset -> flood events (real catalog), notebook -> registered model version (real registry manifests),
event -> quantum run (real, since every quantum job logs its exact chip). No inferred/fabricated edges.

### Run locally (dev)

Two processes, hot reload - separate terminals:

```powershell
.\.venv\Scripts\uvicorn src.api.main:app --reload --port 8000

cd apps\web
npm install   # first time only
npm run dev
```

Open `http://localhost:5173`, sign up (any username/password, 8+ chars), then log in. Run
`04_classical_ml` and `09_patch_unet` from the Jobs tab first (or from Jupyter) so the Inference tab's
Random Forest / U-Net options have a model to load.

## Deploy

No containers, no separate frontend host, no reverse proxy - `src/api/main.py` serves the built React UI
itself once it exists, so the whole platform is **one process on one port**, reachable from any browser
that can reach that port. No hardcoded `localhost` anywhere in the code path that matters: the API base
URL, CORS origins, and password-reset links are all either same-origin by construction or derived from
the actual incoming request - see [Environment variables](#environment-variables).

```powershell
cd apps\web
npm install
npm run build                 # writes apps/web/dist/ - main.py auto-detects and serves it

cd ..\..
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\uvicorn src.api.main:app --host 0.0.0.0 --port 8000   # no --reload in production
```

Open `http://<the machine's address>:8000` - that's the real app (not `/docs`), API and UI on the same
origin. `src/api/main.py`'s catch-all route serves `apps/web/dist/index.html` for any path that isn't a
matched `/api/*` route or an actual built asset, so client-side routes (e.g. a hard refresh on
`/dashboard`) resolve correctly.

### Render

[render.yaml](render.yaml) is a ready-to-use [Blueprint](https://render.com/docs/blueprint-spec) - connect
this repo in the Render dashboard ("New" -> "Blueprint") and it reads that file automatically. What it
does: downloads a self-contained Node tarball (Render's Python runtime doesn't ship Node) to build
`apps/web/`, `pip install`s `requirements.txt`, then starts `uvicorn` bound to Render's `$PORT`.

**Honesty about what's untested**: this environment has no Render account or credentials, so `render.yaml`
has not been verified against a real Render build. It's the most self-contained config for this repo's
shape (Python + Node, no Docker), but if the first real build log turns up something Render-specific,
that's the file to fix.

### Data

`datasets/raw/` (1.7 GB - the real `sen1floods11` SAR imagery) and `datasets/processed/` (269 MB - trained
model files) are both `.gitignore`d - **a fresh clone or a fresh Render deploy does not include them.** The
service still boots and auth still works with neither present, but Catalog, Inference, Jobs, Quantum, and
Registry all need real data to do anything:

- `datasets/raw/sen1floods11/` - re-fetch via `notebooks/01_ingest.ipynb` (`src/ingestion/http_connector.py`
  pulls it from a public HTTPS bucket, no credentials needed) or copy an existing local copy over.
- `datasets/processed/models/` - regenerate by running `04_classical_ml` and `09_patch_unet` (from the
  Jobs tab or Jupyter) once `datasets/raw/` is populated.

### Persistence

Render's free plan has no persistent disk - the filesystem resets on every deploy and every restart. That
means `datasets/metadata/users.db` (accounts), `datasets/metadata/.jwt_secret` (sessions), and everything
under `datasets/reports/` and `datasets/processed/` would all be wiped along with it. For anything beyond a
demo, attach a Render persistent disk (paid plan) mounted at `datasets/` - `render.yaml` has that block
ready, commented out.

### Environment variables

All optional - sensible dev defaults otherwise:

| Variable | Purpose | Default |
|---|---|---|
| `JWT_SECRET` | Signs session + password-reset tokens. Auto-generates and persists one to `datasets/metadata/.jwt_secret` if unset - fine for a single instance, but set this explicitly if you ever run more than one API process, or if `datasets/` isn't persisted (see above) and you don't want every restart to invalidate every session | auto-generated |
| `WEB_BASE_URL` | Base URL baked into password-reset email links | none needed - derived from the actual incoming request, so it's automatically correct for wherever you deploy this. Only set it if the UI is ever hosted on a different origin than the API |
| `SMTP_HOST` / `SMTP_USER` / `SMTP_PASSWORD` | Send real password-reset emails (see `config/platform.yaml`) | unset - reset links are written to `datasets/metadata/outbox/` and returned directly in the API response instead |
| `CORS_ORIGINS` | Extra allowed origins (comma-separated), only needed if the UI is ever hosted separately from the API | dev-server ports only |
| `IBM_QUANTUM_TOKEN` / `IBM_QUANTUM_INSTANCE` / `IBM_QUANTUM_CHANNEL` | Route the Quantum tab's "Real IBM Hardware" mode through your account by default, without typing a token into the UI each run (see [config/platform.yaml](config/platform.yaml)) | unset - the Quantum tab's "Real IBM Hardware" mode still works by entering credentials per-request in the UI; without either, it falls back to "Qiskit Simulation" behavior only if you pick that mode explicitly |

## Checks

No CI workflow is configured (removed - not this repo's current setup). Run these manually before pushing:

```powershell
.\.venv\Scripts\pytest tests/          # smoke tests - API wiring, metrics math, auth, registry, U-Net shape
.\.venv\Scripts\ruff check .           # lint (E/F/I - real bugs, not style opinions, see pyproject.toml)

cd apps\web
npm run lint                           # oxlint
npm run build                          # tsc -b && vite build
```
