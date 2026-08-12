# SphoorthiQ — SAR Flood Segmentation, Classical + Quantum ML Platform

Cloud-agnostic ingestion (S3 / Azure ADLS Gen2 / GCS / public HTTPS / local), notebook-driven processing
and feature engineering, a classical ML (Decision Tree + U-Net) + quantum ML hybrid (real IBM Quantum
connectivity, local-simulator by default), a systematic robustness sweep, per-stage observability
logging, and a full web platform (self-service auth, job engine, model registry, quantum inference) on
top.

Built for Thales' **SAR Image Analysis** track - Quantum Innovation Summit 2026, Algorithm Design
Competition.

See [docs/architecture.md](docs/architecture.md) for the full pipeline, directory layout, and the
architecture/dataflow diagram.

## Contents

- [Status](#status) - what's implemented right now, in plain terms
- [Quickstart](#quickstart) - run the notebooks
- [Web UI](#web-ui) - what the platform layer does, run it locally
- [Deploy](#deploy) - run it as a real, browser-reachable service (incl. Render)
- [Checks](#checks) - lint/test manually (no CI configured)

## Status

Quick orientation for anyone picking this up mid-stream - what the classical-vs-quantum comparison
actually measures right now, and why, since both sides of it changed recently.

- **Classical baseline (`04_classical_ml.ipynb`)** is a **Decision Tree**
  (`sklearn.tree.DecisionTreeClassifier`, `class_weight="balanced"`, tuned via `GridSearchCV` scored on
  precision) - not the Random Forest this project started with, swapped in "Improvement of classical
  model". Trains on a pixel table pooled from **250 train chips / 50 valid chips / 40 event-holdout test
  chips** (up from an earlier 20/8/44) - still a bounded subsample of the full 252-chip train split, not
  all of it; push `N_TRAIN_CHIPS`/`N_VALID_CHIPS`/`N_TEST_CHIPS` further if you want more.
- **Quantum kernel SVM (`05_qml_ibm.ipynb`, `06_hybrid_ensemble_evaluation.ipynb`,
  `/api/quantum/kernel-svm`)** trains and evaluates on pixels pooled across *many* chips from those same
  two splits - sen1floods11's train split for training pixels, this project's own event-holdout split for
  test pixels - instead of one chosen chip. That's a fix, not the original design: earlier versions of
  this code drew their whole sample from a single chip (`load_split("train")[3]`), which made any
  Compare-page number quantum produced unfair next to a classical model evaluated across dozens of chips.
  `utils/fusion/pixel_features.py`'s `sample_balanced_pixels_across_chips()` is the fix, and both the
  notebooks and the live API use it now - the same train/test splits, same seeds, real chip diversity on
  both sides. Confirmed working end-to-end, including a real run against actual IBM Quantum hardware
  (`ibm_marrakesh`), not just the local simulator.
- **Compare page** (`apps/api/routers/compare.py`, the Compare tab) only ever plots quantum kernel SVM
  runs tagged `is_benchmark=True` in job history - i.e. only runs that went through the multi-chip pooling
  above - never a stray single-chip result. Whatever it shows is always the fair comparison, not whichever
  quantum job happened to run most recently.

## Quickstart

```powershell
.\.venv\Scripts\pip install -r requirements.txt   # one file - everything, including qiskit for notebooks 05/06

.\.venv\Scripts\jupyter lab notebooks/
```

Run `notebooks/01_ingest.ipynb` through `09_patch_unet.ipynb` in order. They operate on
`datasets/raw/sen1floods11/` (446 real hand-labeled SAR flood chips - see [Data](#data) below for how
that gets there).

- `04_classical_ml.ipynb` - Decision Tree (pixel-wise).
- `05_qml_ibm.ipynb` / `06_hybrid_ensemble_evaluation.ipynb` - quantum kernel SVM + hybrid vote.
  Default to `FORCE_SIMULATION = True` - local simulator only, no real hardware, no queue, no cost.
  Set `FORCE_SIMULATION = False` and supply credentials (see
  [config/platform.yaml](config/platform.yaml)) to route the same code through real IBM Quantum hardware.
- `08_robustness_sweep.ipynb` - 5 perturbation types x 5 flood events, systematic degradation matrix.
- `09_patch_unet.ipynb` - patch-based U-Net training.

## Web UI

`apps/api/` (FastAPI) serves the notebook pipeline as a real platform - self-service login, notebook jobs,
a versioned model registry, classical-vs-quantum comparison, and a knowledge graph, on top of everything
in [Quickstart](#quickstart) above. `apps/web/` (React + Vite) is the UI on top of it. Nothing here is
mocked: inference loads the same `.pkl`/`.pt` files the notebooks save (via the registry's current-version
pointer), the catalog reads the same `sen1floods11` splits, the Quantum tab submits real circuits through
the same `utils/qml/` code the notebooks use, and the Jobs tab runs real `jupyter nbconvert --execute`
subprocesses.

**Auth** (`utils/auth/`) - self-service signup/login, any username creates a new account, no invite/approval
step. Real SQLite database (`datasets/metadata/users.db`), passwords hashed with `bcrypt` directly (not
passlib - see the comment in `utils/auth/store.py` for why), sessions are JWT bearer tokens. Every `/api/*`
route except `/api/auth/*` and `/api/health` requires a valid session. Forgot-password sends a real,
single-use, 15-minute reset link (real SMTP if configured, honest local fallback otherwise - see
[Environment variables](#environment-variables)).

**Auth0** (`utils/auth/auth0.py`) - a second, ready-to-deploy way to log in, alongside (not replacing) the
self-service flow above. "Log in with Auth0" on the login page starts a real server-side OAuth2
Authorization Code exchange (`/api/auth/auth0/login` -> Auth0's Universal Login -> `/api/auth/auth0/callback`)
and ends at this app's own session JWT, same as a password login - every other route is unaffected. Auto-
provisions a local account on first Auth0 login (`auth_provider="auth0"` in `users.db`); if that exact
username/email already has a password-based account, the callback refuses to silently take it over (409) -
closes the obvious pre-registration hijack. Requires `AUTH0_DOMAIN`/`AUTH0_CLIENT_ID`/`AUTH0_CLIENT_SECRET`
(see [config/platform.yaml](config/platform.yaml)) and, on Auth0's side, that exact callback URL added to
the application's Allowed Callback URLs - the one manual step outside this repo's control.

**Jobs** (`utils/jobs/`) - trigger any notebook (`01`-`09`) to run as a real background subprocess from the
UI, like a Databricks job run. No cell-by-cell live stream (nbconvert only rewrites the notebook once it
finishes) - status goes queued -> running -> success/failed with a captured log tail. Quantum kernel SVM
runs also log into this same job history (`kind: "quantum"`), so real hardware vs. simulation usage is
visible in one place.

**Quantum** (`apps/api/routers/quantum.py`, `utils/qml/`) - runs a real quantum-kernel SVM (Havlicek et al.
2019) fresh on a small balanced pixel sample pooled across many chips (see [Status](#status) above), not a
single chosen one. The UI picks between **Qiskit Simulation** (local
`AerSimulator`, no network call) and **Real IBM Hardware** - the latter lets you type in an IBM Quantum
API token/instance/channel for just that one run (never written to disk, never logged) instead of
requiring server-side env vars. Picking real hardware first calls a fast `/api/quantum/connect`
pre-flight check (auth + list backends only, no per-backend queue-status query - typically a few seconds)
before submitting any circuits, so the UI shows "connected" almost immediately rather than waiting on the
whole job. Every IBM Cloud call (auth, backend listing, least-busy lookup, job result) is timeout-bounded
(`utils/qml/ibm_quantum.py`: `CONNECT_TIMEOUT_S`, `LEAST_BUSY_TIMEOUT_S`, `JOB_RESULT_TIMEOUT_S`), so a
slow or rate-limited real account fails with a clear error instead of hanging the request. AWS Braket
support was removed - IBM Quantum only.

**Model registry** (`utils/registry/`) - every successful `04_classical_ml` / `09_patch_unet` job
auto-registers a new immutable version (real file snapshot + real metrics). A "current" pointer says
which version inference actually loads; "restore" to a prior version takes effect on the next inference
call, no retraining needed. Backed by S3 when `STATE_S3_BUCKET` is set (survives a stateless redeploy),
local disk otherwise - see [Persistence](#persistence).

**Auto-retrain on new data** (`utils/pipeline/auto_retrain.py`) - diffs the on-disk chip set against a
baseline manifest; if new chips appear, triggers real retraining jobs for both classical models.

**Compare** (`apps/api/routers/compare.py`) - the current registry-pointed Decision Tree/U-Net versions
against the most recent successful *multi-chip* quantum kernel SVM run *of each kind* - a simulator run
and a real-hardware run are tracked as separate series and never averaged together (see [Status](#status)
above for why "multi-chip" matters here). Client-side interactive chart (hover for
exact values, keyboard-reachable) plus a server-rendered PNG download; both use the same validated
CVD-safe 4-color palette. The page polls every 5s (with a live/paused toggle) so a job finishing anywhere
else in the app shows up here without a manual reload.

**Knowledge graph** (`utils/graph/`) - built only from relationships the platform actually recorded:
dataset -> flood events (real catalog), notebook -> registered model version (real registry manifests),
event -> quantum run (real, since every quantum job logs its exact chip). No inferred/fabricated edges.

### Run locally (dev)

Two processes, hot reload - separate terminals:

```powershell
.\.venv\Scripts\uvicorn apps.api.main:app --reload --port 8000

cd apps\web
npm install   # first time only
npm run dev
```

Open `http://localhost:5173`, sign up (any username/password, 8+ chars), then log in. Run
`04_classical_ml` and `09_patch_unet` from the Jobs tab first (or from Jupyter) so the Inference tab's
Decision Tree / U-Net options have a model to load.

## Deploy

No containers, no separate frontend host, no reverse proxy - `apps/api/main.py` serves the built React UI
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
.\.venv\Scripts\uvicorn apps.api.main:app --host 0.0.0.0 --port 8000   # no --reload in production
```

Open `http://<the machine's address>:8000` - that's the real app (not `/docs`), API and UI on the same
origin. `apps/api/main.py`'s catch-all route serves `apps/web/dist/index.html` for any path that isn't a
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

The full dataset (the real `sen1floods11` SAR imagery under `datasets/raw/`, ~1.7 GB, plus trained model
files under `datasets/processed/`, ~269 MB) lives in this project's own S3 bucket,
**`s3://sphoorthq-geoverse/datasets/`** - a straight mirror of the local `datasets/` tree, uploaded via
`S3Connector.upload_prefix()`. Both dirs are still `.gitignore`d (S3 is the real source of truth now, not
git) - **a fresh clone has neither, and doesn't need them locally until a notebook or the API actually
reads them:**

- `datasets/raw/sen1floods11/` - `notebooks/01_ingest.ipynb` pulls it from
  `s3://sphoorthq-geoverse/datasets/raw/sen1floods11/` (see [AWS credentials](#aws-credentials) below),
  falling back to the original public GCS bucket (no credentials needed) if the S3 mirror isn't reachable.
  Either way it lands in the same local `datasets/raw/sen1floods11/` folder structure - local disk here is
  a working cache for notebooks/libraries that need real files (rasterio, sklearn, torch), not the source
  of truth.
- `datasets/processed/models/` - regenerate by running `04_classical_ml` and `09_patch_unet` (from the
  Jobs tab or Jupyter), or restore an existing version from the S3-backed registry (see
  [Persistence](#persistence) below) - no retraining needed if a version is already registered.

#### AWS credentials

Every S3 path in this platform (dataset ingestion above, plus registry/job/run state below) uses the
standard boto3 credential chain - env vars, `~/.aws/credentials`, or an IAM role, nothing hardcoded. Local
dev is set up with a named profile:

```ini
# ~/.aws/credentials
[sphoorthq-geoverse]
aws_access_key_id = ...
aws_secret_access_key = ...
```

```ini
# ~/.aws/config
[profile sphoorthq-geoverse]
region = us-east-1
```

Set `AWS_PROFILE=sphoorthq-geoverse` (or make it your `[default]` profile, as this dev setup does) so
notebooks and the API pick it up with no extra config. `region_name` is also passed explicitly wherever
the code constructs an S3 client, so `AWS_DEFAULT_REGION`/profile region are a fallback, not a requirement.

### Persistence

Render's free plan has no persistent disk - the filesystem resets on every deploy and every restart. That
means `datasets/metadata/users.db` (accounts), `datasets/metadata/.jwt_secret` (sessions), and everything
under `datasets/reports/` and `datasets/processed/` would all be wiped along with it. Two ways to fix this:

- **S3-backed state** (`utils/core/cloud_state.py`) - set `STATE_S3_BUCKET=sphoorthq-geoverse` and
  `STATE_S3_PREFIX=datasets` (this dev setup exports both, see [AWS credentials](#aws-credentials) above)
  and the model registry (`utils/registry/`) and job/run history (`utils/jobs/`, `utils/observability/run_logger.py`)
  read and write S3 as the source of truth instead of local disk, so that state survives a redeploy without
  needing a paid plan at all - and, with the prefix above, lands at the exact same
  `s3://sphoorthq-geoverse/datasets/processed/models/registry/...` and
  `s3://sphoorthq-geoverse/datasets/reports/{jobs,runs}/...` paths the one-time full dataset mirror already
  populated, so existing registered model versions and job/run history are visible immediately, not just
  new ones going forward. See [config/platform.yaml](config/platform.yaml) for the full env var list. This
  doesn't cover `users.db`/`.jwt_secret` (a live SQLite file isn't safe to point at S3 - no real file
  locking) - those still need the option below, or a real database service, if accounts need to survive
  redeploys too.
- **Render persistent disk** (paid plan) mounted at `datasets/` - covers everything including the auth DB.
  `render.yaml` has that block ready, commented out.

### Environment variables

All optional - sensible dev defaults otherwise. This dev setup exports `STATE_S3_*` and `AWS_PROFILE` as
persistent user env vars (see [Persistence](#persistence) / [AWS credentials](#aws-credentials) above), so
the app runs S3-backed by default here without re-exporting anything per shell:

| Variable | Purpose | Default |
|---|---|---|
| `JWT_SECRET` | Signs session + password-reset tokens. Auto-generates and persists one to `datasets/metadata/.jwt_secret` if unset - fine for a single instance, but set this explicitly if you ever run more than one API process, or if `datasets/` isn't persisted (see above) and you don't want every restart to invalidate every session | auto-generated |
| `WEB_BASE_URL` | Base URL baked into password-reset email links | none needed - derived from the actual incoming request, so it's automatically correct for wherever you deploy this. Only set it if the UI is ever hosted on a different origin than the API |
| `SMTP_HOST` / `SMTP_USER` / `SMTP_PASSWORD` | Send real password-reset emails (see `config/platform.yaml`) | unset - reset links are written to `datasets/metadata/outbox/` and returned directly in the API response instead |
| `AUTH0_DOMAIN` / `AUTH0_CLIENT_ID` / `AUTH0_CLIENT_SECRET` | Enables the "Log in with Auth0" button (see [config/platform.yaml](config/platform.yaml)) | this dev setup: set, app `sphoorthq-geoverse`. Unset elsewhere -> `/api/auth/auth0/login` returns 501 and the button doesn't work; self-service login is unaffected either way |
| `CORS_ORIGINS` | Extra allowed origins (comma-separated), only needed if the UI is ever hosted separately from the API | dev-server ports only |
| `IBM_QUANTUM_TOKEN` / `IBM_QUANTUM_INSTANCE` / `IBM_QUANTUM_CHANNEL` | Route the Quantum tab's "Real IBM Hardware" mode through your account by default, without typing a token into the UI each run (see [config/platform.yaml](config/platform.yaml)) | unset - the Quantum tab's "Real IBM Hardware" mode still works by entering credentials per-request in the UI; without either, it falls back to "Qiskit Simulation" behavior only if you pick that mode explicitly |
| `STATE_S3_BUCKET` / `STATE_S3_PREFIX` / `STATE_S3_REGION` | Makes the model registry + job/run history S3-backed instead of local-disk-backed (see [Persistence](#persistence) above) | this dev setup: `sphoorthq-geoverse` / `datasets` / `us-east-1` - set, so S3-backed by default here. Unset elsewhere -> local-disk-backed, which a stateless redeploy (e.g. Render free tier) wipes |

## Checks

No CI workflow is configured (removed - not this repo's current setup). Run these manually before pushing:

```powershell
.\.venv\Scripts\pytest tests/          # smoke tests - API wiring, metrics math, auth, registry, U-Net shape
.\.venv\Scripts\ruff check .           # lint (E/F/I - real bugs, not style opinions, see pyproject.toml)

cd apps\web
npm run lint                           # oxlint
npm run build                          # tsc -b && vite build
```
