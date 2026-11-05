# SphoorthiQ — SAR Vehicle-Target Segmentation, Classical + Quantum (MRF → QUBO → QAOA)

Real MSTAR SAR vehicle-target chips, segmented by formulating the problem as a Markov Random Field,
reformulating that exactly as a QUBO, and benchmarking an **exact classical solver** (s-t min-cut graph
cut) against **QAOA on Classiq/Qmod** on the identical objective. Built for THALES' **SAR Image
Segmentation** use-case (Quantum Innovation Summit 2026). Every number this app reports is from a real
run against real data. The few synthetic test cases are labelled as such, and no ground truth or metric is invented.

## Contents

- [What this is](#what-this-is)
- [THALES requirement coverage](#thales-requirement-coverage)
- [Architecture](#architecture)
- [Setup](#setup)
- [Run locally](#run-locally-dev)
- [Deploy](#deploy)
- [API](#api)
- [Verification & testing](#verification--testing)
- [Environment variables](#environment-variables)
- [Known limitations](#known-limitations-stated-plainly)

## What this is

1. **Real data** — MSTAR Public Targets chips (X-band SAR, Sandia National Lab): T72/BMP2/BTR70 vehicles
   plus a SLICY calibration target, real 128×128 (54×54 for SLICY) magnitude imagery. No per-pixel
   mask ships with the public release. Real chips are therefore scored against the **SARBake** overlay
   (Malmgren-Hansen and Nobel-Jørgensen, CC BY 4.0, doi:10.17632/jxhsg8tj7g.3), which labels every
   T72/BMP2/BTR70 chip as background, target or shadow from a 3D CAD model of the vehicle. It is a good
   reference, not hand-drawn ground truth. Without the masks the app falls back to reporting energy only.
2. **MRF formulation** — each pixel gets a unary cost from its brightness in decibels against the chip's own background
   (constants tuned on the TRAIN split, scored on TEST); each
   4-connected neighbor pair gets a contrast-sensitive pairwise smoothness cost
   (`utils/mrf/energy.py`).
3. **Two solvers, one objective**:
   - **Classical** — exact MAP labeling via s-t min-cut graph cut (Greig, Porteous & Seheult 1989), no
     qubit limit at any size.
   - **Quantum** — the same energy reformulated as a QUBO → Ising Hamiltonian, solved by QAOA on
     Classiq (`qaoa_init`/`qaoa_layer` library functions, not hand-rolled circuit logic), with a
     `scipy.optimize` outer loop scoring every sampled bitstring against the real QUBO energy
     (`utils/qml/classiq_qaoa.py`).
4. **Beyond binary** (THALES's own "beyond binary" bullets — real, shipped extensions, not future work):
   - **Edge detection** — THALES's literal definition (a pixel is an "edge" if any neighbor differs),
     derived client-side from the segmentation labels.
   - **3+ classes (shadow, Q=3)** — background/target/shadow, via a multi-label MRF, solved classically
     by alpha-expansion (approximate — Q=2's exactness doesn't generalize past two labels) and by QAOA
     over a one-hot QUBO encoding (`utils/mrf/multilabel_energy.py`).
   - **Pattern detection** — THALES names this bullet with zero definition. This app's disclosed
     interpretation: target-type recognition (BMP2 / BTR70 / T72) via correlation-filter matching
     against real per-class templates (`utils/mstar/pattern_recognition.py`), measured — not
     inflated — at 51.79% real accuracy on 1,365 real test chips (33% is chance for 3 classes).
5. **Web app** — single-page React UI (`apps/web`) over a FastAPI backend (`apps/api`), live-streaming
   real per-iteration QAOA progress, wired into the running app end to end (not a standalone notebook).
6. **Synthetic validation** — exact Accuracy/IoU/Dice on cases whose right answer is known by construction:
   THALES's own worked example (quoted directly from their brief), plus a few generated cases. The classical solver reproduces THALES's own
   stated answer exactly; QAOA matches it too at 9 and 16 qubits, and falls short (92% to 96% over four runs) at 25
   (`utils/mrf/synthetic_validation.py`, `utils/mrf/metrics.py`).

See [docs/Algorithm_Design_Dossier.md](docs/Algorithm_Design_Dossier.md) for the full write-up (problem
framing, objective, constraints, assumptions, candidate formulations considered, and more).

## THALES requirement coverage

| Requirement | Status | Notes |
|---|---|---|
| Binary MRF segmentation, classical vs. quantum | ✅ | exact classical graph cut, real QAOA via Classiq |
| Images without object | ✅ | real, verified target-free ROI preset |
| Low signal-to-noise ratio | ✅ | real, verified low-contrast ROI preset |
| Edge detection | ✅ | THALES's literal definition |
| 3+ classes segmentation | ✅ | Q=3 shadow class — approximate classically, real |
| Pattern detection | ✅ (disclosed interpretation) | undefined by THALES; built + labeled as a judgment call |
| Multi-class edge detection, single-pixel edges | ✅ | explicitly framed by THALES as optional "further work" - built anyway |

## Architecture

```
apps/api/            FastAPI backend
  main.py             serves the built React UI + wires up routers
  routers/auth.py      signup/login/reset, Auth0 optional
  routers/mstar.py      /api/mstar*, /api/mstar/shadow, /api/mstar/pattern* — the actual pipeline

apps/web/             React + TypeScript frontend (Vite)
  src/pages/MrfPage.tsx   the tool: dataset picker, ROI presets, run, and 6 result tabs (Images, Metrics,
                          Logs, Synthetic Validation, Pattern Detection, 3-Class Segmentation)
  src/pages/AboutPage.tsx  workflow diagrams, verification evidence, scaling/robustness/validation data

utils/mrf/            Q=2 + Q=3 MRF/QUBO/Ising math (energy.py, multilabel_energy.py, solvers.py),
                      plus metrics.py + synthetic_validation.py (real Accuracy/IoU/Dice, see below)
utils/mstar/          MSTAR chip reader + pattern-recognition classifier
utils/qml/            classiq_qaoa.py — the actual QAOA-over-Classiq implementation
utils/auth/           local signup/login/reset (SQLite-backed), optional Auth0
utils/jobs/, utils/registry/, utils/ai/classic/  kept working for utils/mrf/experiments.py's dormant
                      evidence-generation path (not on the live app's request path) - see that module

tests/                pytest — real MSTAR chips, exhaustive QUBO-vs-energy checks, live API tests
scripts/              real sweep scripts behind the About page's evidence tables (see Reproducibility
                      in docs/Algorithm_Design_Dossier.md)
notebooks/11_mrf_qaoa_classiq.ipynb   exploratory notebook behind the verified QUBO check on About
docs/Algorithm_Design_Dossier.md      full write-up: problem framing through next-step roadmap
```

## Setup

### 1. Python environment

```bash
python -m venv .venv          # Python 3.12
.venv/Scripts/pip install -r requirements.txt   # Windows
# .venv/bin/pip install -r requirements.txt     # macOS/Linux
```

### 2. Get the real MSTAR dataset

No automated fetch script — the MSTAR Public Targets chips are a real, manually-downloaded public
dataset. Place it at:

```
datasets/raw/mstar/MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY/
    TARGETS/TRAIN/17_DEG/{BMP2,BTR70,T72}/...
    TARGETS/TEST/15_DEG/{BMP2,BTR70,T72}/...
```

`utils/mrf/mstar_scenes.py` and `utils/mstar/pattern_recognition.py` read directly from this path —
nothing here is bundled or downloaded automatically.

### 3. Authenticate with Classiq (required for the quantum path)

The classical path works with no setup. QAOA requires a real, authenticated Classiq session:

```bash
.venv/Scripts/python.exe -c "
import keyring
from keyrings.alt.file import PlaintextKeyring
keyring.set_keyring(PlaintextKeyring())

from classiq import *
authenticate()
"
```

This prints a URL + short user code — open it, log into the Classiq account, and complete any MFA
prompt. The `PlaintextKeyring` swap is a real, necessary Windows workaround: the default Windows
Credential Manager backend can fail (`WinError 1783`) or hang silently when Classiq's SDK tries to
persist the session token from a non-interactive process (e.g. a plain `uvicorn` server, not a
notebook with a visible browser). `utils/qml/classiq_qaoa.py` applies this same workaround
automatically on every quantum request, so the app itself needs no extra configuration once this
one-time login is done — but if the account's refresh token is ever invalidated (expired, revoked,
or an MFA policy change), redo this step with `authenticate(overwrite=True)`.

### 4. Frontend dependencies

```bash
cd apps/web
npm install
```

## Run locally (dev)

Two processes, hot reload — separate terminals:

```bash
.venv/Scripts/python.exe -m uvicorn apps.api.main:app --reload --port 8000

cd apps/web
npm run dev
```

Open `http://localhost:5173`, sign up (any username, no approval step), then use the app.

## Deploy

No containers, no separate frontend host — `apps/api/main.py` serves the built React UI itself, so the
whole app is **one process on one port**.

```bash
cd apps/web
npm install
npm run build                 # writes apps/web/dist/ — main.py auto-detects and serves it

cd ..
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python.exe -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000   # no --reload in production
```

Open `http://<the machine's address>:8000` — API and UI on the same origin.

## API

| Endpoint | What it does |
|---|---|
| `GET /api/health` | liveness check |
| `POST /api/auth/signup` / `login` / `forgot-password` / `reset-password` | local account auth |
| `GET /api/auth/auth0/login` | optional Auth0 SSO (only if `AUTH0_*` env vars are set) |
| `GET /api/mstar/chips` | the curated real-chip catalog (dataset picker) |
| `POST /api/mstar` | run classical + (optionally) quantum Q=2 segmentation on one ROI |
| `POST /api/mstar/stream` | same, as a live NDJSON stream of real per-iteration QAOA progress |
| `POST /api/mstar/shadow` | Q=3 (background/target/shadow) segmentation |
| `POST /api/mstar/pattern` | classify one chip's target type |
| `GET /api/mstar/pattern/evaluation` | real accuracy over the full MSTAR test split |
| `POST /api/mstar/synthetic-validation` | real Accuracy/IoU/Dice against known ground truth (never real MSTAR) |

All `/api/mstar*` routes require a bearer token from `/api/auth/login` or `/signup`.

## Verification & testing

```bash
.venv/Scripts/python.exe -m pytest tests/ -q     # real MSTAR chips, no mocks — 53 tests
ruff check .                                     # lint (E/F/I — real bugs, not style opinions)

cd apps/web
npm run lint                                     # oxlint
npm run build                                    # tsc -b && vite build
```

Specific evidence worth knowing about (see the About page for the full story):

- Against the SARBake masks, on 120 random TEST chips (`scripts/sarbake_eval.py`), the classical two-class
  solver reaches a target IoU of 0.56 (0.04 with the earlier raw-magnitude unary term), and the
  three-class solver a mean IoU of 0.66 (0.16 before). The decibel constants were tuned on TRAIN chips only.
  On the small edge tiles the quantum solver runs on, scores are much lower (target IoU 0.18 on 4×4, 0.34 on 8×8;
  three-class accuracy below the majority-class baseline), because a few speckled pixels give little evidence.
- The QUBO reformulation is checked **exact** over all 65,536 states of a 4×4 patch
  (`utils/mrf/verification.py`, `tests/test_mrf_energy.py`) — max deviation 2.13×10⁻¹⁴ (float noise).
- The brief names a MATLAB genetic algorithm as its baseline. MATLAB is not part of this project, so `scripts/ga_baseline.py` runs a
  genetic algorithm implemented in Python (not MATLAB's `ga`) on the same energy and QAOA's evaluation budget. It reaches the optimum on
  27 to 30 of 30 seeds on every scaling tile except the 25-qubit BMP2 tile, and beats QAOA's average energy at 25 qubits.
- QAOA is live-tested at 9/16/25 qubits on two real chips (edge tiles, `scripts/scaling_sweep.py`): exact
  match at 9 and 16 qubits, a shortfall at 25 under the same optimizer budget. Random sampling with the
  same sample budget also finds the optimum at 9 qubits, so that match proves the pipeline, not an advantage.
- The Q=3 alpha-expansion reduction was caught wrong on its first implementation (a plain-XOR graph-cut
  primitive reused incorrectly) and fixed only after 500/500 brute-force trials passed — see
  `utils/mrf/multilabel_energy.py`'s module docstring for the exact bug.
- QAOA matched the exact classical optimum at every synthetic-noise level tested (`scripts/robustness_sweep.py`),
  and reproduced THALES's own worked example exactly at the classical level, 100% quantum match at 9/16
  qubits, falling short (92% to 96% over four runs) at 25 (`scripts/synthetic_validation_run.py`).

## Environment variables

All optional — sensible dev defaults otherwise.

| Variable | Purpose | Default |
|---|---|---|
| `JWT_SECRET` | signs session + reset tokens | auto-generated, persisted to `datasets/metadata/.jwt_secret` |
| `WEB_BASE_URL` | base URL for password-reset links | derived from the incoming request |
| `SMTP_HOST` / `SMTP_USER` / `SMTP_PASSWORD` | send real reset emails | unset — links written to `datasets/metadata/outbox/` instead |
| `AUTH0_DOMAIN` / `AUTH0_CLIENT_ID` / `AUTH0_CLIENT_SECRET` | enables "Log in with Auth0" | unset → that button 501s; self-service login still works |
| `CORS_ORIGINS` | extra allowed origins, comma-separated | dev-server ports only |
| `CLASSIQ_AUTH_ORGANIZATION` | Classiq org id, if the account needs one | read from `%APPDATA%\classiq\config.env` if unset |
| `SARBAKE_DIR` | folder holding the SARBake `BMP2/`, `BTR70/` and `T72/` CSV mask folders | `datasets/raw/sarbake` |

## Known limitations (stated plainly)

- **Real qubit ceiling (~25 qubits)** on the Classiq simulators available to this project — caps Q=2 at
  a 5×5 tile, Q=3 at a 2×2 tile. A full 128×128 chip needs tiling into many small ROIs; this app runs
  one ROI at a time and says so, rather than claiming otherwise.
- **No hand-drawn ground truth** for MSTAR. Real-chip scores use the SARBake masks, which come from CAD
  models, so they are a good guide and not perfect truth. Chips without a mask (SLICY) are compared on energy only.
- **Q=3 loses Q=2's exactness guarantee** — s-t graph cut is a two-terminal construction; the classical
  Q=3 solver (alpha-expansion) is a real, working, but approximate algorithm.
- **Pattern detection accuracy (51.79%)** is a deliberately simple, transparent baseline (correlation
  filter), not a claim of matching modern CNN-based MSTAR classifiers (~95%+).
