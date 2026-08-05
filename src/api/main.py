"""FastAPI app exposing the notebook pipeline (catalog, run history, saved
models, architecture diagram, live inference) over HTTP for the web UI, and
- once `apps/web` has been built - serving that built UI itself, so the
whole platform is one process on one port.

Dev (two processes, hot reload):
    uvicorn src.api.main:app --reload --port 8000
    cd apps/web && npm run dev                      # separate terminal, port 5173

Deploy (one process, the real way to open this in a browser over the
network - see README.md "Deploy"):
    cd apps/web && npm run build                    # writes apps/web/dist/
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

import os

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.api.routers import auth, catalog, compare, graph, inference, jobs, observability, pipeline, quantum, registry
from src.api.routers.auth import get_current_user
from src.core.paths import PROJECT_ROOT

app = FastAPI(title="SphoorthiQ — SAR Flood Segmentation API")

# Dev-mode origins (separate `npm run dev` on :5173/:5174) plus anything
# listed in CORS_ORIGINS (comma-separated) for a real deployment where the
# built UI is instead served from a different origin than the API. When
# main.py serves the built UI itself (the common case - see below), the
# browser never makes a cross-origin request at all, so this only matters
# for split deployments.
_default_origins = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174", "http://127.0.0.1:5174"]
_extra_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_default_origins + _extra_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# auth is public (that's how you get a token); everything else requires a
# valid session - any signed-up user can reach the whole platform once
# logged in, matching "any new user can create the login for him".
app.include_router(auth.router)
_protected = [Depends(get_current_user)]
app.include_router(catalog.router, dependencies=_protected)
app.include_router(observability.router, dependencies=_protected)
app.include_router(inference.router, dependencies=_protected)
app.include_router(quantum.router, dependencies=_protected)
app.include_router(jobs.router, dependencies=_protected)
app.include_router(registry.router, dependencies=_protected)
app.include_router(compare.router, dependencies=_protected)
app.include_router(graph.router, dependencies=_protected)
app.include_router(pipeline.router, dependencies=_protected)


@app.get("/api/health")
def health():
    return {"status": "ok"}


# Serve the built React UI (apps/web/dist, produced by `npm run build`) if
# it exists, so the whole platform is reachable at one URL - this is the
# real "open it in a browser" deployment path, no separate frontend server
# or reverse proxy needed. Registered last so it never shadows the /api/*
# routers or FastAPI's own /docs, /openapi.json above.
WEB_DIST = PROJECT_ROOT / "apps" / "web" / "dist"

if WEB_DIST.is_dir():

    @app.get("/{full_path:path}")
    def serve_web_ui(full_path: str):
        # exact static asset (JS/CSS/images under dist/) if it exists,
        # else fall back to index.html so client-side routes (e.g. a
        # hard refresh on /dashboard) still resolve - same as any SPA.
        # resolve() + is_relative_to() blocks path traversal (e.g.
        # full_path="../../../etc/passwd") from escaping WEB_DIST.
        candidate = (WEB_DIST / full_path).resolve()
        if full_path and candidate.is_file() and candidate.is_relative_to(WEB_DIST):
            return FileResponse(candidate)
        return FileResponse(WEB_DIST / "index.html")

else:

    @app.get("/")
    def root():
        return {
            "service": "SphoorthiQ — SAR Flood Segmentation API",
            "web_ui": "not built yet - run: cd apps/web && npm run build (or `npm run dev` for local hot-reload dev)",
            "docs": "/docs",
        }
