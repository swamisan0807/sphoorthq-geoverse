"""FastAPI app exposing the notebook pipeline (catalog, run history, saved
models, architecture diagram, live inference) over HTTP for the web UI.

Run: uvicorn src.api.main:app --reload --port 8000
"""

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import auth, catalog, compare, graph, inference, jobs, observability, pipeline, quantum, registry
from src.api.routers.auth import get_current_user

app = FastAPI(title="geoverse SAR flood segmentation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
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


@app.get("/")
def root():
    return {
        "service": "geoverse SAR flood segmentation API",
        "web_ui": "http://localhost:5173",
        "docs": "/docs",
    }
