from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import catalog, experiments, inference, tiles

app = FastAPI(title="geoverse-api", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(catalog.router, prefix="/api/datasets", tags=["catalog"])
app.include_router(tiles.router, prefix="/api/tiles", tags=["tiles"])
app.include_router(inference.router, prefix="/api/inference", tags=["inference"])
app.include_router(experiments.router, prefix="/api/experiments", tags=["experiments"])


@app.get("/api/health")
def health():
    return {"status": "ok"}
