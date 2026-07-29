from fastapi import APIRouter, HTTPException, Response

router = APIRouter()

# TODO: wire to rio-tiler over COGs in datasets/processed once the
# co-registration pipeline (src/processing) produces them.


@router.get("/{dataset_id}/{z}/{x}/{y}.png")
def get_tile(dataset_id: str, z: int, x: int, y: int):
    raise HTTPException(
        status_code=501,
        detail="tile serving not wired yet - depends on datasets/processed output",
    )
