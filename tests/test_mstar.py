"""End-to-end HTTP tests for GET /api/mstar/chips, POST /api/mstar, and
POST /api/mstar/stream (apps/api/routers/mstar.py) - the classical-only
path, which needs no Classiq account. Uses the real, locally-cached MSTAR
target chips (utils/mrf/mstar_scenes.py), not a mock.
"""

import json

import pytest
from fastapi.testclient import TestClient

from utils.mrf.mstar_scenes import CATALOG, MSTAR_DIR

_CHIP_ID = "t72_132_test_1"
_CHIP_FILE = MSTAR_DIR / CATALOG[_CHIP_ID]

pytestmark = pytest.mark.skipif(
    not _CHIP_FILE.exists(),
    reason=f"real MSTAR chip not present locally at {_CHIP_FILE} - this test exercises the real pipeline, not a mock",
)


def _auth_headers() -> dict[str, str]:
    from utils.auth.tokens import issue_token

    return {"Authorization": f"Bearer {issue_token('mstar-test-user')}"}


def test_mstar_chips_lists_the_real_catalog():
    from apps.api.main import app

    client = TestClient(app)
    resp = client.get("/api/mstar/chips", headers=_auth_headers())
    assert resp.status_code == 200, resp.text
    chips = resp.json()
    assert any(c["chip_id"] == _CHIP_ID for c in chips)
    chip = next(c for c in chips if c["chip_id"] == _CHIP_ID)
    assert chip["target_type"] == "t72_tank"
    assert chip["height"] == 128
    assert chip["width"] == 128
    # real, verified ROI presets - all in-bounds for this chip
    for key in ("bright_row", "bright_col", "no_object_row", "no_object_col", "low_snr_row", "low_snr_col"):
        assert 0 <= chip[key] < 128, f"{key}={chip[key]} out of bounds"


def test_mstar_no_object_preset_is_genuinely_target_free():
    """THALES's validation set explicitly includes "images without object
    (all pixels labelled 0)" - the no_object preset must produce that for
    real, not just exist as a field."""
    from apps.api.main import app

    client = TestClient(app)
    chips_resp = client.get("/api/mstar/chips", headers=_auth_headers())
    chip = next(c for c in chips_resp.json() if c["chip_id"] == _CHIP_ID)

    resp = client.post(
        "/api/mstar",
        json={
            "chip_id": _CHIP_ID,
            "row0": chip["no_object_row"],
            "col0": chip["no_object_col"],
            "size": 5,
            "run_quantum": False,
        },
        headers=_auth_headers(),
    )
    assert resp.status_code == 200, resp.text
    labels = resp.json()["classical"]["labels"]
    flat = [v for row in labels for v in row]
    assert all(v == 0 for v in flat), "no-object preset ROI produced a non-background label - not genuinely target-free"


def test_mstar_classical_only_returns_real_segmentation():
    from apps.api.main import app

    client = TestClient(app)
    resp = client.post(
        "/api/mstar",
        json={"chip_id": _CHIP_ID, "row0": 69, "col0": 61, "size": 4, "run_quantum": False},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["chip_id"] == _CHIP_ID
    assert body["target_type"] == "t72_tank"
    assert body["quantum"] is None
    assert len(body["classical"]["labels"]) == 4
    assert len(body["classical"]["labels"][0]) == 4
    assert all(v in (0, 1) for row in body["classical"]["labels"] for v in row)
    # a real bright/dark ROI must not be trivially all-one-label
    flat = [v for row in body["classical"]["labels"] for v in row]
    assert 0 in flat and 1 in flat
    # no per-pixel ground truth exists for MSTAR's public release - metrics must never be fabricated
    assert body["classical"]["metrics"] is None
    assert body["note"] is not None


def test_mstar_rejects_out_of_bounds_roi():
    from apps.api.main import app

    client = TestClient(app)
    resp = client.post(
        "/api/mstar",
        json={"chip_id": _CHIP_ID, "row0": 999, "col0": 999, "size": 4, "run_quantum": False},
        headers=_auth_headers(),
    )
    assert resp.status_code == 400


def test_mstar_rejects_unknown_chip():
    from apps.api.main import app

    client = TestClient(app)
    resp = client.post(
        "/api/mstar",
        json={"chip_id": "not-a-real-chip", "row0": 0, "col0": 0, "size": 4, "run_quantum": False},
        headers=_auth_headers(),
    )
    assert resp.status_code == 400


def test_mstar_stream_classical_only_emits_real_ndjson_events():
    from apps.api.main import app

    client = TestClient(app)
    with client.stream(
        "POST",
        "/api/mstar/stream",
        json={"chip_id": _CHIP_ID, "row0": 69, "col0": 61, "size": 4, "run_quantum": False},
        headers=_auth_headers(),
    ) as resp:
        assert resp.status_code == 200, resp.text
        events = [json.loads(line) for line in resp.iter_lines() if line]

    # classical-only: exactly one classical_done event, then one done event
    # carrying the same full response POST /api/mstar itself returns - no
    # "progress" events since there's no QAOA to report iterations for.
    types = [e["type"] for e in events]
    assert types == ["classical_done", "done"]
    assert events[1]["result"]["quantum"] is None
    assert events[1]["result"]["classical"]["metrics"] is None
    assert events[1]["result"]["chip_id"] == _CHIP_ID


def test_mstar_rejects_roi_too_large_for_qaoa():
    from apps.api.main import app

    client = TestClient(app)
    resp = client.post(
        "/api/mstar",
        json={"chip_id": _CHIP_ID, "row0": 0, "col0": 0, "size": 6, "run_quantum": True},
        headers=_auth_headers(),
    )
    assert resp.status_code == 400
    assert "qubits" in resp.json()["detail"]
