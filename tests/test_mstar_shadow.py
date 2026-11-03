"""End-to-end HTTP tests for POST /api/mstar/shadow (apps/api/routers/mstar.py)
- the 3-class (background/target/shadow) extension, classical-only (no
Classiq account needed). Uses the real, locally-cached t72_132_test_1
chip, not a mock.
"""

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

    return {"Authorization": f"Bearer {issue_token('mstar-shadow-test-user')}"}


def test_shadow_classical_returns_real_three_class_segmentation():
    from apps.api.main import app

    client = TestClient(app)
    resp = client.post(
        "/api/mstar/shadow",
        json={"chip_id": _CHIP_ID, "row0": 69, "col0": 61, "size": 4, "run_quantum": False},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["chip_id"] == _CHIP_ID
    assert body["quantum"] is None
    labels = body["classical"]["labels"]
    assert len(labels) == 4 and len(labels[0]) == 4
    flat = [v for row in labels for v in row]
    assert all(v in (0, 1, 2) for v in flat)
    # this real ROI (verified earlier in the session) has a genuine target - must not be all-background
    assert 1 in flat
    assert 0.0 < body["background_mu"] < 1.0


def test_shadow_rejects_roi_too_large_for_qaoa():
    from apps.api.main import app

    client = TestClient(app)
    # 3x3 needs 3*3*3=27 qubits, over the real 25-qubit ceiling
    resp = client.post(
        "/api/mstar/shadow",
        json={"chip_id": _CHIP_ID, "row0": 69, "col0": 61, "size": 3, "run_quantum": True},
        headers=_auth_headers(),
    )
    assert resp.status_code == 400
    assert "qubits" in resp.json()["detail"]


def test_shadow_rejects_unknown_chip():
    from apps.api.main import app

    client = TestClient(app)
    resp = client.post(
        "/api/mstar/shadow",
        json={"chip_id": "not-a-real-chip", "row0": 0, "col0": 0, "size": 4, "run_quantum": False},
        headers=_auth_headers(),
    )
    assert resp.status_code == 400
