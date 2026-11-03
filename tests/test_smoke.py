"""Real smoke tests for CI - not exhaustive, but each assertion is a
genuine check against actual behavior, not a placeholder. Run with:
pytest tests/test_smoke.py
"""

import numpy as np


def _collect_paths(routes) -> set[str]:
    """This FastAPI version wraps `include_router`-mounted routes in an
    `_IncludedRouter` that has no `.path` of its own - the real Route
    objects (with `.path`) live on `.original_router.routes`. Recurse
    through that instead of just skipping anything without `.path`, or
    this "checks every router is wired up" test silently checks nothing."""
    paths = set()
    for route in routes:
        if hasattr(route, "path"):
            paths.add(route.path)
        elif hasattr(route, "original_router"):
            paths |= _collect_paths(route.original_router.routes)
    return paths


def test_api_app_imports_and_wires_routers():
    from apps.api.main import app

    paths = _collect_paths(app.routes)
    assert "/api/health" in paths
    assert "/api/auth/login" in paths
    assert "/api/mstar/chips" in paths
    assert "/api/mstar" in paths


def test_evaluate_metrics_are_correct_on_known_input():
    from utils.ai.objectives.registry import evaluate

    # 2 true positives, 1 false positive, 1 false negative, 4 true negatives
    pred = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    target = np.array([1, 1, 0, 1, 0, 0, 0, 0])

    metrics = evaluate("flood-segmentation", pred, target)
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 2 / 3
    assert metrics["iou"] == 2 / 4  # tp=2, fp=1, fn=1 -> 2/(2+1+1)
    assert abs(metrics["f1"] - 2 / 3) < 1e-9


def test_password_hash_roundtrip():
    from utils.auth.store import _hash_password, _verify_password

    hashed = _hash_password("correct horse battery staple")
    assert _verify_password("correct horse battery staple", hashed)
    assert not _verify_password("wrong password", hashed)


def test_jwt_token_roundtrip():
    from utils.auth.tokens import decode_token, issue_token

    token = issue_token("testuser")
    assert decode_token(token) == "testuser"
    assert decode_token("not-a-real-token") is None


def test_jobs_engine_lists_known_notebooks():
    from utils.jobs.engine import list_notebooks

    notebooks = list_notebooks()
    assert "11_mrf_qaoa_classiq" in notebooks
    assert "keyring_check" in notebooks


def test_registry_handles_unknown_model_gracefully():
    # utils.registry looks unused by the live MSTAR app at first glance,
    # but utils.ai.classic.model_io (imported transitively via
    # utils.mrf.energy.patch_energy_from_chip, kept real and working for
    # utils.mrf.experiments.py's evidence generation - see that module's
    # docstring) imports it at module level, so it must stay importable.
    from utils.registry import model_registry

    assert model_registry.list_versions("nonexistent_model_xyz") == []
    assert model_registry.get_current_version("nonexistent_model_xyz") is None


def test_unet_forward_pass_shape():
    import torch

    from utils.ai.classic.unet import UNet

    model = UNet(in_channels=2, num_classes=1, base_channels=4)
    x = torch.zeros(1, 2, 64, 64)
    out = model(x)
    assert out.shape == (1, 1, 64, 64)
