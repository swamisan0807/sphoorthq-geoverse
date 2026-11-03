"""Shared loading for the persisted classical_rf (Decision Tree) and
patch_unet model files - registry-current-version-aware, with a fallback
path for models trained before the registry existed (or a notebook run
directly, outside the Jobs engine, never registered).

Split out of apps/api/routers/inference.py so utils/mrf (which needs the
same already-trained models as its MRF energy's unary/data term - see
utils/mrf/energy.py) doesn't duplicate this loading/caching logic; both
inference.py and utils/mrf/energy.py call the functions here.
"""

import pickle
from pathlib import Path

import torch

from utils.ai.classic.unet import UNet
from utils.core.paths import PROCESSED_DIR
from utils.registry import model_registry

FALLBACK_RF_PATH = PROCESSED_DIR / "models" / "classical_rf_v1.pkl"
FALLBACK_UNET_PATH = PROCESSED_DIR / "models" / "patch_unet_v1.pt"

# (resolved path, loaded object) - reloaded whenever the registry's
# "current" pointer resolves to a different path than what's cached, so a
# restore() takes effect on the next call without a restart
_rf_cache: tuple[str, object] | None = None
_unet_cache: tuple[str, object] | None = None


def rf_path() -> Path:
    return model_registry.get_current_path("classical_rf") or FALLBACK_RF_PATH


def unet_path() -> Path:
    return model_registry.get_current_path("patch_unet") or FALLBACK_UNET_PATH


def load_rf():
    """Returns the trained Decision Tree (sklearn) - raises FileNotFoundError
    if notebook 04 has never been run."""
    global _rf_cache
    path = rf_path()
    if not path.exists():
        raise FileNotFoundError(f"Decision Tree model not found at {path} - run notebooks/04_classical_ml.ipynb first")
    if _rf_cache is None or _rf_cache[0] != str(path):
        with open(path, "rb") as f:
            _rf_cache = (str(path), pickle.load(f))
    return _rf_cache[1]


def load_unet():
    """Returns the trained U-Net (PyTorch, eval mode) - raises
    FileNotFoundError if notebook 09 has never been run."""
    global _unet_cache
    path = unet_path()
    if not path.exists():
        raise FileNotFoundError(f"U-Net model not found at {path} - run notebooks/09_patch_unet.ipynb first")
    if _unet_cache is None or _unet_cache[0] != str(path):
        model = UNet(in_channels=2, num_classes=1, base_channels=16)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        _unet_cache = (str(path), model)
    return _unet_cache[1]
