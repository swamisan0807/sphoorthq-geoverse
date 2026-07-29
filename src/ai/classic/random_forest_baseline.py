"""Pixel-wise Random Forest baseline - cheap sanity check before spending compute
on the U-Net/quantum models. Trains on flattened per-pixel feature vectors."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def cube_to_pixel_features(cube: np.ndarray) -> np.ndarray:
    """cube: (bands, H, W) -> (H*W, bands)."""
    bands, h, w = cube.shape
    return cube.reshape(bands, h * w).T


def train_rf(cube: np.ndarray, label: np.ndarray, n_estimators: int = 200, **kwargs) -> RandomForestClassifier:
    x = cube_to_pixel_features(cube)
    y = label.reshape(-1)
    valid = ~np.isnan(x).any(axis=1)

    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, **kwargs)
    clf.fit(x[valid], y[valid])
    return clf


def predict_rf(clf: RandomForestClassifier, cube: np.ndarray) -> np.ndarray:
    bands, h, w = cube.shape
    x = cube_to_pixel_features(cube)
    valid = ~np.isnan(x).any(axis=1)

    preds = np.zeros(h * w, dtype=np.uint8)
    preds[valid] = clf.predict(x[valid])
    return preds.reshape(h, w)
