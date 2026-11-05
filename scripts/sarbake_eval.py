"""Score the segmentation solvers on real MSTAR chips against the SARBake masks.

Runs the classical solvers (graph cut for two classes, alpha-expansion for
three) on a random sample of MSTAR TEST chips (15 degree depression) and
compares them with the SARBake reference masks (utils/mrf/sarbake.py). The
decibel calibration in utils/mrf/mstar_scenes.py was tuned on the TRAIN split
(17 degrees), so this scores it on chips it was not tuned on. The previous
unary term (log1p magnitude, min-max normalized) is scored on the same chips
for comparison.

    python scripts/sarbake_eval.py [--per-class 40] [--crop 48] [--seed 1]

It also scores small edge tiles (2x2, 4x4, 8x8), the sizes the quantum solver can
run on. Each tile is placed where the mask is closest to half target, which is
the hardest case, so those scores are lower than the 48x48 ones.

Writes datasets/reports/sarbake_eval.json. No quantum jobs are run here.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.core.paths import DATASETS_DIR  # noqa: E402
from utils.mrf import sarbake  # noqa: E402
from utils.mrf.energy import build_patch_energy, solve_graph_cut  # noqa: E402
from utils.mrf.mstar_scenes import (  # noqa: E402
    MSTAR_DIR,
    THREE_CLASS_SHADOW_Z,
    THREE_CLASS_TARGET_Z,
    THREE_CLASS_Z_SCALE,
    Z_SPAN,
    calibrate_magnitude,
    target_probability,
    z_to_intensity,
)
from utils.mrf.multilabel_energy import build_multilabel_energy, solve_alpha_expansion  # noqa: E402
from utils.mstar.reader import read_mstar_chip  # noqa: E402

VEHICLES = ("BMP2", "BTR70", "T72")


def legacy_intensity(magnitude: np.ndarray) -> np.ndarray:
    lm = np.log1p(magnitude.astype(np.float64))
    return (lm - lm.min()) / (lm.max() - lm.min())


def sample_chips(split: str, per_class: int, seed: int) -> list[tuple[str, str]]:
    rng = np.random.default_rng(seed)
    out = []
    for vehicle in VEHICLES:
        files = sorted(glob.glob(str(MSTAR_DIR / "TARGETS" / split / "*" / vehicle / "*" / "*")))
        out += [(vehicle, files[i]) for i in rng.choice(len(files), per_class, replace=False)]
    return out


def crop_origin(mask: np.ndarray, crop: int) -> tuple[int, int]:
    ys, xs = np.nonzero(mask > 0)
    r0 = int(np.clip(int(ys.mean()) - crop // 2, 0, mask.shape[0] - crop))
    c0 = int(np.clip(int(xs.mean()) - crop // 2, 0, mask.shape[1] - crop))
    return r0, c0


def solve_binary(inten: np.ndarray, calibrated: bool) -> np.ndarray:
    prob = target_probability(inten) if calibrated else inten
    return solve_graph_cut(build_patch_energy(prob, inten)).reshape(inten.shape).astype(int)


def solve_three(inten: np.ndarray, calibrated: bool) -> np.ndarray:
    if calibrated:
        problem = build_multilabel_energy(
            inten,
            float(z_to_intensity(0.0)),
            target_mu=float(z_to_intensity(THREE_CLASS_TARGET_Z)),
            shadow_mu=float(z_to_intensity(THREE_CLASS_SHADOW_Z)),
            unary_scale=(Z_SPAN / THREE_CLASS_Z_SCALE) ** 2,
        )
    else:
        problem = build_multilabel_energy(inten, float(np.median(inten)))
    return solve_alpha_expansion(problem).reshape(inten.shape)


TILE_SIZES = (2, 4, 8)


def edge_tile(mask: np.ndarray, size: int, search: int = 16) -> tuple[int, int]:
    """Top-left corner of the size x size window, within `search` pixels of the
    mask's centroid, whose share of target pixels is closest to one half."""
    ys, xs = np.nonzero(mask > 0)
    cy, cx = int(ys.mean()), int(xs.mean())
    best = None
    for r in range(max(0, cy - search), min(mask.shape[0] - size, cy + search) + 1):
        for c in range(max(0, cx - search), min(mask.shape[1] - size, cx + search) + 1):
            share = float(np.mean(mask[r : r + size, c : c + size] == 1))
            key = (abs(share - 0.5), abs(r - cy) + abs(c - cx))
            if best is None or key < best[0]:
                best = (key, (r, c))
    return best[1]


def mean_of(rows: list[dict], key: str) -> float:
    return round(float(np.mean([r[key] for r in rows if r[key] is not None])), 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="TEST")
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--crop", type=int, default=48)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    start = time.time()
    chips = sample_chips(args.split, args.per_class, args.seed)
    results: dict = {"split": args.split, "chips": len(chips), "crop": args.crop, "per_vehicle": {}}
    rows = {"legacy_bin": [], "cal_bin": [], "legacy_tri": [], "cal_tri": []}
    pixels = {"target": [], "shadow": [], "baseline": []}
    tiles = {size: {"bin": [], "tri": []} for size in TILE_SIZES}
    for vehicle, path in chips:
        mask = np.loadtxt(sarbake.mask_path_for_file(vehicle, os.path.basename(path)), delimiter=",").astype(int)
        mag = read_mstar_chip(path).magnitude
        r0, c0 = crop_origin(mask, args.crop)
        sl = (slice(r0, r0 + args.crop), slice(c0, c0 + args.crop))
        gt = mask[sl]
        pixels["target"].append(int((gt == 1).sum()))
        pixels["shadow"].append(int((gt == 2).sum()))
        pixels["baseline"].append(float((gt == 0).mean()))
        cal_full = calibrate_magnitude(mag)
        for size in TILE_SIZES:
            tr, tc = edge_tile(mask, size)
            tile, tile_gt = cal_full[tr : tr + size, tc : tc + size], mask[tr : tr + size, tc : tc + size]
            tiles[size]["bin"].append(sarbake.score_binary(solve_binary(tile, True), tile_gt))
            three = sarbake.score_three_class(solve_three(tile, True), tile_gt)
            tiles[size]["tri"].append({**three, "majority": float(max(np.mean(tile_gt == k) for k in (0, 1, 2)))})
        for name, calibrated, inten_full in (("legacy", False, legacy_intensity(mag)), ("cal", True, calibrate_magnitude(mag))):
            inten = inten_full[sl]
            b = sarbake.score_binary(solve_binary(inten, calibrated), gt)
            t = sarbake.score_three_class(solve_three(inten, calibrated), gt)
            rows[f"{name}_bin"].append({**b, "vehicle": vehicle})
            rows[f"{name}_tri"].append({**t, "vehicle": vehicle})

    for name in ("legacy", "cal"):
        results[name] = {
            "binary": {k: mean_of(rows[f"{name}_bin"], k) for k in ("iou", "dice", "precision", "recall", "accuracy")},
            "binary_iou_above_0.5": int(sum(r["iou"] > 0.5 for r in rows[f"{name}_bin"])),
            "three_class": {k: mean_of(rows[f"{name}_tri"], k) for k in ("accuracy", "iou_background", "iou_target", "iou_shadow", "mean_iou")},
        }
        results["per_vehicle"][name] = {
            v: {"binary_iou": mean_of([r for r in rows[f"{name}_bin"] if r["vehicle"] == v], "iou"),
                "three_class_mean_iou": mean_of([r for r in rows[f"{name}_tri"] if r["vehicle"] == v], "mean_iou")}
            for v in VEHICLES
        }
    results["edge_tiles"] = {
        str(size): {
            "binary": {k: mean_of(tiles[size]["bin"], k) for k in ("accuracy", "iou", "dice")},
            "three_class": {k: mean_of(tiles[size]["tri"], k) for k in ("accuracy", "mean_iou", "majority")},
        }
        for size in TILE_SIZES
    }
    results["mask_pixels_in_crop_mean"] = {"target": round(float(np.mean(pixels["target"])), 1), "shadow": round(float(np.mean(pixels["shadow"])), 1)}
    results["all_background_accuracy_three_class"] = round(float(np.mean(pixels["baseline"])), 3)
    results["seconds"] = round(time.time() - start)
    print(json.dumps(results, indent=2))

    out = DATASETS_DIR / "reports" / "sarbake_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
