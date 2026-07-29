"""Classical ML vs Quantum ML comparison - server-side (Python/matplotlib)
observability plots, not a JS charting library, per the ask."""

import io

import matplotlib

matplotlib.use("Agg")  # headless - no display server on this backend process
import matplotlib.pyplot as plt
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.jobs.engine import list_jobs
from src.registry import model_registry

router = APIRouter(prefix="/api/compare", tags=["compare"])

METRIC_ORDER = ["iou", "f1", "precision", "recall", "boundary_f1"]


def _strip_prefix(metrics: dict) -> dict:
    """registry metrics are logged as valid_iou/valid_f1/... - strip the
    'valid_' prefix so they line up with the quantum job's bare metric names."""
    return {k.removeprefix("valid_"): v for k, v in metrics.items()}


def _latest_quantum_metrics() -> dict | None:
    for job in list_jobs(limit=50):
        if job.kind == "quantum" and job.status == "success":
            return job.extra.get("metrics")
    return None


def _collect_series() -> dict[str, dict]:
    series = {}
    rf_manifest = model_registry.get_current_manifest("classical_rf")
    if rf_manifest:
        series["Random Forest (classical)"] = _strip_prefix(rf_manifest["metrics"])
    unet_manifest = model_registry.get_current_manifest("patch_unet")
    if unet_manifest:
        series["U-Net (classical)"] = _strip_prefix(unet_manifest["metrics"])
    quantum_metrics = _latest_quantum_metrics()
    if quantum_metrics:
        series["Quantum kernel SVM"] = quantum_metrics
    return series


@router.get("/data")
def comparison_data():
    return _collect_series()


@router.get("/plot.png")
def comparison_plot():
    series = _collect_series()

    fig, ax = plt.subplots(figsize=(9, 5))
    if not series:
        ax.text(0.5, 0.5, "No model runs yet - train RF/U-Net or run a quantum kernel SVM first",
                 ha="center", va="center", fontsize=11)
        ax.axis("off")
    else:
        n_models = len(series)
        bar_width = 0.8 / n_models
        x = range(len(METRIC_ORDER))
        colors = ["#4ade80", "#60a5fa", "#f472b6", "#facc15"]

        for i, (model_name, metrics) in enumerate(series.items()):
            values = [metrics.get(m, 0.0) for m in METRIC_ORDER]
            offsets = [xi + i * bar_width for xi in x]
            ax.bar(offsets, values, width=bar_width, label=model_name, color=colors[i % len(colors)])

        ax.set_xticks([xi + bar_width * (n_models - 1) / 2 for xi in x])
        ax.set_xticklabels(METRIC_ORDER)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("score")
        ax.set_title("Classical ML vs Quantum ML - flood segmentation (real data)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, facecolor="#0f1115")
    plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
