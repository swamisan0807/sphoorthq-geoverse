"""Knowledge graph built ONLY from relationships the platform actually
recorded - dataset -> flood events (real catalog), notebook -> registered
model version (real registry manifests), event -> quantum run (real, since
every quantum job records the exact chip_id it used, and chip_id encodes
its source event). No inferred or fabricated edges: if we don't track a
relationship precisely (e.g. exactly which chips a given RF/U-Net training
run sampled), it is not drawn as an edge here.
"""

import networkx as nx

from src.ai.classic.sen1floods11_dataset import chip_id_from_s1_filename, load_split
from src.jobs.engine import list_jobs
from src.registry import model_registry

DATASET_NODE = "dataset:sen1floods11"


def _event_chip_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ("train", "valid", "test"):
        try:
            pairs = load_split(split)
        except FileNotFoundError:
            continue
        for s1_filename, _ in pairs:
            chip_id = chip_id_from_s1_filename(s1_filename)
            event = chip_id.split("_")[0]
            counts[event] = counts.get(event, 0) + 1
    return counts


def build_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node(DATASET_NODE, kind="dataset", label="sen1floods11")

    for event, n_chips in _event_chip_counts().items():
        node_id = f"event:{event}"
        g.add_node(node_id, kind="event", label=event, n_chips=n_chips)
        g.add_edge(DATASET_NODE, node_id, relation="has_event")

    for model_name in ("classical_rf", "patch_unet"):
        current_version = model_registry.get_current_version(model_name)
        for manifest in model_registry.list_versions(model_name):
            notebook = manifest.get("notebook", "unknown")
            notebook_node = f"notebook:{notebook}"
            if notebook_node not in g:
                g.add_node(notebook_node, kind="notebook", label=notebook)

            version_node = f"model:{model_name}:v{manifest['version']}"
            metrics = manifest.get("metrics", {})
            iou = metrics.get("valid_iou", metrics.get("iou"))
            g.add_node(
                version_node,
                kind="model_version",
                label=f"{model_name} v{manifest['version']}",
                is_current=(manifest["version"] == current_version),
                iou=iou,
            )
            g.add_edge(notebook_node, version_node, relation="produced")

    for job in list_jobs(limit=200):
        if job.kind != "quantum" or job.status != "success":
            continue
        chip_id = job.extra.get("chip_id")
        if not chip_id:
            continue
        event_node = f"event:{chip_id.split('_')[0]}"
        if event_node not in g:
            continue
        run_node = f"quantum_run:{job.job_id}"
        g.add_node(
            run_node,
            kind="quantum_run",
            label=f"quantum SVM: {chip_id}",
            backend=job.extra.get("backend"),
            is_real_hardware=job.extra.get("is_real_hardware"),
            iou=job.extra.get("metrics", {}).get("iou"),
        )
        g.add_edge(event_node, run_node, relation="used_in")

    return g


def _safe(node_id: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in node_id)


def _fmt_iou(value) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "?"


def to_mermaid(g: nx.DiGraph) -> str:
    lines = ["graph LR"]
    for node_id, data in g.nodes(data=True):
        kind = data.get("kind", "")
        label = data.get("label", node_id)
        node = _safe(node_id)
        if kind == "dataset":
            lines.append(f'    {node}[("{label}")]')
        elif kind == "event":
            lines.append(f'    {node}(["{label}\\n{data.get("n_chips", "?")} chips"])')
        elif kind == "notebook":
            lines.append(f'    {node}[/"{label}"/]')
        elif kind == "model_version":
            marker = " (current)" if data.get("is_current") else ""
            lines.append(f'    {node}["{label}{marker}\\niou={_fmt_iou(data.get("iou"))}"]')
        elif kind == "quantum_run":
            hw = "real HW" if data.get("is_real_hardware") else "simulated"
            lines.append(f'    {node}{{"{label}\\n{hw}, iou={_fmt_iou(data.get("iou"))}"}}')
        else:
            lines.append(f'    {node}["{label}"]')

    for u, v, data in g.edges(data=True):
        lines.append(f"    {_safe(u)} -->|{data.get('relation', '')}| {_safe(v)}")

    return "\n".join(lines)
