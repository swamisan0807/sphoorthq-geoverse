"""Multi-size classical-vs-QAOA experiment sweep: the evidence trail behind
"how far does this scale today" - classical graph cut is exact at any size
(utils.mrf.energy.solve_graph_cut has no qubit budget), QAOA is capped by
today's simulator qubit ceilings (see utils.qml.classiq_qaoa.MAX_QUBITS's
docstring for why the practical progression here is 3x3/4x4/5x5, not a
naive 4x4/6x6/8x8 pixel-grid doubling - no general-purpose simulator on
this project's Classiq account exceeded ~28-35 qubits at time of writing,
well under 6x6=36).

Each row runs the same comparison notebooks/11's Parts 1-3 run by hand for
one patch, just looped over multiple (chip, ROI, size) entries and written
to one CSV instead of printed interactively - see run_sweep()/to_csv().
"""

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from utils.ai.objectives.registry import evaluate
from utils.mrf.energy import build_qubo, energy, patch_energy_from_chip, solve_graph_cut
from utils.qml import classiq_qaoa


@dataclass
class RoiSpec:
    chip_id: str
    row0: int
    col0: int
    size: int
    model: str = "rf"


@dataclass
class ExperimentRow:
    chip_id: str
    row0: int
    col0: int
    size: int
    n_qubits: int
    model: str
    n_valid_pixels: int
    classical_energy: float
    classical_iou: float | None
    classical_duration_s: float
    ran_quantum: bool
    quantum_energy: float | None
    quantum_iou: float | None
    quantum_duration_s: float | None
    circuit_width: int | None
    circuit_depth: int | None
    qaoa_reps: int | None
    qaoa_num_shots: int | None
    quantum_error: str | None


def _iou_or_none(labels, patch) -> float | None:
    if not patch.valid_mask.any():
        return None
    return evaluate("flood-segmentation", labels[patch.valid_mask], patch.ground_truth[patch.valid_mask])["iou"]


def run_experiment(spec: RoiSpec, run_quantum: bool, qaoa_kwargs: dict | None = None) -> ExperimentRow:
    """Runs the classical graph-cut baseline (always - cheap, exact, no
    qubit limit) and, when run_quantum, QAOA via Classiq on the identical
    QUBO - see utils/mrf/energy.py's module docstring for why these two are
    directly comparable. A QAOA failure (over MAX_QUBITS, not
    authenticated, synthesis error, ...) is recorded in quantum_error
    rather than raising, so one bad row doesn't abort the whole sweep."""
    qaoa_kwargs = qaoa_kwargs or {}
    patch = patch_energy_from_chip(spec.chip_id, spec.model, spec.row0, spec.col0, spec.size)
    n_valid = int(patch.valid_mask.sum())

    t0 = time.time()
    classical_labels = solve_graph_cut(patch.problem)
    classical_duration = time.time() - t0

    row = dict(
        chip_id=spec.chip_id,
        row0=spec.row0,
        col0=spec.col0,
        size=spec.size,
        n_qubits=patch.problem.n,
        model=spec.model,
        n_valid_pixels=n_valid,
        classical_energy=round(energy(patch.problem, classical_labels), 4),
        classical_iou=_iou_or_none(classical_labels, patch),
        classical_duration_s=round(classical_duration, 4),
        ran_quantum=False,
        quantum_energy=None,
        quantum_iou=None,
        quantum_duration_s=None,
        circuit_width=None,
        circuit_depth=None,
        qaoa_reps=None,
        qaoa_num_shots=None,
        quantum_error=None,
    )
    if row["classical_iou"] is not None:
        row["classical_iou"] = round(row["classical_iou"], 4)

    if run_quantum:
        if patch.problem.n > classiq_qaoa.MAX_QUBITS:
            row["quantum_error"] = f"{patch.problem.n} qubits exceeds MAX_QUBITS={classiq_qaoa.MAX_QUBITS}"
        else:
            qubo, const = build_qubo(patch.problem)
            t0 = time.time()
            try:
                qaoa = classiq_qaoa.run_qaoa(qubo, const, patch.problem.n, **qaoa_kwargs)
            except Exception as e:  # noqa: BLE001 - record and continue the sweep, see docstring
                row["quantum_error"] = str(e)
            else:
                q_duration = time.time() - t0
                q_labels = qaoa.labels.reshape(spec.size, spec.size)
                q_iou = _iou_or_none(q_labels, patch)
                row.update(
                    ran_quantum=True,
                    quantum_energy=round(qaoa.best_energy, 4),
                    quantum_iou=round(q_iou, 4) if q_iou is not None else None,
                    quantum_duration_s=round(q_duration, 4),
                    circuit_width=qaoa.circuit_width,
                    circuit_depth=qaoa.circuit_depth,
                    qaoa_reps=qaoa.reps,
                    qaoa_num_shots=qaoa.num_shots,
                )

    return ExperimentRow(**row)


def run_sweep(specs: list[RoiSpec], run_quantum: bool, qaoa_kwargs: dict | None = None) -> list[ExperimentRow]:
    return [run_experiment(spec, run_quantum=run_quantum, qaoa_kwargs=qaoa_kwargs) for spec in specs]


def to_csv(rows: list[ExperimentRow], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


# Real sen1floods11 ROIs with at least one water pixel and no nodata,
# spanning two water-rich chips (India_1050276, India_1068117 - picked
# because most locally available chips are >99% background, which makes
# for a trivial/uninteresting demo) at each of the three qubit-budget tiers
# below MAX_QUBITS. Not cherry-picked for a flattering result - these are
# simply the first two mixed-label windows found per size per chip by a
# left-to-right, top-to-bottom scan (see the search used to find them in
# notebooks/11's experiment-sweep section).
DEFAULT_SWEEP_SPECS: list[RoiSpec] = [
    RoiSpec(chip_id="India_1050276", row0=0, col0=51, size=3),
    RoiSpec(chip_id="India_1068117", row0=0, col0=84, size=3),
    RoiSpec(chip_id="India_1050276", row0=0, col0=48, size=4),
    RoiSpec(chip_id="India_1068117", row0=0, col0=84, size=4),
    RoiSpec(chip_id="India_1050276", row0=0, col0=50, size=5),
    RoiSpec(chip_id="India_1068117", row0=0, col0=115, size=5),
]
