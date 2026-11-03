"""Runs utils/mrf/synthetic_validation.py's real cases through BOTH the
classical and quantum (Classiq QAOA) solvers, computing real Accuracy/IoU/
Dice against real known ground truth for each. Not simulated: every quantum
row is a real round trip to Classiq's platform.

Usage: .venv/Scripts/python.exe scripts/synthetic_validation_run.py
(Classiq session must already be authenticated - see README.md)
"""

from utils.mrf.metrics import compute_metrics
from utils.mrf.solvers import run_classical, run_quantum
from utils.mrf.synthetic_validation import build_validation_cases, problem_for_case


def main() -> None:
    rows = []
    for case in build_validation_cases():
        problem = problem_for_case(case)
        size = case.intensity.shape[0]

        c_labels, c_energy, c_dur = run_classical(problem)
        c_metrics = compute_metrics(c_labels, case.true_labels)

        q_labels, q_result = run_quantum(
            problem, size=size, run_quantum=True, reps=2, max_iterations=15, num_shots=512, backend="simulator"
        )
        q_metrics = compute_metrics(q_labels, case.true_labels) if q_labels is not None else None

        row = {
            "name": case.name,
            "size": size,
            "qubits": size * size,
            "classical": c_metrics | {"energy": round(c_energy, 4), "duration_s": round(c_dur, 4)},
            "quantum": (q_metrics | {"energy": q_result.energy, "duration_s": q_result.duration_s}) if q_metrics else None,
        }
        rows.append(row)
        print(row, flush=True)

    print("\n| case | size | classical acc/IoU/dice | quantum acc/IoU/dice | classical energy | quantum energy |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        c = r["classical"]
        q = r["quantum"]
        c_str = f"{c['accuracy']:.2f} / {c['iou']:.2f} / {c['dice']:.2f}"
        q_str = f"{q['accuracy']:.2f} / {q['iou']:.2f} / {q['dice']:.2f}" if q else "-"
        print(f"| {r['name']} | {r['size']}x{r['size']} | {c_str} | {q_str} | {c['energy']} | {q['energy'] if q else '-'} |")


if __name__ == "__main__":
    main()
