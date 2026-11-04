"""Runs classical + QAOA on two real MSTAR chips at three qubit tiers
(3x3=9, 4x4=16, 5x5=25), against the live app's own POST /api/mstar - the
real sweep behind the About page's Scaling section and
Algorithm_Design_Dossier.md's Feasibility & Scalability table. Not
simulated: every quantum row is a real round trip to Classiq's platform.

Each tile sits on the vehicle's edge (utils.mrf.sarbake.boundary_window), where
the segmentation has to decide between target and background, and both solvers
are scored against the SARBake mask as well as compared on energy.

Usage: .venv/Scripts/python.exe scripts/scaling_sweep.py [APP_BASE=http://127.0.0.1:8000]
(server must already be running on 127.0.0.1:8000, Classiq session must
already be authenticated - see README.md)
"""

import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mrf import sarbake  # noqa: E402

BASE = os.environ.get("APP_BASE", "http://127.0.0.1:8000")
CHIPS = ["t72_132_test_1", "bmp2_9563_test_1"]
SIZES = [3, 4, 5]  # 9 / 16 / 25 qubits


def _post(path: str, body: dict, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{BASE}{path}", data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.load(resp)


def _get(path: str, token: str) -> list:
    req = urllib.request.Request(f"{BASE}{path}", headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def main() -> None:
    login = _post("/api/auth/login", {"username": "arun_authcheck", "password": "TestPass123!"})
    token = login["token"]

    rows = []
    for chip_id in CHIPS:
        for size in SIZES:
            row0, col0 = sarbake.boundary_window(chip_id, size)
            result = _post(
                "/api/mstar",
                {
                    "chip_id": chip_id,
                    "row0": row0,
                    "col0": col0,
                    "size": size,
                    "run_quantum": True,
                    "qaoa_reps": 2,
                    "qaoa_max_iterations": 15,
                    "qaoa_num_shots": 512,
                },
                token=token,
            )
            classical_e = result["classical"]["energy"]
            quantum = result["quantum"]
            row = {
                "chip": chip_id,
                "size": size,
                "qubits": size * size,
                "classical_energy": classical_e,
                "quantum_energy": quantum["energy"] if quantum else None,
                "quantum_duration_s": quantum["duration_s"] if quantum else None,
                "row0": row0,
                "col0": col0,
                "classical_iou": result["classical"]["metrics"]["iou"],
                "quantum_iou": quantum["metrics"]["iou"] if quantum else None,
                "classical_accuracy": result["classical"]["metrics"]["accuracy"],
                "quantum_accuracy": quantum["metrics"]["accuracy"] if quantum else None,
            }
            rows.append(row)
            print(row, flush=True)

    print("\n| chip | size | qubits | tile | classical energy | quantum energy | classical IoU | quantum IoU | QAOA duration |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['chip']} | {r['size']}x{r['size']} | {r['qubits']} | ({r['row0']},{r['col0']}) | {r['classical_energy']} | "
              f"{r['quantum_energy']} | {r['classical_iou']:.2f} | {r['quantum_iou']:.2f} | {r['quantum_duration_s']}s |")
    print()
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
