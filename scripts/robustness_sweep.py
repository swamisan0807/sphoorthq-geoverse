"""Real robustness sweep - runs classical + QAOA on the SAME real MSTAR ROI
at increasing synthetic-noise levels (utils.mrf.energy.inject_gaussian_noise),
against the live app's own POST /api/mstar, and prints the results as a
Markdown table. Not simulated: every row is a real HTTP round trip to a
running uvicorn server, including a real Classiq QAOA execution per row.

Usage: .venv/Scripts/python.exe scripts/robustness_sweep.py
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
CHIP_ID = "t72_132_test_1"
SIZE = 3  # 3x3 = 9 qubits - the tier QAOA matches classical exactly at (see About page Scaling)
ROW0, COL0 = sarbake.boundary_window(CHIP_ID, SIZE)  # a tile on the vehicle's edge, so noise can change the answer
SIGMAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]


def _post(path: str, body: dict, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{BASE}{path}", data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.load(resp)


def main() -> None:
    login = _post("/api/auth/login", {"username": "arun_authcheck", "password": "TestPass123!"})
    token = login["token"]

    rows = []
    for sigma in SIGMAS:
        result = _post(
            "/api/mstar",
            {
                "chip_id": CHIP_ID,
                "row0": ROW0,
                "col0": COL0,
                "size": SIZE,
                "noise_sigma": sigma,
                "run_quantum": True,
                "qaoa_reps": 2,
                "qaoa_max_iterations": 15,
                "qaoa_num_shots": 512,
            },
            token=token,
        )
        classical_e = result["classical"]["energy"]
        quantum = result["quantum"]
        quantum_e = quantum["energy"] if quantum else None
        duration = quantum["duration_s"] if quantum else None
        match = quantum_e is not None and abs(quantum_e - classical_e) < 1e-6
        c_iou = result["classical"]["metrics"]["iou"]
        q_iou = quantum["metrics"]["iou"] if quantum else None
        rows.append((sigma, classical_e, quantum_e, duration, match, c_iou, q_iou))
        print(f"sigma={sigma:<5} classical={classical_e:<10} quantum={quantum_e!s:<10} match={match} classicalIoU={c_iou:.2f} quantumIoU={q_iou} duration={duration}s", flush=True)

    print("\n| noise sigma | classical energy | quantum energy | QAOA matches classical | classical IoU | quantum IoU | QAOA duration |")
    print("|---|---|---|---|---|---|---|")
    for sigma, ce, qe, dur, match, ci, qi in rows:
        print(f"| {sigma} | {ce} | {qe} | {'yes' if match else 'no'} | {ci:.2f} | {qi:.2f} | {dur:.1f}s |" if dur else f"| {sigma} | {ce} | {qe} | {match} | {ci:.2f} | - | - |")
    print()
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
