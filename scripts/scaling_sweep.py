"""Runs classical + QAOA on two real MSTAR chips at three qubit tiers
(3x3=9, 4x4=16, 5x5=25), against the live app's own POST /api/mstar - the
real sweep behind the About page's Scaling section and
Algorithm_Design_Dossier.md's Feasibility & Scalability table. Not
simulated: every quantum row is a real round trip to Classiq's platform.

Usage: .venv/Scripts/python.exe scripts/scaling_sweep.py
(server must already be running on 127.0.0.1:8000, Classiq session must
already be authenticated - see README.md)
"""

import json
import urllib.request

BASE = "http://127.0.0.1:8000"
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
    chips_by_id = {c["chip_id"]: c for c in _get("/api/mstar/chips", token)}

    rows = []
    for chip_id in CHIPS:
        info = chips_by_id[chip_id]
        for size in SIZES:
            half = size // 2
            row0 = min(max(0, info["bright_row"] - half), info["height"] - size)
            col0 = min(max(0, info["bright_col"] - half), info["width"] - size)
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
            }
            rows.append(row)
            print(row, flush=True)

    print("\n| chip | size | qubits | classical energy | quantum energy | QAOA duration |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['chip']} | {r['size']}x{r['size']} | {r['qubits']} | {r['classical_energy']} | "
              f"{r['quantum_energy']} | {r['quantum_duration_s']}s |")


if __name__ == "__main__":
    main()
