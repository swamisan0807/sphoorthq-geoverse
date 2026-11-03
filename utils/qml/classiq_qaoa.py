"""QAOA over Classiq/Qmod - the quantum alternative to utils.mrf.energy's
exact classical graph-cut solver, run on the SAME MRF energy (see that
module's docstring) so the two are comparable numbers.

Unlike a Qiskit/AerSimulator-based path, there is no local-simulator fallback here:
Classiq synthesis and execution are both real round trips to the Classiq
platform (classiq.authenticate() must have already been run in this Python
process - see README_Setup.md for the Windows keyring workaround), even
when the chosen backend is Classiq's own simulator. A missing/expired
Classiq session surfaces as whatever error classiq.synthesize()/sample()
itself raises - not a silent fallback to a different, non-quantum result.

Circuit construction uses Classiq's own qaoa_init/qaoa_layer library
functions (classiq.qaoa_layer = classiq.qaoa_cost_layer + qaoa_mixer_layer)
rather than hand-rolled Pauli rotations, applied to the Ising Hamiltonian
utils.mrf.energy.qubo_to_ising derives from the same QUBO the classical
solver minimizes exactly.

The classical outer loop (gamma/beta) is a plain scipy.optimize.minimize
over ExecutionSession.sample, scoring every sampled bitstring with the true
MRF QUBO cost (utils.mrf.energy.qubo_energy) rather than a generic
Hamiltonian expectation value doing the same thing more opaquely - so both
the optimizer's objective and the reported result are traceable back to the
same energy() function the classical graph-cut baseline is scored with.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from utils.mrf.energy import qubo_energy, qubo_to_ising

MAX_QUBITS = 25  # one qubit per pixel (a 5x5 ROI) - not an arbitrary demo
# cutoff: Classiq's own general-purpose simulators topped out at 28-35
# qubits on this project's account (dgx_simulator 35, nvidia_simulator 29,
# simulator/simulator_density_matrix/simulator_matrix_product_state 28 -
# see `classiq.execution.get_backend_details`), so a naive 6x6 (36 qubits)
# ROI already exceeds every one of them. 25 leaves headroom under that
# ceiling while keeping the classical optimizer loop's per-iteration
# circuit sampling within a demo's interactive budget - matches the spirit
# of apps/api/routers/quantum.py's MAX_N_TRAIN/MAX_N_TEST caps on the
# quantum-kernel-SVM path.

DEFAULT_REPS = 2  # QAOA layers (p) - circuit depth grows with this
# Measured live against Classiq's platform: ~4-5s per ExecutionSession.sample()
# round trip (job submit + queue + execute), so 40 iterations is ~3-4 minutes
# of real wall-clock wait - fine for a background/notebook run, poor for an
# interactive API click. 15 keeps a live request under ~1-1.5 minutes while
# still finding the exact classical optimum on the 9/16-qubit tiers tested
# (see datasets/reports/11_experiments.csv) - notebooks/scripts that want the
# more thorough search can still pass max_iterations=40 explicitly.
DEFAULT_MAX_ITERATIONS = 15
DEFAULT_NUM_SHOTS = 512
FINAL_NUM_SHOTS = 2048  # wider sample at the optimized params, for firming up the best observed bitstring


@dataclass
class QaoaResult:
    labels: np.ndarray  # (n,) best bitstring observed, as a 0/1 int array
    best_energy: float  # utils.mrf.energy energy of `labels` - comparable to solve_graph_cut()'s result
    optimized_params: list[float]  # final [gamma_0, beta_0, gamma_1, beta_1, ...]
    optimizer_cost_history: list[float] = field(repr=False)
    n_qubits: int = 0
    reps: int = DEFAULT_REPS
    backend: str = "simulator"
    circuit_width: int = 0
    circuit_depth: int | None = None  # None when Classiq reports a symbolic (non-integer) depth
    num_shots: int = DEFAULT_NUM_SHOTS


def _configure_classiq_environment() -> None:
    """Makes this process' Classiq session actually work on Windows,
    regardless of whether the caller (a notebook cell, this API server,
    anything else importing this module) remembered to do it first:

    1. Windows CredWrite workaround - without forcing PlaintextKeyring,
       classiq's own keyring lookups can hang indefinitely (no exception,
       no timeout) on a plain `uvicorn apps.api.main:app` process, since
       there's no interactive prompt or browser for Windows Credential
       Manager to fail loudly against - see README_Setup.md.
    2. CLASSIQ_AUTH_ORGANIZATION - the installed Classiq SDK's own
       config-file loader only reads config.ini/classiq.conf under
       %APPDATA%\\classiq\\, not config.env (confirmed against
       classiq/_internals/config.py), so a config.env placed there per
       README_Setup.md is otherwise silently never applied. Parsed by hand
       here instead, only when the env var isn't already set explicitly.

    Idempotent and cheap - safe to call at the top of every run_qaoa().
    """
    import os

    if os.name == "nt":
        import keyring
        from keyrings.alt.file import PlaintextKeyring

        if not isinstance(keyring.get_keyring(), PlaintextKeyring):
            keyring.set_keyring(PlaintextKeyring())

    if "CLASSIQ_AUTH_ORGANIZATION" not in os.environ:
        from pathlib import Path

        config_env = Path(os.environ.get("APPDATA", "")) / "classiq" / "config.env"
        if config_env.exists():
            for line in config_env.read_text(encoding="utf-8").splitlines():
                key, _, value = line.strip().partition("=")
                if key.strip() == "CLASSIQ_AUTH_ORGANIZATION" and value:
                    os.environ["CLASSIQ_AUTH_ORGANIZATION"] = value.strip().strip('"')
                    break


def _build_hamiltonian(h: dict[int, float], J: dict[tuple[int, int], float], n: int):
    """Ising H(z) = sum h[i]*Z_i + sum J[i,j]*Z_i*Z_j as Classiq's dense
    PauliTerm list (one term per h/J entry: Pauli.Z at the relevant
    qubit(s), Pauli.I elsewhere) - the form classiq.qaoa_layer's
    `hamiltonian` argument expects."""
    from classiq import Pauli, PauliTerm

    terms = []
    for i, coeff in h.items():
        paulis = [Pauli.I] * n
        paulis[i] = Pauli.Z
        terms.append(PauliTerm(pauli=paulis, coefficient=coeff))
    for (i, j), coeff in J.items():
        paulis = [Pauli.I] * n
        paulis[i] = Pauli.Z
        paulis[j] = Pauli.Z
        terms.append(PauliTerm(pauli=paulis, coefficient=coeff))
    if not terms:
        # Keeps the Hamiltonian non-empty for a degenerate (no unary
        # preference, no edges) patch - never expected from a real image,
        # but build_qubo() could in principle hand back an empty QUBO for a
        # constant-cost patch.
        terms.append(PauliTerm(pauli=[Pauli.I] * n, coefficient=0.0))
    return terms


def _synthesize_qaoa_circuit(n_qubits: int, hamiltonian, reps: int):
    from classiq import CArray, CReal, Output, QArray, QBit, allocate, create_model, qaoa_init, qaoa_layer, qfunc, repeat, synthesize

    @qfunc
    def main(params: CArray[CReal, 2 * reps], target: Output[QArray[QBit]]) -> None:
        allocate(n_qubits, target)
        qaoa_init(target)
        repeat(
            reps,
            lambda i: qaoa_layer(params[2 * i], params[2 * i + 1], hamiltonian, target),
        )

    return synthesize(create_model(main))


def _score_samples(df, qubo: dict[tuple[int, int], float], const: float, n_qubits: int) -> tuple[float, np.ndarray, float]:
    """Scores every distinct bitstring in an ExecutionSession.sample()
    DataFrame against the real QUBO energy. Returns (shot-weighted mean
    energy, best labels seen, best energy seen)."""
    total_weighted, total_counts = 0.0, 0
    best_labels, best_energy = None, float("inf")
    for row in df.itertuples(index=False):
        bits = getattr(row, "target")
        labels = np.array([int(b) for b in bits], dtype=np.int64)
        if labels.shape[0] != n_qubits:
            raise RuntimeError(
                f"expected {n_qubits} measured bits for output 'target', got {labels.shape[0]} - "
                "Classiq's sample() DataFrame layout may have changed"
            )
        e = qubo_energy(qubo, const, labels)
        count = int(getattr(row, "counts"))
        total_weighted += e * count
        total_counts += count
        if e < best_energy:
            best_energy, best_labels = e, labels
    mean_energy = total_weighted / total_counts if total_counts else float("inf")
    return mean_energy, best_labels, best_energy


def run_qaoa(
    qubo: dict[tuple[int, int], float],
    const: float,
    n_qubits: int,
    reps: int = DEFAULT_REPS,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    num_shots: int = DEFAULT_NUM_SHOTS,
    backend: str = "simulator",
    on_iteration: Callable[[int, float, float], None] | None = None,
) -> QaoaResult:
    """Runs QAOA on `qubo` (as returned by utils.mrf.energy.build_qubo) via
    Classiq, and returns the best bitstring actually observed across the
    optimization + a final wider sample at the optimized parameters - QAOA
    is a heuristic search, so "best seen" (scored by the real QUBO energy,
    not the circuit's own expectation value) is the honest thing to report,
    the same way a classical local search is judged by its best iterate
    rather than its last one.

    on_iteration(iteration, mean_energy, best_energy_so_far), if given, is
    called synchronously after every real ExecutionSession.sample() round
    trip - the only per-iteration hook available, since each iteration is a
    genuine network call to Classiq's platform, not a local computation to
    poll. Lets a caller (see apps/api/routers/mstar.py's /stream endpoint)
    surface real progress instead of a fake spinner."""
    if n_qubits > MAX_QUBITS:
        raise ValueError(f"QAOA patch capped at {MAX_QUBITS} qubits (one per pixel), got {n_qubits}")

    _configure_classiq_environment()

    from classiq import ExecutionSession, get_circuit_metrics

    h, J, _ising_const = qubo_to_ising(qubo, const, n_qubits)
    hamiltonian = _build_hamiltonian(h, J, n_qubits)
    qprog = _synthesize_qaoa_circuit(n_qubits, hamiltonian, reps)
    metrics = get_circuit_metrics(qprog)
    circuit_depth = metrics.depth if isinstance(metrics.depth, int) else None

    cost_history: list[float] = []
    best_labels, best_energy = None, float("inf")

    # Linear-ramp initial guess (Farhi et al.'s standard QAOA schedule: cost
    # angle ramping up, mixer angle ramping down across layers) rather than
    # all-zeros, which is a stationary point of this ansatz (qaoa_init's
    # uniform superposition is already symmetric under the mixer at beta=0).
    gammas0 = np.linspace(0.1, 0.8, reps)
    betas0 = np.linspace(0.8, 0.1, reps)
    x0 = np.empty(2 * reps)
    x0[0::2] = gammas0
    x0[1::2] = betas0

    with ExecutionSession(qprog, backend=backend) as session:

        def objective(params: np.ndarray) -> float:
            nonlocal best_labels, best_energy
            df = session.sample({"params": list(params)}, num_shots=num_shots)
            mean_energy, iter_best_labels, iter_best_energy = _score_samples(df, qubo, const, n_qubits)
            cost_history.append(mean_energy)
            if iter_best_energy < best_energy:
                best_energy, best_labels = iter_best_energy, iter_best_labels
            if on_iteration is not None:
                on_iteration(len(cost_history), mean_energy, best_energy)
            return mean_energy

        result = minimize(objective, x0=x0, method="COBYLA", options={"maxiter": max_iterations})

        # Wider final sample at the optimized parameters, to firm up the
        # best-observed bitstring beyond whatever num_shots caught during
        # the (cheaper, per-iteration) optimization loop.
        final_df = session.sample({"params": list(result.x)}, num_shots=FINAL_NUM_SHOTS)
        _, final_best_labels, final_best_energy = _score_samples(final_df, qubo, const, n_qubits)
        if final_best_energy < best_energy:
            best_energy, best_labels = final_best_energy, final_best_labels

    return QaoaResult(
        labels=best_labels,
        best_energy=best_energy,
        optimized_params=list(result.x),
        optimizer_cost_history=cost_history,
        n_qubits=n_qubits,
        reps=reps,
        backend=backend,
        circuit_width=metrics.width,
        circuit_depth=circuit_depth,
        num_shots=num_shots,
    )
