"""A differentiable quantum layer usable inside a PyTorch model - the
"Quantum-Enhanced Bottleneck" component of the reference architecture
(ZZFeatureMap-style angle encoding + a trainable RealAmplitudes
variational ansatz, one Z-expectation observable per qubit).

Wrapped via qiskit-machine-learning's TorchConnector, which computes
gradients through the quantum circuit using the parameter-shift rule
automatically under standard PyTorch autograd - both weight gradients
(circuit parameters) AND input gradients (so backprop can flow further
back into the classical encoder) are enabled. This is what lets the whole
network - classical conv layers AND the quantum layer - train end-to-end
with one optimizer (AdamW), rather than a hand-rolled bi-level loop that
alternates a classical optimizer with a separate gradient-free quantum
optimizer (SPSA/COBYLA). That split-optimizer design is a real, valid
alternative (and is what the reference architecture diagram shows) - it
adds real complexity (two loss landscapes, two step schedules, no shared
gradient signal) for a benefit (avoiding parameter-shift's backward-pass
cost) that doesn't materialize here: measured backward cost is
~0.6s/sample on a local simulator, cheap enough at this network's scale
(6-qubit bottleneck, batch size 2) that end-to-end differentiability is
strictly simpler and was chosen over replicating the split design.

Runs on a local Estimator (exact statevector simulation, no shots) by
default - real quantum hardware is not used during training (repeated
forward+backward over many steps at real hardware queue times would be
infeasible), matching this project's established simulation-by-default
pattern (see notebooks 05/06's FORCE_SIMULATION).
"""

from qiskit.circuit.library import real_amplitudes, zz_feature_map
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN


def build_quantum_bottleneck(n_qubits: int, reps: int = 2) -> TorchConnector:
    """n_qubits classical inputs in -> n_qubits real-valued outputs out
    (one Z-expectation per qubit), trainable end-to-end."""
    feature_map = zz_feature_map(feature_dimension=n_qubits, reps=1, entanglement="linear")
    ansatz = real_amplitudes(n_qubits, reps=reps, entanglement="linear")
    circuit = feature_map.compose(ansatz)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (n_qubits - i - 1)) for i in range(n_qubits)]

    qnn = EstimatorQNN(
        circuit=circuit,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        observables=observables,
        input_gradients=True,
    )
    return TorchConnector(qnn)
