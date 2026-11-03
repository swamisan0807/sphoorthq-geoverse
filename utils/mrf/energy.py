"""Binary-label (Q=2) MRF energy for pixel segmentation - the Markov-random-
field formulation from the THALES SAR-segmentation brief (unary + pairwise
energy, MAP inference via graph cut for an Ising prior), applied to this
project's real MSTAR SAR vehicle-target segmentation. Each pixel gets a
unary ("data") cost per label - the chip's own real magnitude (no trained
model: MSTAR's public release has no per-pixel mask to train one against) -
and each 4-connected neighbor pair gets a pairwise ("smoothness") cost that
penalizes disagreeing labels, scaled down across a real intensity edge in
the image so the energy does not smooth away a genuine object boundary.

Q=2 (background/target) by design, not by omission - see
docs/Algorithm_Design_Dossier.md for the reasoning. Three reasons this
project targets Q=2 specifically, not a general Q-ary label set:
  1. THALES's own validation set is binary ("class 0 as the background,
     and class 1 as the object"); 3+ classes (the vehicle's shadow, Q=3)
     is explicitly named "beyond binary" further work in the brief, not
     the core task.
  2. The exactness result below (graph cut = exact MAP) is inherently a
     two-terminal (source/sink) construction. It does not generalize to
     Q>2 - multi-label graph cut falls back to alpha-expansion/alpha-beta-
     swap (Boykov-Veksler-Zabih), only a 2-approximation for metric
     pairwise costs, not exact. Targeting Q=2 is what makes the classical
     baseline's exactness claim (and thus the classical-vs-quantum
     comparison) meaningful in the first place.
  3. Each qubit is inherently 2-level. Q>2 needs either one-hot encoding
     (Q qubits/pixel + penalty terms enforcing exactly-one-hot) or binary/
     log encoding (log2(Q) qubits/pixel) - a real multiplier against the
     already-tight qubit ceiling (utils.qml.classiq_qaoa.MAX_QUBITS=25,
     set by this project's actual Classiq simulator limits), not a free
     generalization.

Two ways to minimize the SAME energy are provided, so a classical run and a
quantum run are scored on the literal same objective, not two different
problems that happen to look similar:

  - solve_graph_cut(): exact, via s-t min-cut (Greig/Porteous/Seheult 1989;
    Boykov & Jolly 2001). Valid because the pairwise term here is a
    ferromagnetic Potts penalty (0 when neighbors agree, >=0 when they
    don't), which is always submodular. This is the classical baseline
    apps/api/routers/mstar.py's QAOA alternative is benchmarked against -
    and the same equivalence the THALES brief cites: "the MAP can in fact
    be computed exactly ... using graph cut, at least when the prior P(x)
    is an Ising distribution."
  - build_qubo() / qubo_to_ising(): the same energy as an unconstrained
    binary quadratic program / Ising Hamiltonian, for a quantum optimizer
    (QAOA via Classiq - see utils/qml/classiq_qaoa.py) to search instead.
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass
class PatchEnergy:
    h: int
    w: int
    unary0: np.ndarray  # cost of assigning label 0, shape (h*w,)
    unary1: np.ndarray  # cost of assigning label 1, shape (h*w,)
    edges: list[tuple[int, int]]  # 4-connected neighbor pairs (i < j), flat pixel indices
    weights: np.ndarray  # disagreement cost for each edge, shape (len(edges),)

    @property
    def n(self) -> int:
        return self.h * self.w


def _flat_index(w: int, r: int, c: int) -> int:
    return r * w + c


def inject_gaussian_noise(intensity: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    """Adds reproducible synthetic Gaussian noise to a normalized [0,1]
    intensity array, clipped back to [0,1] - a controlled, honest way to
    test robustness under varying data conditions (see apps/web's About
    page "Robustness" section) that's separate from, and on top of, the
    real speckle noise already present in the MSTAR magnitude itself.
    sigma=0 returns the array unchanged (no-op, not even a copy needed for
    correctness but returned as one for a consistent return type). A fixed
    seed per call (not a module-level RNG) keeps a given (intensity, sigma)
    pair reproducible regardless of call order - required for the sweep in
    apps/api/routers/mstar.py to be honestly re-runnable."""
    if sigma <= 0:
        return intensity
    rng = np.random.default_rng(seed)
    noisy = intensity + rng.normal(0.0, sigma, size=intensity.shape)
    return np.clip(noisy, 0.0, 1.0)


def build_patch_energy(
    prob_positive: np.ndarray,
    intensity: np.ndarray,
    smoothness: float = 1.0,
    contrast_sigma: float = 0.15,
    eps: float = 1e-6,
) -> PatchEnergy:
    """prob_positive: (h, w) model-predicted P(label=1) per pixel, in (0, 1)
    (e.g. utils.ai.classic.model_io.load_unet()'s sigmoid output, an rf's
    predict_proba, or - for MSTAR, which has no trained model - a chip's
    own normalized magnitude used directly, see apps/api/routers/mstar.py).
    intensity: (h, w) normalized backscatter (e.g. normalized VV) used only
    for the contrast-sensitive smoothness term, not the unary term.

    Unary cost is the negative log-likelihood of the model's own
    probability (theta_i(x_i, z_i, w) in the THALES brief's energy
    formula) - confident-water pixels get a low cost for label 1 and a high
    cost for label 0, and vice versa. Pairwise cost is a contrast-sensitive
    Potts penalty: `smoothness` at a flat, low-contrast boundary, decaying
    towards 0 as the intensity jump across the edge grows past
    contrast_sigma - encourages same-label neighbors, but backs off exactly
    where the image itself shows a real edge.
    """
    h, w = prob_positive.shape
    p = np.clip(prob_positive, eps, 1 - eps)
    unary1 = -np.log(p).reshape(-1)
    unary0 = -np.log(1 - p).reshape(-1)

    edges: list[tuple[int, int]] = []
    edge_weights: list[float] = []
    for r in range(h):
        for c in range(w):
            i = _flat_index(w, r, c)
            if c + 1 < w:
                j = _flat_index(w, r, c + 1)
                d = float(intensity[r, c] - intensity[r, c + 1])
                edges.append((i, j))
                edge_weights.append(smoothness * float(np.exp(-(d * d) / (2 * contrast_sigma**2))))
            if r + 1 < h:
                j = _flat_index(w, r + 1, c)
                d = float(intensity[r, c] - intensity[r + 1, c])
                edges.append((i, j))
                edge_weights.append(smoothness * float(np.exp(-(d * d) / (2 * contrast_sigma**2))))

    return PatchEnergy(h=h, w=w, unary0=unary0, unary1=unary1, edges=edges, weights=np.array(edge_weights))


def energy(problem: PatchEnergy, labels: np.ndarray) -> float:
    """Total MRF energy of a binary label assignment - lower is better.
    The one objective every solver here (classical or quantum) is scored
    against."""
    x = labels.reshape(-1)
    total = float(np.sum(np.where(x == 1, problem.unary1, problem.unary0)))
    for (i, j), w in zip(problem.edges, problem.weights):
        if x[i] != x[j]:
            total += float(w)
    return total


def solve_graph_cut(problem: PatchEnergy) -> np.ndarray:
    """Exact MAP labeling via s-t min-cut. Boykov-Jolly convention: source
    S represents label 1, sink T represents label 0; cutting t-link (S, i)
    strands i on the T side (label 0), paying unary0[i]; cutting (i, T)
    strands i on the S side (label 1), paying unary1[i]. n-links use equal
    capacity in both directions, which reproduces the symmetric
    disagreement cost w_ij for a binary Potts term exactly."""
    g = nx.DiGraph()
    S, T = "s", "t"
    for i in range(problem.n):
        g.add_edge(S, i, capacity=float(problem.unary0[i]))
        g.add_edge(i, T, capacity=float(problem.unary1[i]))
    for (i, j), w in zip(problem.edges, problem.weights):
        w = float(w)
        g.add_edge(i, j, capacity=w)
        g.add_edge(j, i, capacity=w)

    _, (reachable_from_s, _) = nx.minimum_cut(g, S, T)
    labels = np.zeros(problem.n, dtype=np.int64)
    for i in reachable_from_s:
        if i != S:
            labels[i] = 1
    return labels.reshape(problem.h, problem.w)


def build_qubo(problem: PatchEnergy) -> tuple[dict[tuple[int, int], float], float]:
    """The same energy as an unconstrained binary quadratic program:
        E(x) = const + sum_i qubo[i,i]*x_i + sum_{i<j} qubo[i,j]*x_i*x_j,   x_i in {0,1}

    Derivation (x_i=1 selects label 1, x_i=0 selects label 0):
      unary:    x_i*unary1[i] + (1-x_i)*unary0[i] = unary0[i] + x_i*(unary1[i]-unary0[i])
      pairwise: w_ij*(x_i XOR x_j) = w_ij*(x_i + x_j - 2*x_i*x_j)   (exact for binary x)
    """
    const = 0.0
    linear = np.zeros(problem.n)
    quadratic: dict[tuple[int, int], float] = {}

    for i in range(problem.n):
        const += float(problem.unary0[i])
        linear[i] += float(problem.unary1[i] - problem.unary0[i])

    for (i, j), w in zip(problem.edges, problem.weights):
        w = float(w)
        linear[i] += w
        linear[j] += w
        key = (i, j) if i < j else (j, i)
        quadratic[key] = quadratic.get(key, 0.0) - 2 * w

    qubo: dict[tuple[int, int], float] = {(i, i): v for i, v in enumerate(linear) if v != 0.0}
    qubo.update(quadratic)
    return qubo, const


def qubo_energy(qubo: dict[tuple[int, int], float], const: float, labels: np.ndarray) -> float:
    """Evaluates a QUBO (as returned by build_qubo) directly against a
    label assignment - used to cross-check build_qubo() against energy()
    (see tests/test_mrf_energy.py), and to score a QAOA bitstring without
    reconstructing a PatchEnergy."""
    x = labels.reshape(-1)
    total = const
    for (i, j), coeff in qubo.items():
        total += coeff * (x[i] if i == j else x[i] * x[j])
    return float(total)


def qubo_to_ising(
    qubo: dict[tuple[int, int], float], const: float, n: int
) -> tuple[dict[int, float], dict[tuple[int, int], float], float]:
    """Substitutes x_i = (1 - z_i) / 2, z_i in {-1, +1} (the standard QUBO
    <-> Ising map), returning the spin Hamiltonian H(z) = ising_const +
    sum_i h[i]*z_i + sum_{i<j} J[i,j]*z_i*z_j that a QAOA cost layer
    (utils.qml.classiq_qaoa) applies as Z / ZZ rotations. Needed because
    QAOA's phase-separator unitary is naturally expressed in Pauli-Z terms,
    not in the {0,1} QUBO's variables directly."""
    h: dict[int, float] = {i: 0.0 for i in range(n)}
    J: dict[tuple[int, int], float] = {}
    ising_const = const

    for (i, j), coeff in qubo.items():
        if i == j:
            ising_const += coeff / 2
            h[i] -= coeff / 2
        else:
            ising_const += coeff / 4
            h[i] -= coeff / 4
            h[j] -= coeff / 4
            J[(i, j)] = J.get((i, j), 0.0) + coeff / 4

    h = {i: v for i, v in h.items() if v != 0.0}
    return h, J, ising_const


@dataclass
class ChipPatch:
    problem: PatchEnergy
    chip_id: str
    row0: int
    col0: int
    size: int
    intensity: np.ndarray  # (size, size) - the same normalized-VV crop passed into build_patch_energy
    ground_truth: np.ndarray  # (size, size) - sen1floods11 label crop, values in {-1, 0, 1}
    valid_mask: np.ndarray  # (size, size) bool - label != -1 and not S1 nodata


def _predict_prob_positive(s1: np.ndarray, model_name: str) -> np.ndarray:
    """Full-chip P(label=1) map from an already-trained model - the same
    unary/data-term source apps/api/routers/inference.py uses for its own
    predictions, so an MRF/QAOA run and a plain inference run on the same
    chip start from the same model output, not two different data terms."""
    from utils.ai.classic import model_io
    from utils.ai.classic.sen1floods11_dataset import normalize_s1
    from utils.fusion.pixel_features import build_feature_cube

    if model_name == "rf":
        rf = model_io.load_rf()
        cube = build_feature_cube(s1)
        channels, h, w = cube.shape
        flat = cube.reshape(channels, h * w).T
        return rf.predict_proba(flat)[:, 1].reshape(h, w)
    if model_name == "unet":
        import torch

        unet = model_io.load_unet()
        x = torch.from_numpy(normalize_s1(s1)).unsqueeze(0).float()
        with torch.no_grad():
            probs = torch.sigmoid(unet(x))[0, 0]
        return probs.numpy()
    raise ValueError(f"unknown model '{model_name}', expected 'rf' or 'unet'")


def patch_energy_from_chip(
    chip_id: str,
    model_name: str,
    row0: int,
    col0: int,
    size: int,
    smoothness: float = 1.0,
    contrast_sigma: float = 0.15,
) -> ChipPatch:
    """Builds a PatchEnergy for a size x size ROI of a real sen1floods11
    chip, from an already-trained model's own prediction on the FULL chip -
    a CNN/RF needs the surrounding context build_feature_cube's local
    texture window and the U-Net's receptive field both rely on, so
    cropping the raw input first would starve them of it - then crops both
    the resulting probability map and the normalized VV intensity down to
    the requested ROI for the actual MRF/QAOA problem. This is the "tile a
    chip into small patches" step, since a whole chip has far more pixels
    than a QAOA circuit can hold qubits for (see
    utils.qml.classiq_qaoa.MAX_QUBITS). Dormant since this project's pivot
    to MSTAR vehicle segmentation - no live API route calls this anymore
    (apps/api/routers/mrf.py, which used to, was removed) - but kept real
    and working for utils/mrf/experiments.py, which still generates
    evidence from it."""
    from utils.ai.classic.sen1floods11_dataset import normalize_s1, read_label, read_s1, s1_nodata_mask

    s1 = read_s1(chip_id)
    label = read_label(chip_id)
    h, w = label.shape
    if row0 < 0 or col0 < 0 or row0 + size > h or col0 + size > w:
        raise ValueError(f"ROI (row0={row0}, col0={col0}, size={size}) is out of bounds for a {h}x{w} chip")

    prob_full = _predict_prob_positive(s1, model_name)
    intensity_full = normalize_s1(s1)[0]  # VV channel

    sl = (slice(row0, row0 + size), slice(col0, col0 + size))
    prob = prob_full[sl]
    intensity = intensity_full[sl]
    ground_truth = label[sl]
    valid_mask = (ground_truth != -1) & ~s1_nodata_mask(s1)[sl]

    problem = build_patch_energy(prob, intensity, smoothness=smoothness, contrast_sigma=contrast_sigma)
    return ChipPatch(
        problem=problem,
        chip_id=chip_id,
        row0=row0,
        col0=col0,
        size=size,
        intensity=intensity,
        ground_truth=ground_truth,
        valid_mask=valid_mask,
    )
