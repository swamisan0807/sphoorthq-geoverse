# Algorithm Design Dossier — SAR Vehicle-Target Segmentation (MRF → QUBO → QAOA)

This is SphoorthiQ's submission to the THALES SAR (Synthetic Aperture Radar) Image Segmentation use-case.
Every figure in this document comes from runs of the code in this repository on real MSTAR SAR data, with one
labelled exception: the synthetic validation cases in [§12b](#12b-synthetic-validation--real-accuracy--iou--dice),
which have known answers by construction. Real-chip scores use the SARBake reference masks, which are derived from CAD models (see [§12c](#12c-real-mask-results-sarbake)
and [Assumptions](#5-assumptions)). Where a figure cannot be produced with confidence, this is stated rather than estimated.

## What this project does

Given a real radar image of a vehicle in noisy background clutter, the task is to label which pixels belong to the
vehicle and which belong to the background. The same problem is solved in two ways and the results are compared:

- **Classically**, with graph cut, an algorithm that is guaranteed to find the optimal labelling for two classes at any
  image size.
- **With a quantum algorithm**, QAOA, executed through Classiq's platform on its simulator backend. QAOA searches for a good
  labelling rather than guaranteeing the best one. No physical quantum hardware is connected (see §15), so "quantum" here
  means a real quantum circuit executed on a simulator.

The purpose is not to select a winner but to compare the two methods on one objective. At small scale (9 and 16 qubits)
QAOA finds the same labelling as the classical method on the tiles tested. At 25 qubits it falls behind, because current
simulators can handle only a limited number of qubits before the search becomes harder than the optimizer budget allows.
This gap is reported as measured, because it reflects the present state of quantum optimization rather than a defect of
the implementation.

The additional topics mentioned in the THALES brief are also implemented and can be run in the application: a third class
(the vehicle's shadow), edge detection between regions, and recognition of the vehicle type shown in a chip.

**Where to find what.** Sections 1-11 set out the formulation and the classical and quantum comparison (§7b explains how
the two results are connected: the same objective, scored the same way). §12 tests the comparison under noise. §12b reports
exact accuracy on test cases with known answers. §12c scores real chips against the SARBake masks. §13 covers the shadow
and pattern-detection extensions. §15 and §16 describe the limitations and next steps.

## 1. Industrial problem framing

THALES asks for SAR imagery to be segmented into background and target regions (optionally shadow too),
using a Markov Random Field formulated so quantum parallelism can evaluate every pixel-neighbor
interaction at once - sidestepping the exponential blow-up classical MRF solvers hit as an image grows.
We validate against real MSTAR vehicle-target chips: T72, BMP2, and BTR70 tanks, real X-band SAR.

## 2. Objective function

Minimize the MRF energy:

```
E(x) = Σ θ_i(x_i, z_i)  +  Σ Ψ_i,j(x_i, x_j)
       i∈V              (i,j)∈E
```

`θ_i` — unary ("data") cost: how well label `x_i` fits pixel `i`'s own real magnitude `z_i`.
`Ψ_i,j` — pairwise ("smoothness") cost: a contrast-sensitive Potts penalty for disagreeing neighbors,
scaled down across a real intensity edge so the energy doesn't smooth away an actual object boundary.
Identical objective for the classical and quantum solver — they're benchmarked on the same number, not
two formulations that happen to look similar.

## 3. Variables

One label per pixel. Q=2 (background/target): one binary variable, one qubit, per pixel. Q=3
(background/target/shadow, see [§13](#13-beyond-binary-extensions)): one-hot encoded, 3 qubits per pixel.

## 4. Constraints

- Exactly one label per pixel — structural for Q=2 (a binary variable can't be two labels at once);
  enforced by an explicit penalty term in the Q=3 one-hot QUBO (nothing else stops it favoring two
  labels simultaneously).
- 4-connected pairwise neighbor structure only, not 8-connected.
- A real, measured qubit ceiling (~25 qubits on the Classiq simulators available to this project) bounds
  ROI size directly: 5×5 for Q=2, 2×2 for Q=3.
- QAOA's classical outer loop is budget-limited (COBYLA, default 15 iterations) — a real, disclosed
  compute cap, not an unlimited search.

## 5. Assumptions

- **MSTAR's public release has no per-pixel ground truth.** Real chips are scored against the SARBake
  overlay ([§12c](#12c-real-mask-results-sarbake)), whose masks come from 3D CAD models of the vehicles. They
  are a good reference, not hand-drawn truth. For chips without a mask (SLICY), and whenever the masks are
  missing, energy is the only comparison between classical and quantum.
- The chip's own brightness in decibels, measured against its background, is assumed sufficient unary
  evidence with no trained classifier behind it — the same assumption THALES's own worked example makes ("We consider here only
  the intensity to label the pixels").
- The pairwise term assumes same-label neighbors are more likely than not (a standard Potts-style vision
  prior), not something fit to MSTAR-specific statistics.
- The robustness sweep's synthetic noise ([§12](#12-robustness)) assumes i.i.d. Gaussian perturbation is
  a reasonable controlled stressor, layered on top of — not a replacement for — the real SAR speckle
  already present in the chip.

## 6. Candidate formulations considered

- **One-hot vs. binary/log₂(Q) qubit encoding** for Q=3 — one-hot was chosen despite tripling the
  per-pixel qubit cost, because a binary/log encoding needs its own nontrivial penalty terms to rule out
  invalid codewords, and this project's standard is verified-exact-or-clearly-labeled-approximate, not
  clever-but-unverified.
- **QAOA vs. quantum annealing** — THALES's own reference (Otgonbaatar & Datcu, IGARSS 2020) uses
  annealing for SAR segmentation. QAOA was chosen instead because Classiq/Qmod, the platform this project
  standardizes on, is a gate-model toolkit, not an annealer.
- **Alpha-expansion vs. alpha-beta swap** for the Q=3 classical baseline — alpha-expansion is the
  standard Boykov-Veksler-Zabih choice for Potts-like pairwise costs.

## 7. Chosen formulation

```
MRF  →  QUBO (verified exact)  →  Ising Hamiltonian  →  QAOA
```

QUBO reformulation checked **exact** over all 65,536 states of a 4×4 test patch — max deviation between
the MRF energy and the QUBO energy: 2.13×10⁻¹⁴ (float noise). Evidence file:
`datasets/reports/11_mrf_qubo_verification.json`; reproducible via `utils/mrf/verification.py`'s
`exhaustive_equivalence_check` and `tests/test_mrf_energy.py`.

## 7b. How the classical and quantum results actually connect

This is the part that makes "classical energy = 3.15, quantum energy = 3.15" a real comparison rather than
two unrelated numbers that happen to look similar — walked through file by file:

1. **One input, one problem object.** `utils/mrf/mstar_scenes.py` reads the real MSTAR ROI and normalizes
   it. `utils/mrf/energy.py`'s `build_patch_energy()` turns that into a single `PatchEnergy` object (unary
   + pairwise costs). Nothing about this object is classical or quantum yet - both solvers branch off the
   exact same one.
2. **Classical path**: `utils/mrf/solvers.py`'s `run_classical()` calls `solve_graph_cut()` directly on
   that `PatchEnergy`, and scores the result with `energy(problem, labels)`.
3. **Quantum path**: `run_quantum()` first calls `build_qubo(problem)` - the *same* `PatchEnergy` object,
   rewritten algebraically as a QUBO matrix. This is the connection point: `build_qubo()` is proven
   identical to `energy()` (§7's exactness check), so the QUBO isn't an approximation of the problem, it's
   the same problem in a different, quantum-compatible shape.
4. That QUBO goes to `utils/qml/classiq_qaoa.py`: `qubo_to_ising()` converts it to a Pauli-Z Hamiltonian,
   a Qmod circuit (`qaoa_init` + `qaoa_layer`) gets synthesized and executed on Classiq's platform, and
   Classiq returns sampled bitstrings.
5. **The critical step**: each bitstring is converted to labels and scored by `qubo_energy()` - the *same
   function, on the same QUBO*, that step 3 built from the *same* `PatchEnergy` the classical solver used.
   Not a generic Hamiltonian expectation value that only looks related - the literal identical objective.

So classical minimizes `energy(problem, labels)` exactly; quantum searches for labels that minimize
`qubo_energy(qubo, const, labels)` - two evaluations of a mathematically proven-identical objective. That
identity is what every energy comparison in this document (§9, §12, §12b) actually rests on.

## 8. Quantum suitability

Correct at small scale: QAOA matches the exact classical optimum at 9 and 16 qubits on two real chips,
including across the full noise sweep in [§12](#12-robustness) at 9 qubits. At 9 qubits random sampling
finds the optimum too, so that match shows the pipeline is correct and nothing more. Qubit-budget- and
optimizer-limited at 25 qubits - see [§9](#9-feasibility--scalability) for the shortfall.

## 9. Feasibility & scalability

Two real MSTAR test chips (T72 #132, BMP2 #9563), at the three qubit tiers Classiq's own simulators
support today (no general-purpose simulator on this account exceeded ~28-35 qubits, so a naive 6×6=36
tier was never realistic). Each tile sits on the vehicle's edge, where about half of it is target
according to the SARBake mask, so the solver has a real decision to make. The random-sampling column is
the best of about 9,700 uniformly random labelings (QAOA's sample budget), averaged over 30 seeds
(`scripts/random_baseline.py`):

| chip | size | qubits | classical energy | quantum energy | random sampling | genetic algorithm | classical / quantum IoU | QAOA duration |
|---|---|---|---|---|---|---|---|---|
| T72 #132 (test) | 3×3 | 9 | 6.5307 | 6.5307 | 6.5307 | 6.5307 | 0.40 / 0.40 | 66.5s |
| BMP2 #9563 (test) | 3×3 | 9 | 5.3805 | 5.3805 | 5.3805 | 5.3805 | 0.75 / 0.75 | 62.7s |
| T72 #132 (test) | 4×4 | 16 | 11.3321 | 11.3321 | 12.1226 | 11.4294 | 0.44 / 0.44 | 67.6s |
| BMP2 #9563 (test) | 4×4 | 16 | 7.4675 | 7.4675 | 8.7559 | 7.4675 | 0.78 / 0.78 | 70.7s |
| T72 #132 (test) | 5×5 | 25 | 13.8564 | 14.3462 | 20.1669 | 13.9659 | 0.59 / 0.57 | 127.5s |
| BMP2 #9563 (test) | 5×5 | 25 | 12.1758 | 13.2295 | 18.4777 | 12.3871 | 0.71 / 0.67 | 124.2s |

QAOA matches the classical optimum at 9 and 16 qubits on both chips. At 25 qubits, with the same
optimizer budget, it falls short on both (14.3462 against 13.8564, and 13.2295 against 12.1758), and its
IoU against the mask is a little lower. Random sampling finds the optimum on every seed at 9 qubits, on 6
and 3 of 30 seeds at 16 qubits, and on none at 25, and QAOA's energy is below the random average in all
four of the 16- and 25-qubit rows. A genetic algorithm with the same budget (§11) does better than random sampling at 16 and 25 qubits and than QAOA at 25 qubits. Each QAOA
value is a single run and QAOA is random, so repeat runs can differ. The IoU figures are for these two chips only; averaged over 120 chips, edge tiles of this size score lower
([§12c](#12c-real-mask-results-sarbake)). We do not claim a quantum advantage: the classical solver is exact and takes milliseconds.
Classical graph cut has no qubit limit: on 8×8 edge patches (64 pixels, four times QAOA's ceiling) it got
energy 34.1487 (T72 #132) and 27.4969 (BMP2 #9563) in about 3 ms and 2 ms, with IoU 0.71 and 0.78 against
the mask. Classical scales easily, and quantum is qubit-capped today.

## 10. Execution pathway

```
Classiq Qmod (qaoa_init + qaoa_layer)  →  synthesis  →  ExecutionSession  →  scipy-driven classical optimizer loop
```

Every sampled bitstring is scored against the real QUBO energy — the same objective the classical solver
is scored on — not a generic Hamiltonian expectation value.

## 11. Classical baseline

Exact MAP labeling via s-t min-cut graph cut (Greig, Porteous & Seheult 1989) — valid because the
pairwise term is a ferromagnetic Potts penalty, always submodular. This is the exact method THALES's own
document names ("the MAP can in fact be computed exactly ... using graph cut, at least when the prior
P(x) is an Ising distribution") — stronger than a heuristic baseline, and the reason the classical-vs-quantum energy comparison is meaningful in the first place. The brief also names a MATLAB genetic algorithm as its baseline. MATLAB is not part of this project, so `scripts/ga_baseline.py` runs a genetic algorithm implemented in Python (not MATLAB's `ga`) on the same tiles and energy, with QAOA's evaluation budget. It reaches the optimum on 27 to 30 of 30 seeds on every scaling tile except the 25-qubit BMP2 tile (18 of 30), and its average energy at 25 qubits (13.9659 and 12.3871) is below QAOA's (14.3462 and 13.2295).

## 12. Robustness

A controlled evaluation under varying data conditions: a fixed real ROI (T72 #132, the same 3×3 edge tile
as the first row of §9, where QAOA matches classical) run at increasing levels of synthetic Gaussian noise
added on top of the real chip, seeded for reproducibility (`utils/mrf/energy.py`'s
`inject_gaussian_noise`):

| noise σ | classical energy | quantum energy | QAOA matches classical | classical / quantum IoU | QAOA duration |
|---|---|---|---|---|---|
| 0.0 | 6.5307 | 6.5307 | yes | 0.40 / 0.40 | 63.3s |
| 0.05 | 6.1372 | 6.1372 | yes | 0.56 / 0.56 | 62.6s |
| 0.1 | 5.7537 | 5.7537 | yes | 0.62 / 0.62 | 66.8s |
| 0.15 | 4.7773 | 4.7773 | yes | 0.62 / 0.62 | 62.9s |
| 0.2 | 4.1112 | 4.1112 | yes | 0.62 / 0.62 | 63.2s |
| 0.3 | 3.3779 | 3.3779 | yes | 0.50 / 0.50 | 60.0s |

QAOA matched the exact classical optimum at every noise level, so at this size the match does not depend
on clean data. Energies cannot be compared between rows, because each noise level is a different image.
The scores against the mask go from 0.40 with no added noise to 0.62 at moderate noise and 0.50 at the
highest level. On a 9-pixel tile one pixel moves IoU a lot, so we do not read a trend into that. QAOA and
the classical solver found the same labeling in every row. Reproducible via `scripts/robustness_sweep.py`
against a running instance of the app.

## 12b. Synthetic validation — real Accuracy / IoU / Dice

These cases are not MSTAR chips. Each has a known right answer by construction, so Accuracy/IoU/Dice
are exact here. Two sources: THALES's own worked example,
quoted directly from page 3 of their brief (the same 5×5 intensity matrix and label matrix printed in the
document, not paraphrased), and a few small generated cases with a known target region
(`utils/mrf/synthetic_validation.py`). Runnable live via `POST /api/mstar/synthetic-validation`, or
reproduced via `scripts/synthetic_validation_run.py`.

| case | size | classical acc/IoU/Dice | quantum acc/IoU/Dice | classical energy | quantum energy | QAOA duration |
|---|---|---|---|---|---|---|
| THALES worked example (first run) | 5×5 (25q) | 1.00 / 1.00 / 1.00 | 0.96 / 0.92 / 0.96 | 1.4751 | 5.6723 | 94.6s |
| clean 3×3, corner target | 3×3 (9q) | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 0.5268 | 0.5268 | 68.4s |
| clean 4×4, block target | 4×4 (16q) | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.2643 | 1.2643 | 64.8s |
| noisy 3×3, σ=0.15 | 3×3 (9q) | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 0.9571 | 0.9571 | 58.1s |

The classical solver reproduces THALES's own stated answer exactly on every case, including their own
worked example — the strongest validation this project can offer. QAOA matches exactly at 9 and, on this
cleaner synthetic case, at 16 qubits too, as in the Scaling sweep in §9, where QAOA is exact up to 16 qubits. At 25 qubits (THALES's own exact example size), QAOA falls short
here too (92% to 96% accuracy across four runs) — consistent with the real qubit-scaling limit disclosed throughout this
document, not contradicted by the 16-qubit result.

## 12c. Real-mask results (SARBake)

MSTAR's public release has no per-pixel masks, but the SARBake overlay (Malmgren-Hansen and
Nobel-Jørgensen, CC BY 4.0, doi:10.17632/jxhsg8tj7g.3) provides one for every T72, BMP2 and BTR70 chip we
use: background, target or shadow per pixel, generated from a 3D CAD model of the vehicle at the chip's
viewing angle (`utils/mrf/sarbake.py`). It is a good reference, not hand-drawn ground truth. It covers the
bright vehicle return rather than every pixel a person might outline, and includes thin antenna slivers.

The unary term measures each pixel's brightness in decibels against its own chip's background, with a few
constants (where a pixel starts to count as target, and the expected brightness of target and shadow) set on
the MSTAR TRAIN chips (17°). The classical solvers were then scored on 120 random TEST chips (15°), 40 each
of BMP2, BTR70 and T72, using a 48×48 region around each vehicle (`scripts/sarbake_eval.py`, no quantum
runs). The earlier unary term used the log-scaled, min-max normalized magnitude directly as the target
probability.

| score on 120 TEST chips | earlier setup | decibel setup |
|---|---|---|
| two classes: target IoU | 0.04 | 0.56 |
| two classes: target Dice | 0.07 | 0.70 |
| two classes: precision / recall | 0.99 / 0.04 | 0.79 / 0.66 |
| two classes: chips with IoU above 0.5 | 0 of 120 | 89 of 120 |
| three classes: pixel accuracy | 0.44 | 0.82 |
| three classes: mean IoU | 0.16 | 0.66 |
| three classes: target / shadow IoU | 0.03 / 0.00 | 0.61 / 0.72 |

The earlier setup marked about 13 target pixels where the mask has about 320, so nearly everything came
out as background. That is why it was replaced. Per vehicle, two-class IoU with the decibel setup is 0.55
(BMP2), 0.50 (BTR70) and 0.62 (T72).

On the small edge tiles the quantum solver can run on, scores are much lower. Each tile is placed where the mask is about half target, which is the hardest case. Averaged over the same 120 chips, target IoU is 0.12 on 2x2 tiles, 0.18 on 4x4 and 0.34 on 8x8. For three classes on these tiles, pixel accuracy is 0.32 to 0.36, below the 0.50 you would get by always choosing the most common class. A few speckled pixels give little evidence for separating target, shadow and background. The two chips in the Scaling table score higher than these averages, so their IoU should not be read as typical. These small-tile scores are in `datasets/reports/sarbake_eval.json`
under `edge_tiles`.

## 13. Beyond binary extensions

THALES's brief lists "beyond binary" bullets with no further spec. All are real, running extensions of
the same app, not future work:

- **Edge detection** — THALES's literal definition (a pixel is an edge if any neighbor differs), plus
  both of the document's own named refinements: single-pixel edge marking (one pixel per boundary,
  instead of two) and multi-class edge detection distinguishing object edges from shadow edges (once a
  Q=3 run exists).
- **3+ classes (shadow, Q=3)** — background/target/shadow via a multi-label MRF, solved classically by
  alpha-expansion (approximate — Q=2's exactness doesn't generalize past two labels) and by QAOA over a
  one-hot QUBO, capped at a 2×2 tile (12 qubits; 3×3 would need 27, over the real ceiling).
- **Pattern detection** — THALES names this bullet with zero definition. This project's disclosed
  interpretation: target-type recognition (BMP2/BTR70/T72) via correlation-filter matching against real
  per-class templates built from the real MSTAR train split. Measured, not inflated: 51.79% real accuracy
  on 1,365 real test chips (33% is chance for 3 classes) — real ground truth exists here (each chip's own
  header), unlike the segmentation task itself.

## 14. System integration

`GET /api/mstar/chips`, `POST /api/mstar`, `POST /api/mstar/stream`, `POST /api/mstar/shadow`,
`POST /api/mstar/pattern`, `GET /api/mstar/pattern/evaluation`, `POST /api/mstar/synthetic-validation` — all
wired into a running FastAPI + React application (`apps/api`, `apps/web`), not a standalone notebook.
Live-streaming real per-iteration QAOA progress, not a simulated loading animation.

## 15. Known limitations

- Real qubit ceiling (~25 qubits) caps Q=2 at a 5×5 tile, Q=3 at a 2×2 tile. A full 128×128 chip needs
  tiling into many small ROIs; this app runs one ROI at a time and says so.
- No hand-drawn ground truth for MSTAR. Real-chip scores use the SARBake masks, which come from CAD models,
  so they are a good guide and not perfect truth. Chips without a mask (SLICY) are compared on energy only.
- The decibel constants were tuned on a small grid with some values at the edge of the range, so a wider
  search might do slightly better. The real-mask scores are classical only. Quantum was scored on edge
  tiles of up to 5×5 pixels (§9), not on whole regions.
- Q=3 loses the exactness guarantee of Q=2: alpha-expansion is a working but approximate algorithm.
- Pattern detection's 51.79% accuracy is a deliberately simple, transparent baseline, not a claim of
  matching modern CNN-based MSTAR classifiers (~95%+).
- Every quantum result here runs on a Classiq simulator backend (`simulator` / `nvidia_simulator` /
  `dgx_simulator`) — no real QPU wired up yet.

## 16. Next-step roadmap

- Full-chip tiling with cross-patch consistency, so one run covers a whole 128×128 chip instead of a
  single ROI.
- Execution on real quantum hardware.
- General Q-ary generalization beyond Q=3, if a real requirement ever calls for it — the
  alpha-expansion/QUBO machinery in `utils/mrf/multilabel_energy.py` already loops over a generic label
  count internally, so this would be a scoped extension, not a rewrite.

## Reproducibility

- QUBO exactness: `tests/test_mrf_energy.py`, `utils/mrf/verification.py`
- Scaling sweep: `scripts/scaling_sweep.py`
- Robustness sweep: `scripts/robustness_sweep.py`
- Synthetic validation sweep: `scripts/synthetic_validation_run.py`
- Real-mask evaluation: `scripts/sarbake_eval.py` (classical only, no Classiq session needed)
- Random-sampling baseline: `scripts/random_baseline.py`
- Full test suite: `pytest tests/ -q` (53 tests, real MSTAR chips, no mocks)
- Live app: see [README.md](../README.md) for setup and run instructions
