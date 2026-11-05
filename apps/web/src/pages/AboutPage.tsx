// Scores from scripts/sarbake_eval.py (datasets/reports/sarbake_eval.json): 120 random MSTAR TEST chips
// (40 per vehicle), 48x48 region around each vehicle, classical solvers only, scored against the SARBake masks.
const REAL_MASK_ROWS = [
  { label: "two classes: target IoU", before: "0.04", after: "0.56" },
  { label: "two classes: target Dice", before: "0.07", after: "0.70" },
  { label: "two classes: precision / recall", before: "0.99 / 0.04", after: "0.79 / 0.66" },
  { label: "two classes: chips with IoU above 0.5", before: "0 of 120", after: "89 of 120" },
  { label: "three classes: pixel accuracy", before: "0.44", after: "0.82" },
  { label: "three classes: mean IoU", before: "0.16", after: "0.66" },
  { label: "three classes: target / shadow IoU", before: "0.03 / 0.00", after: "0.61 / 0.72" },
];

// Real sweep: two MSTAR chips (T72 #132, BMP2 #9563), three qubit tiers, run
// live against this app's own POST /api/mstar with scripts/scaling_sweep.py.
// Each tile sits on the vehicle's edge (half target by the SARBake mask), and
// both results are scored against that mask. randomE is the lowest energy
// found by random sampling with QAOA's sample budget, averaged over 30 seeds
// (scripts/random_baseline.py). gaE is the same for a Python genetic algorithm
// (scripts/ga_baseline.py). Each QAOA value is one run.
const SWEEP = [
  { chip: "T72 #132 (test)", size: 3, qubits: 9, classicalE: 6.5307, quantumE: 6.5307, randomE: 6.5307, gaE: 6.5307, classicalIoU: 0.4, quantumIoU: 0.4, quantumDuration: 66.5 },
  { chip: "BMP2 #9563 (test)", size: 3, qubits: 9, classicalE: 5.3805, quantumE: 5.3805, randomE: 5.3805, gaE: 5.3805, classicalIoU: 0.75, quantumIoU: 0.75, quantumDuration: 62.7 },
  { chip: "T72 #132 (test)", size: 4, qubits: 16, classicalE: 11.3321, quantumE: 11.3321, randomE: 12.1226, gaE: 11.4294, classicalIoU: 0.44, quantumIoU: 0.44, quantumDuration: 67.6 },
  { chip: "BMP2 #9563 (test)", size: 4, qubits: 16, classicalE: 7.4675, quantumE: 7.4675, randomE: 8.7559, gaE: 7.4675, classicalIoU: 0.78, quantumIoU: 0.78, quantumDuration: 70.7 },
  { chip: "T72 #132 (test)", size: 5, qubits: 25, classicalE: 13.8564, quantumE: 14.3462, randomE: 20.1669, gaE: 13.9659, classicalIoU: 0.59, quantumIoU: 0.57, quantumDuration: 127.5 },
  { chip: "BMP2 #9563 (test)", size: 5, qubits: 25, classicalE: 12.1758, quantumE: 13.2295, randomE: 18.4777, gaE: 12.3871, classicalIoU: 0.71, quantumIoU: 0.67, quantumDuration: 124.2 },
];

// Robustness sweep (scripts/robustness_sweep.py): the T72 #132 3x3 edge tile
// from the first Scaling row, at increasing Gaussian noise
// (utils.mrf.energy.inject_gaussian_noise) added on top of the real chip, both
// results scored against the SARBake mask. Each noise level is a different
// image, so energies are not comparable between rows.
const NOISE_SWEEP = [
  { sigma: 0.0, classicalE: 6.5307, quantumE: 6.5307, classicalIoU: 0.4, quantumIoU: 0.4, quantumDuration: 63.3 },
  { sigma: 0.05, classicalE: 6.1372, quantumE: 6.1372, classicalIoU: 0.56, quantumIoU: 0.56, quantumDuration: 62.6 },
  { sigma: 0.1, classicalE: 5.7537, quantumE: 5.7537, classicalIoU: 0.62, quantumIoU: 0.62, quantumDuration: 66.8 },
  { sigma: 0.15, classicalE: 4.7773, quantumE: 4.7773, classicalIoU: 0.62, quantumIoU: 0.62, quantumDuration: 62.9 },
  { sigma: 0.2, classicalE: 4.1112, quantumE: 4.1112, classicalIoU: 0.62, quantumIoU: 0.62, quantumDuration: 63.2 },
  { sigma: 0.3, classicalE: 3.3779, quantumE: 3.3779, classicalIoU: 0.5, quantumIoU: 0.5, quantumDuration: 60.0 },
];

// Real Accuracy/IoU/Dice - the one place these exist in this app, because
// it is the one place real ground truth exists: THALES's own worked
// example (quoted directly from their brief, page 3) plus small generated
// cases with a known target region (utils/mrf/synthetic_validation.py).
// Run live via scripts/synthetic_validation_run.py against this app's own
// POST /api/mstar/synthetic-validation. None of these cases is an MSTAR chip.
// The 16-qubit row matches exactly, as in the Scaling sweep. The 25-qubit THALES
// example falls short in every run (accuracy 0.92 to 0.96 over four runs); the
// table shows the first run.
const SYNTHETIC_VALIDATION = [
  { name: "THALES worked example", size: 5, qubits: 25, classicalAcc: 1.0, quantumAcc: 0.96, classicalIoU: 1.0, quantumIoU: 0.9167, classicalE: 1.4751, quantumE: 5.6723, quantumDuration: 94.6 },
  { name: "clean 3x3, corner target", size: 3, qubits: 9, classicalAcc: 1.0, quantumAcc: 1.0, classicalIoU: 1.0, quantumIoU: 1.0, classicalE: 0.5268, quantumE: 0.5268, quantumDuration: 68.4 },
  { name: "clean 4x4, block target", size: 4, qubits: 16, classicalAcc: 1.0, quantumAcc: 1.0, classicalIoU: 1.0, quantumIoU: 1.0, classicalE: 1.2643, quantumE: 1.2643, quantumDuration: 64.8 },
  { name: "noisy 3x3, sigma=0.15", size: 3, qubits: 9, classicalAcc: 1.0, quantumAcc: 1.0, classicalIoU: 1.0, quantumIoU: 1.0, classicalE: 0.9571, quantumE: 0.9571, quantumDuration: 58.1 },
];

interface WfStage {
  key: string;
  title: string;
  detail: string;
  kind: "challenge" | "input" | "prep" | "roi" | "classical" | "quantum" | "merge" | "metrics" | "compare" | "scaling" | "dossier" | "pattern";
}

const CHALLENGE: WfStage = { key: "challenge", kind: "challenge", title: "THALES SAR", detail: "Industrial challenge: SAR image segmentation with quantum optimization" };
const LINEAR: WfStage[] = [
  { key: "input", kind: "input", title: "MSTAR SAR chip", detail: "X-band target chip, 128x128 (or 54x54 for SLICY). THALES describes 64x64 pre-segmented images. The public MSTAR release has 128x128 chips, and the masks come from the separate SARBake overlay" },
  { key: "prep", kind: "prep", title: "Preprocessing", detail: "Convert to decibels, measured against the chip's own background" },
  { key: "roi", kind: "roi", title: "ROI selection", detail: "Row, column and size. Up to 5x5 (25 qubits) for QAOA, larger for the classical solver alone" },
];
const CLASSICAL_BRANCH: WfStage[] = [
  { key: "c1", kind: "classical", title: "Classical MRF", detail: "Unary + pairwise energy" },
  { key: "c2", kind: "classical", title: "Classical optimization", detail: "Exact s-t min-cut graph cut" },
  { key: "c3", kind: "classical", title: "Classical segmentation", detail: "Exact MAP labeling, no qubit limit" },
];
const QUANTUM_BRANCH: WfStage[] = [
  { key: "q1", kind: "quantum", title: "QUBO", detail: "The same energy, rewritten as a QUBO and checked exact on all 65,536 labelings of a 4x4 patch" },
  { key: "q2", kind: "quantum", title: "Ising Hamiltonian", detail: "The QUBO converted into the form QAOA works with" },
  { key: "q3", kind: "quantum", title: "QAOA circuit on Classiq", detail: "Built from Classiq's qaoa_init and qaoa_layer (2 layers by default) and synthesized on their platform" },
  { key: "q4", kind: "quantum", title: "Optimizer loop", detail: "COBYLA tunes the circuit for up to 15 steps. Each step is one 512-shot job, scored on the real QUBO energy" },
  { key: "q5", kind: "quantum", title: "Final sample", detail: "2,048 shots at the tuned settings. The lowest-energy bitstring seen becomes the mask" },
  { key: "q6", kind: "quantum", title: "Classiq simulator", detail: "Every run so far is on a simulator (simulator, nvidia_simulator or dgx_simulator). No real quantum hardware is connected" },
];
const TAIL: WfStage[] = [
  { key: "merge", kind: "merge", title: "Segmentation masks", detail: "Classical + quantum results, same chip, same ROI" },
  { key: "metrics", kind: "metrics", title: "Energy comparison", detail: "Energy for both solvers, plus IoU and Dice against the SARBake mask where one exists" },
  { key: "compare", kind: "compare", title: "Classical vs quantum", detail: "Side-by-side comparison & analysis" },
  { key: "scaling", kind: "scaling", title: "Feasibility & scaling", detail: "Qubit ceiling (9, 16 and 25 qubits tested), tiling and known limits. See Scaling below" },
  { key: "dossier", kind: "dossier", title: "Algorithm dossier", detail: "The final report, part of this running app rather than a separate notebook" },
];

// THALES's "beyond binary" bullets - real, shipped extensions of the same
// pipeline, not planned/future work (see apps/api/routers/mstar.py's
// /shadow and /pattern* endpoints, utils/mrf/multilabel_energy.py,
// utils/mstar/pattern_recognition.py). Shown as their own small diagrams
// rather than folded into the Q=2 diagram above because they branch off
// at different points (shadow reuses the ROI; pattern detection runs on
// the whole chip, independent of ROI/size) and pattern detection is not an
// MRF/QUBO/QAOA pipeline at all - honest to keep it visually separate.
const SHADOW_ROI: WfStage = { key: "shadow-roi", kind: "roi", title: "Same ROI", detail: "Reuses the row, column and size from the two-class pipeline above, so the same chip patch" };
const SHADOW_MRF: WfStage = { key: "shadow-mrf", kind: "prep", title: "Multi-label MRF", detail: "Three labels this time: background, target and shadow (Q=3), in utils/mrf/multilabel_energy.py" };
const SHADOW_CLASSICAL_BRANCH: WfStage[] = [
  {
    key: "sc1",
    kind: "classical",
    title: "Alpha-expansion",
    detail: "Repeated binary graph-cut moves (Boykov-Veksler-Zabih). Approximate for three labels, unlike the single exact cut for two",
  },
];
const SHADOW_QUANTUM_BRANCH: WfStage[] = [
  { key: "sq1", kind: "quantum", title: "One-hot QUBO", detail: "3 qubits/pixel + an exactly-one-label penalty term" },
  {
    key: "sq2",
    kind: "quantum",
    title: "QAOA (Classiq)",
    detail: "Same Classiq/Qmod route as two labels, capped at a 2x2 tile (12 qubits). A 3x3 tile needs 27, over the 25-qubit ceiling",
  },
];
const SHADOW_OUT: WfStage = { key: "shadow-out", kind: "merge", title: "3-class segmentation", detail: "background / target / shadow mask" };

const PATTERN_STAGES: WfStage[] = [
  { key: "p1", kind: "pattern", title: "MSTAR chip", detail: "The whole chip, not just a region. This is the brief's \"Pattern detection\" bullet, which it leaves undefined" },
  { key: "p2", kind: "pattern", title: "Target patch extraction", detail: "Connected-component blob isolation -> 32x32 crop centered on the real target" },
  {
    key: "p3",
    kind: "pattern",
    title: "Correlation-filter match",
    detail: "Pearson correlation vs. real per-class templates built from the real MSTAR train split (17-deg depression)",
  },
  {
    key: "p4",
    kind: "pattern",
    title: "Predicted target type",
    detail: "bmp2_tank, btr70_transport or t72_tank. 51.79% accuracy on 1,365 test chips (15 degrees), against 33% by chance",
  },
];

function ShadowDiagram() {
  return (
    <div className="wf-diagram wf-diagram-compact">
      <div className="wf-row">
        <WfBox s={SHADOW_ROI} />
        <div className="wf-arrow-down" />
      </div>
      <div className="wf-row">
        <WfBox s={SHADOW_MRF} />
        <div className="wf-arrow-down" />
      </div>
      <div className="wf-branch-row">
        <div className="wf-branch">
          {SHADOW_CLASSICAL_BRANCH.map((s, i) => (
            <div key={s.key} className="wf-row">
              <WfBox s={s} />
              {i < SHADOW_CLASSICAL_BRANCH.length - 1 && <div className="wf-arrow-down" />}
            </div>
          ))}
          <div className="wf-filler" />
        </div>
        <div className="wf-branch">
          {SHADOW_QUANTUM_BRANCH.map((s, i) => (
            <div key={s.key} className="wf-row">
              <WfBox s={s} />
              {i < SHADOW_QUANTUM_BRANCH.length - 1 && <div className="wf-arrow-down" />}
            </div>
          ))}
        </div>
      </div>
      <div className="wf-converge" />
      <div className="wf-row">
        <WfBox s={SHADOW_OUT} />
      </div>
    </div>
  );
}

function PatternDiagram() {
  return (
    <div className="wf-diagram wf-diagram-compact">
      {PATTERN_STAGES.map((s, i) => (
        <div key={s.key} className="wf-row">
          <WfBox s={s} />
          {i < PATTERN_STAGES.length - 1 && <div className="wf-arrow-down" />}
        </div>
      ))}
    </div>
  );
}

function WfBox({ s }: { s: WfStage }) {
  return (
    <div className={`wf-box wf-${s.kind}`}>
      <strong>{s.title}</strong>
      <span>{s.detail}</span>
    </div>
  );
}

function WorkflowDiagram() {
  return (
    <div className="wf-diagram">
      <div className="wf-row">
        <WfBox s={CHALLENGE} />
        <div className="wf-arrow-down" />
      </div>
      {LINEAR.map((s) => (
        <div key={s.key} className="wf-row">
          <WfBox s={s} />
          <div className="wf-arrow-down" />
        </div>
      ))}

      <div className="wf-branch-row">
        <div className="wf-branch">
          {CLASSICAL_BRANCH.map((s, i) => (
            <div key={s.key} className="wf-row">
              <WfBox s={s} />
              {i < CLASSICAL_BRANCH.length - 1 && <div className="wf-arrow-down" />}
            </div>
          ))}
          <div className="wf-filler" />
        </div>
        <div className="wf-branch">
          {QUANTUM_BRANCH.map((s, i) => (
            <div key={s.key} className="wf-row">
              <WfBox s={s} />
              {i < QUANTUM_BRANCH.length - 1 && <div className="wf-arrow-down" />}
            </div>
          ))}
        </div>
      </div>

      <div className="wf-converge" />

      {TAIL.map((s) => (
        <div key={s.key} className="wf-row">
          <WfBox s={s} />
          {s.key !== "dossier" && <div className="wf-arrow-down" />}
        </div>
      ))}
    </div>
  );
}

export default function AboutPage() {
  return (
    <div className="page about-page">
      <div className="hero-intro">
        <h1>About</h1>
        <p>
          SphoorthiQ takes radar images of vehicles from the MSTAR dataset and turns segmentation into a Markov
          Random Field. We rewrite that as a QUBO, then run an exact classical solver and QAOA (through Classiq's
          Qmod) on the same objective, so the two can be compared fairly. Every figure on this page comes from runs performed for this project. Test cases generated for the project, or reproduced from the THALES brief, are labelled as such.
        </p>
      </div>

      <nav className="about-nav">
        <a href="#workflow">Workflow</a>
        <a href="#classical">Classical</a>
        <a href="#verification">QUBO Verification</a>
        <a href="#quantum">Quantum</a>
        <a href="#scaling">Scaling</a>
        <a href="#robustness">Robustness</a>
        <a href="#synthetic-validation">Synthetic validation</a>
        <a href="#real-masks">Real-mask results</a>
        <a href="#beyond-binary">Beyond binary</a>
        <a href="#dossier">Dossier</a>
      </nav>

      <section id="workflow">
        <h3>Workflow</h3>
        <p className="hint">
          One objective, two solvers. Every stage shown is implemented in this application and runs on MSTAR chips.
        </p>
        <WorkflowDiagram />
      </section>

      <section id="classical">
        <h3>Classical</h3>
        <p>
          Each pixel gets a cost for each label, based on its brightness in decibels relative to the chip's own
          background. A small number of constants determine where a pixel begins to count as target and how bright target and shadow are expected to be. We set them on the MSTAR TRAIN chips and score on the TEST chips, so the results
          below are on chips the constants never saw. There is no trained classifier. Neighboring pixels (up, down,
          left, right) also get a smoothness cost that is lower across strong edges. This pairwise cost is a Potts penalty,
          which makes the whole energy submodular. That is what allows an exact answer through an s-t min-cut (Greig,
          Porteous &amp; Seheult 1989), at any image size and with no qubits involved.
        </p>
        <p>
          The THALES brief names a MATLAB genetic algorithm as its baseline. MATLAB is not part of this project, so we implemented a genetic algorithm in Python (it is not MATLAB's ga) and ran it on the same energy with the same evaluation budget as QAOA. The results are in Scaling. For two labels graph cut is exact, so it remains the reference that the quantum result is compared against.
        </p>
      </section>

      <section id="verification">
        <h3>QUBO Verification</h3>
        <p>
          We checked the QUBO from <code>build_qubo</code> in <code>utils/mrf/energy.py</code> against the MRF energy
          on every one of the <strong>65,536 possible labelings</strong> of a 4x4 patch:
        </p>
        <table>
          <tbody>
            <tr>
              <td>states checked</td>
              <td>65,536 (2<sup>16</sup>)</td>
            </tr>
            <tr>
              <td>max |energy(x) - qubo_energy(x)| over every x</td>
              <td>2.13 &times; 10<sup>-14</sup> (rounding error, effectively 0)</td>
            </tr>
            <tr>
              <td>graph cut matches the true brute-force optimum</td>
              <td>true</td>
            </tr>
          </tbody>
        </table>
        <p className="hint">
          The result is saved in <code>datasets/reports/11_mrf_qubo_verification.json</code>. To rerun it, use{" "}
          <code>exhaustive_equivalence_check</code> in <code>utils/mrf/verification.py</code> or{" "}
          <code>tests/test_mrf_energy.py</code>. The check runs on a synthetic patch, so it holds for MSTAR chips too,
          since they go through the same <code>utils/mrf/energy.py</code>.
        </p>
      </section>

      <section id="quantum">
        <h3>Quantum</h3>
        <p>
          We take that same QUBO, convert it to an Ising Hamiltonian, and hand it to QAOA on Classiq. The circuit
          is built from Classiq's own <code>qaoa_init</code> and <code>qaoa_layer</code> functions, then
          synthesized and run on their platform. A classical <code>scipy.optimize</code> loop (COBYLA) sits on
          top and adjusts the circuit is angles. Every sampled bitstring is scored against the real QUBO energy,
          which is the same number the classical solver is judged on, so the two results are directly
          comparable.
        </p>

        <h4>What happens when a run is started</h4>
        <ol>
          <li>The app reads the chosen region of the real MSTAR chip and builds the energy for it.</li>
          <li>The classical graph cut solves it exactly. This takes milliseconds.</li>
          <li>The energy is converted to a QUBO and then an Ising Hamiltonian, and Classiq synthesizes the circuit.</li>
          <li>
            The optimizer takes up to 15 steps (the default). Each step sends one 512-shot job to Classiq, and the
            progress streams back to the Results panel as it goes.
          </li>
          <li>A final 2,048-shot sample at the tuned settings picks the best bitstring seen, which becomes the mask.</li>
          <li>The quantum result is shown next to the classical one, and the run is added to the job history.</li>
        </ol>
        <p className="hint">
          Each optimizer step is its own Classiq job, which is why a quantum run takes about a minute. You can
          change the layers, steps and shots under "Show advanced options" on the main page.
        </p>

        <h4>Maximum tile size</h4>
        <p>
          The qubit count depends on the number of classes. With two classes (background and target) each pixel
          needs one qubit, so the roughly 25-qubit limit of the Classiq simulators allows a 5x5 tile. With three
          classes (adding the shadow) each pixel needs three qubits, plus a penalty that forces exactly one class
          per pixel, so the limit drops to a 2x2 tile (12 qubits). A 3x3 tile would need 27.
        </p>

        <h4>Findings</h4>
        <p>
          We tested this on real MSTAR regions, running on Classiq's simulator. At 9 and 16 qubits, QAOA found the{" "}
          <em>exact</em> classical optimum on both test chips, and at every noise level we tried at 9 qubits. At 9
          qubits that shows the pipeline is correct and nothing more, because there are only 512 possible labelings and
          random sampling with the same budget finds the optimum too. At 16 qubits random sampling rarely does, so
          that result is better than chance, but it is one run per chip. At 25 qubits, with the same budget, QAOA approached but did not reach the optimum on either chip. The figures are in the Scaling section. QAOA is stochastic, so repeating the same tile can return a slightly different energy.
        </p>
      </section>

      <section id="scaling">
        <h3>Scaling</h3>
        <p>
          We ran two MSTAR test chips (T72 #132 and BMP2 #9563) at three qubit sizes. Each tile sits on the
          vehicle's edge, where about half of it is target according to the SARBake mask, so that the solver must decide between target and background. We compare energy and also score each result against the mask. Classiq's simulators on our
          account stop at about 28 to 35 qubits, so a 6x6 tile (36 qubits) was not feasible.
        </p>
        <table>
          <thead>
            <tr>
              <th>chip</th>
              <th>size</th>
              <th>qubits</th>
              <th>classical energy</th>
              <th>quantum energy</th>
              <th>random sampling, best (mean of 30)</th>
              <th>genetic algorithm, best (mean of 30)</th>
              <th>classical / quantum IoU</th>
              <th>QAOA duration</th>
            </tr>
          </thead>
          <tbody>
            {SWEEP.map((r, i) => (
              <tr key={i}>
                <td>{r.chip}</td>
                <td>
                  {r.size}x{r.size}
                </td>
                <td>{r.qubits}</td>
                <td>{r.classicalE}</td>
                <td className={r.quantumE === r.classicalE ? "mrf-metric-better" : undefined}>{r.quantumE}</td>
                        <td>{r.randomE}</td>
                <td>{r.gaE}</td>
                        <td>
                          {r.classicalIoU.toFixed(2)} / {r.quantumIoU.toFixed(2)}
                        </td>
                        <td>{r.quantumDuration}s</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="hint">
                  At 9 and 16 qubits QAOA reached the classical optimum on both chips. At 25 qubits, with the same
                  optimizer budget, it fell short on both (14.3462 against 13.8564, and 13.2295 against 12.1758), and its
                  IoU against the mask is a little lower there. The random-sampling and genetic-algorithm columns provide context: the genetic algorithm gets the same number of evaluations as QAOA has samples, and it reached the optimum on 27 to 30 of 30 seeds on every tile except the 25-qubit BMP2 tile (18 of 30). At 25 qubits its average energy (13.9659 and 12.3871) is below QAOA's (14.3462 and 13.2295). Random
                  sampling with the same number of samples finds the optimum on every seed at 9 qubits, so that match
                  shows the pipeline works and nothing more. At 16 qubits it found it on only 6 and 3 of 30 seeds, and at
                  25 qubits on none, and QAOA's energy was below the random average in all four of those rows. The IoU column is for these two chips only; averaged over 120 chips, edge tiles of this size score lower (see Real-mask results). Each QAOA value is a single run and QAOA is random, so repeat runs can differ. We do not claim a quantum advantage:
                  the classical solver is exact and takes milliseconds.
                </p>
        <p>
          Graph cut has no qubit limit, so it continues beyond the point at which QAOA is limited. We ran it on 8x8 edge patches (64 pixels, four
          times QAOA's ceiling) on the same two chips and got energies of 34.1487 (T72 #132) and 27.4969 (BMP2 #9563),
          in about 3 ms and 2 ms, with IoU against the mask of 0.71 and 0.78. The classical solver scales without difficulty, whereas the quantum solver is currently limited by qubit count.
        </p>
      </section>

      <section id="robustness">
        <h3>Robustness</h3>
        <p className="hint">
          This test is more controlled. We take one fixed region (T72 #132, the same 3x3 edge tile as the first row of Scaling, where QAOA
          matches the classical result, so qubit count is not a factor) and add increasing amounts of Gaussian noise to
          it. The noise comes from <code>inject_gaussian_noise</code> in <code>utils/mrf/energy.py</code> and is
          seeded, so it repeats exactly. It is added on top of the speckle already in the chip. It is also different
          from the "Low SNR" preset, which is one fixed window rather than a setting we vary.
        </p>
        <table>
          <thead>
            <tr>
              <th>noise sigma</th>
              <th>classical energy</th>
              <th>quantum energy</th>
              <th>QAOA matches classical</th>
              <th>classical / quantum IoU</th>
              <th>QAOA duration</th>
            </tr>
          </thead>
          <tbody>
            {NOISE_SWEEP.map((r, i) => (
              <tr key={i}>
                <td>{r.sigma}</td>
                <td>{r.classicalE}</td>
                <td className={r.quantumE === r.classicalE ? "mrf-metric-better" : undefined}>{r.quantumE}</td>
                <td>{r.quantumE === r.classicalE ? "yes" : "no"}</td>
                <td>
                  {r.classicalIoU.toFixed(2)} / {r.quantumIoU.toFixed(2)}
                </td>
                <td>{r.quantumDuration}s</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="hint">
          QAOA matched the classical optimum at every noise level, so at this size the match does not depend on
          clean data. Energies cannot be compared between rows, because each noise level is a different image. The
          scores against the mask go from 0.40 with no added noise to 0.62 at moderate noise and 0.50 at the highest
          level. On a 9-pixel tile a single pixel changes IoU substantially, so no trend is inferred. QAOA and the
          classical solver found the same labeling in every row, which is why their IoU is the same. To rerun the
          sweep, use <code>scripts/robustness_sweep.py</code> against a running copy of this app.
        </p>
      </section>

      <section id="synthetic-validation">
        <h3>Synthetic validation</h3>
        <p className="hint">
          These cases are not MSTAR chips. Each has a known correct answer by construction, so accuracy, IoU and Dice
          are exact here. The first case is THALES's own 5x5 example, taken from page 3 of their brief. The other three are small cases we generated with a known target
          region. You can run them from the Synthetic Validation tab in the app, through{" "}
          <code>POST /api/mstar/synthetic-validation</code>, or with <code>scripts/synthetic_validation_run.py</code>.
        </p>
        <table>
          <thead>
            <tr>
              <th>case</th>
              <th>size</th>
              <th>classical acc / IoU</th>
              <th>quantum acc / IoU</th>
              <th>classical energy</th>
              <th>quantum energy</th>
            </tr>
          </thead>
          <tbody>
            {SYNTHETIC_VALIDATION.map((r, i) => (
              <tr key={i}>
                <td>{r.name}</td>
                <td>
                  {r.size}x{r.size} ({r.qubits}q)
                </td>
                <td>
                  {(r.classicalAcc * 100).toFixed(0)}% / {r.classicalIoU.toFixed(2)}
                </td>
                <td className={r.quantumAcc === r.classicalAcc ? "mrf-metric-better" : undefined}>
                  {(r.quantumAcc * 100).toFixed(0)}% / {r.quantumIoU.toFixed(2)}
                </td>
                <td>{r.classicalE}</td>
                <td>{r.quantumE}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="hint">
          The classical solver reproduces THALES's answer exactly on every case, including their own example (100%
          accuracy, IoU and Dice). QAOA is exact at 9 and 16 qubits. At 25 qubits, the size of THALES's example, its accuracy drops to between 92% and 96% across four runs (the table shows the first). That fits the Scaling results on real chips, where QAOA is exact up to 16 qubits
          and falls short at 25.
        </p>
      </section>

      <section id="real-masks">
        <h3>Real-mask results</h3>
        <p>
          MSTAR's public release has no per-pixel masks, but the SARBake overlay (Malmgren-Hansen and
          Nobel-J&oslash;rgensen, CC BY 4.0) provides one for every chip we use. Each mask labels pixels as
          background, target or shadow, and is generated from a 3D CAD model of the vehicle at the chip's viewing
          angle. It is a good reference but not hand-drawn ground truth. It covers the bright vehicle return, not
          every pixel a person might outline, and it includes thin antenna slivers. Scores below should be read
          with that in mind.
        </p>
        <p>
          We scored the classical solvers on 120 random MSTAR TEST chips (40 each of BMP2, BTR70 and T72, 15&deg;
          depression), using a 48x48 region around each vehicle. The decibel setup was tuned on the TRAIN chips
          only. The older setup used the chip's log-scaled, min-max normalized magnitude directly as the target
          probability.
        </p>
        <table>
          <thead>
            <tr>
              <th>score on 120 TEST chips</th>
              <th>older setup</th>
              <th>decibel setup</th>
            </tr>
          </thead>
          <tbody>
            {REAL_MASK_ROWS.map((r) => (
              <tr key={r.label}>
                <td>{r.label}</td>
                <td>{r.before}</td>
                <td className="mrf-metric-better">{r.after}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="hint">
          The older setup marked about 13 target pixels where the mask has about 320, so nearly all pixels were labelled background, which is why it was replaced. Per vehicle, binary IoU with the decibel setup is 0.55
          for BMP2, 0.50 for BTR70 and 0.62 for T72. These are classical results only. On the small edge tiles the quantum solver can run on, scores are much lower. Each tile is placed where the mask is about half target, which is the hardest case. Averaged over the same 120 chips, target IoU is 0.12 on 2x2 tiles, 0.18 on 4x4 and 0.34 on 8x8. For three classes on these tiles, pixel accuracy is 0.32 to 0.36, below the 0.50 you would get by always choosing the most common class. A few speckled pixels give little evidence for separating target, shadow and background. The two chips in the Scaling table score higher than these averages, so their IoU should not be read as typical. Reproduce them with{" "}
          <code>scripts/sarbake_eval.py</code>. In the app, the Images and 3-Class Segmentation tabs now show the
          mask next to each solver's result, with scores in the Metrics tab.
        </p>
      </section>

      <section id="beyond-binary">
        <h3>Beyond binary</h3>
        <p className="hint">
          The THALES brief lists "beyond binary" ideas as bullets with no detail: edge detection, 3+ classes and
          pattern detection. We built all three, and they run in this app. Each gets its own small diagram below,
          because each branches off the main pipeline at a different point. We also built the brief's two optional edge
          refinements: telling object edges from shadow edges, and marking one pixel per boundary instead of two.
          Single-pixel edges are in the Images tab. Object-versus-shadow edges are in the 3-Class Segmentation tab,
          since they need a three-class run first. Edge maps are worked out from the labels on screen, so they need no
          separate solver.
        </p>

        <h4>3+ classes: shadow segmentation (Q=3)</h4>
        <p>
          Here the vehicle's shadow becomes a third label, using the same region as the two-class run above but with
          a multi-label MRF. The exactness guarantee does not carry over. Graph cut needs exactly two labels, so for
          three the classical solver falls back to alpha-expansion, which is approximate. The one-hot QUBO also needs
          three qubits per pixel instead of one, so the quantum path is limited to a 2x2 tile instead of 5x5. Scored
                    against the SARBake masks on 120 TEST chips (see Real-mask results), the three-class classical result
                    reaches a mean IoU of 0.66, with 0.72 for the shadow.
        </p>
        <ShadowDiagram />

        <p className="hint">
          Our results differ from the brief's printed examples in two places. In its three-class example the shadow
          has a middle brightness (50, between the target's 100 and the background's 10), but we model the shadow as
          dark. On that example the solver agrees with the printed labels on 21 of 25 pixels (84%) when its background
          reference is 0.2 or lower, but it labels all four shadow pixels as background. With a reference of 0.3 or
          higher it agrees on 11 of 25 (44%), because it then labels ten pixels as shadow. In its edge example only the
          object-side pixel of each boundary is marked. Our both-sides map differs from that matrix on 5 pixels, and our
          single-pixel map (which keeps the top or left pixel) on 1.
        </p>

        <h4>Pattern detection: target-type recognition</h4>
        <p>
          The brief lists "pattern detection" and says nothing more, so there was no equation or example to build
          against. We chose the reading its MSTAR references point to (Mahalanobis et al.'s correlation-filter
          recognition): given a chip, work out which vehicle it is. That is our own interpretation, and we say so.
          Accuracy is 51.79%, well above the 33% one would get by guessing among three classes and well below the roughly
          95% a modern CNN reaches.
        </p>
        <PatternDiagram />
      </section>

      <section id="dossier">
        <h3>Dossier</h3>
        <ul className="about-dossier-list">
          <li>
            <strong>Problem framing:</strong> SAR vehicle segmentation on MSTAR chips, treated as MAP inference over a binary MRF (the THALES brief).
          </li>
          <li>
            <strong>Objective:</strong> minimize unary plus pairwise energy. The classical and quantum solvers use the identical objective.
          </li>
          <li>
            <strong>Variables:</strong> one binary label per pixel, which means one qubit per pixel on the quantum path.
          </li>
          <li>
            <strong>Constraints:</strong> each pixel gets exactly one label. With two labels that holds automatically. With
            three, a one-hot penalty term enforces it, because otherwise the QUBO could favor two labels at once.
            Neighbors are 4-connected, not 8. The qubit ceiling of about 25 on the available Classiq simulators limits
            the tile to 5x5 for two labels and 2x2 for three. QAOA's outer loop is capped too (COBYLA,{" "}
            <code>qaoa_max_iterations</code>, 15 by default), so it is a limited search, not an exhaustive one.
          </li>
          <li>
            <strong>Assumptions:</strong> MSTAR's public release has no per-pixel ground truth. We score real chips against
          the SARBake masks, which come from CAD models and are a good reference, not hand-drawn truth. We assume a
          chip's brightness in decibels, against its own background, is enough evidence for the unary term, with no
          trained classifier, as THALES's own example does ("We consider here only the intensity to label the pixels"). The pairwise term assumes neighbors tend to share a label, a standard
            Potts prior that is not fitted to MSTAR statistics. The noise test assumes independent Gaussian noise is a
            fair stress test, added on top of the speckle already in the chip.
          </li>
          <li>
            <strong>Candidate formulations:</strong> for three labels we considered one-hot and binary (log&#8322;Q) qubit
            encodings. We picked one-hot even though it triples the qubits per pixel, because a binary encoding needs
            extra penalty terms to rule out invalid codes, and we wanted results we could verify. THALES's reference [7]
            (Otgonbaatar &amp; Datcu) uses quantum annealing, but we chose QAOA because Classiq/Qmod is a gate-model
            platform, not an annealer. For the three-label classical baseline we used alpha-expansion rather than
            alpha-beta swap, the standard Boykov-Veksler-Zabih choice for Potts-style costs.
          </li>
          <li>
            <strong>Label count:</strong> two labels (background and target) is THALES's core task, and the only case
            where the graph-cut exactness claim holds, since the s-t construction does not extend past two labels. Three
            labels (adding the shadow) is implemented too (see Beyond binary above), but it gives up exactness for
            approximate alpha-expansion, and each extra label adds qubits per pixel against the same ceiling.
          </li>
          <li>
            <strong>Formulation:</strong> MRF -&gt; QUBO (checked exact) -&gt; Ising Hamiltonian -&gt; QAOA.
          </li>
          <li>
            <strong>Quantum suitability:</strong> correct at small scale: exact at 9 and 16 qubits on both chips tested. At
          25 qubits it falls short, limited by the qubit budget and the optimizer, and the available simulators are limited to around 25 to 28 qubits.
          </li>
          <li>
            <strong>Execution pathway:</strong> Classiq Qmod, synthesis, ExecutionSession, with a scipy optimizer loop on top.
          </li>
          <li>
            <strong>Scalability:</strong> a full 128x128 chip would have to be split into many small regions rather than run
            as one circuit. Nothing here claims otherwise.
          </li>
          <li>
            <strong>Integration:</strong> <code>GET /api/mstar/chips</code>, <code>POST /api/mstar</code> (and its
            live-streaming twin <code>/api/mstar/stream</code>), <code>POST /api/mstar/shadow</code>,{" "}
            <code>POST /api/mstar/pattern</code> with <code>GET /api/mstar/pattern/evaluation</code>, and{" "}
            <code>POST /api/mstar/synthetic-validation</code>. All of them are part of this running app, not just a notebook.
          </li>
          <li>
            <strong>Next steps:</strong> tile a whole 128x128 chip with consistency between neighboring tiles, so one run
            covers the full chip instead of one region. Run on real quantum hardware (everything here used a Classiq
            simulator: <code>simulator</code>, <code>nvidia_simulator</code> or <code>dgx_simulator</code>). Go beyond
            three labels if a real need appears. <code>utils/mrf/multilabel_energy.py</code> already loops over a generic
            label count, so that would be a small extension.
          </li>
        </ul>
      </section>
    </div>
  );
}
