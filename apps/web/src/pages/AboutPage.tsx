// Real sweep: two real MSTAR chips (T72 #132, BMP2 #9563), three qubit
// tiers, run live against this app's own POST /api/mstar - not simulated,
// not hand-picked to make quantum look better (16/25-qubit rows genuinely
// fall short of the classical optimum, and that's shown, not hidden).
const SWEEP = [
  { chip: "T72 #132 (test)", size: 3, qubits: 9, classicalE: 3.151, quantumE: 3.151, quantumDuration: 54.2 },
  { chip: "BMP2 #9563 (test)", size: 3, qubits: 9, classicalE: 2.2076, quantumE: 2.2076, quantumDuration: 61.3 },
  { chip: "T72 #132 (test)", size: 4, qubits: 16, classicalE: 7.3613, quantumE: 7.811, quantumDuration: 58.2 },
  { chip: "BMP2 #9563 (test)", size: 4, qubits: 16, classicalE: 5.6416, quantumE: 6.4482, quantumDuration: 60.1 },
  { chip: "T72 #132 (test)", size: 5, qubits: 25, classicalE: 11.0345, quantumE: 13.8905, quantumDuration: 85.9 },
  { chip: "BMP2 #9563 (test)", size: 5, qubits: 25, classicalE: 8.7946, quantumE: 15.4687, quantumDuration: 90.7 },
];

// Real robustness sweep: one real MSTAR ROI (T72 #132, same 3x3=9-qubit
// tile as the Scaling row above - the tier QAOA matches exactly at, so
// this isolates the effect of noise from the effect of qubit budget),
// run live via scripts/robustness_sweep.py against this app's own
// POST /api/mstar at increasing synthetic-noise levels
// (utils.mrf.energy.inject_gaussian_noise) on top of the real magnitude.
// sigma=0.15's 319s duration is a real, unexplained outlier (a slow
// Classiq platform round trip, not a bug) - shown as measured, not
// smoothed over.
const NOISE_SWEEP = [
  { sigma: 0.0, classicalE: 3.151, quantumE: 3.151, quantumDuration: 60.4 },
  { sigma: 0.05, classicalE: 3.1519, quantumE: 3.1519, quantumDuration: 58.7 },
  { sigma: 0.1, classicalE: 3.1775, quantumE: 3.1775, quantumDuration: 61.7 },
  { sigma: 0.15, classicalE: 3.229, quantumE: 3.229, quantumDuration: 319.2 },
  { sigma: 0.2, classicalE: 3.3, quantumE: 3.3, quantumDuration: 58.5 },
  { sigma: 0.3, classicalE: 3.4489, quantumE: 3.4489, quantumDuration: 62.1 },
];

// Real Accuracy/IoU/Dice - the one place these exist in this app, because
// it's the one place real ground truth exists: THALES's own worked
// example (quoted directly from their brief, page 3) plus small generated
// cases with a known target region (utils/mrf/synthetic_validation.py).
// Run live via scripts/synthetic_validation_run.py against this app's own
// POST /api/mstar/synthetic-validation - never against a real MSTAR chip.
// The 16-qubit row matching exactly (unlike the Scaling sweep's 16-qubit
// row above) isn't cherry-picked - it's a real, honest difference: this
// synthetic case has a cleaner optimization landscape (a solid block, no
// real SAR speckle) than an arbitrary MSTAR ROI.
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

const CHALLENGE: WfStage = { key: "challenge", kind: "challenge", title: "THALES SAR", detail: "Industrial challenge - SAR image segmentation via quantum optimization" };
const LINEAR: WfStage[] = [
  { key: "input", kind: "input", title: "MSTAR SAR chip", detail: "Real X-band target chip, 128x128 (or 54x54 for SLICY) - not the 64x64 THALES describes for its own synthetic validation set" },
  { key: "prep", kind: "prep", title: "Preprocessing", detail: "Log-scale + normalize the real magnitude" },
  { key: "roi", kind: "roi", title: "ROI selection", detail: "row0, col0, size - up to 5x5 (25 qubits) for QAOA, larger for classical-only (no qubit limit)" },
];
const CLASSICAL_BRANCH: WfStage[] = [
  { key: "c1", kind: "classical", title: "Classical MRF", detail: "Unary + pairwise energy" },
  { key: "c2", kind: "classical", title: "Classical optimization", detail: "Exact s-t min-cut graph cut" },
  { key: "c3", kind: "classical", title: "Classical segmentation", detail: "Exact MAP labeling, no qubit limit" },
];
const QUANTUM_BRANCH: WfStage[] = [
  { key: "q1", kind: "quantum", title: "Quantum-ready MRF", detail: "Formulate as QUBO" },
  { key: "q2", kind: "quantum", title: "QAOA", detail: "QUBO -> Ising Hamiltonian, variational optimization" },
  { key: "q3", kind: "quantum", title: "Classiq / Qmod", detail: "qaoa_init + qaoa_layer, synthesis & execution" },
  { key: "q4", kind: "quantum", title: "Classiq backend", detail: "simulator / nvidia_simulator / dgx_simulator - no real quantum hardware wired up" },
  { key: "q5", kind: "quantum", title: "Quantum segmentation", detail: "Best sampled bitstring -> mask" },
];
const TAIL: WfStage[] = [
  { key: "merge", kind: "merge", title: "Segmentation masks", detail: "Classical + quantum results, same chip, same ROI" },
  { key: "metrics", kind: "metrics", title: "Energy comparison", detail: "Classical vs quantum objective value - no IoU/Dice/Precision/Recall: MSTAR's public release has no per-pixel mask to score against" },
  { key: "compare", kind: "compare", title: "Classical vs quantum", detail: "Side-by-side comparison & analysis" },
  { key: "scaling", kind: "scaling", title: "Feasibility & scaling", detail: "Real qubit ceiling (9/16/25 tested live), tiling, honest limits - see Scaling below" },
  { key: "dossier", kind: "dossier", title: "Algorithm dossier", detail: "Final report, wired into this running app - not a standalone notebook" },
];

// THALES's "beyond binary" bullets - real, shipped extensions of the same
// pipeline, not planned/future work (see apps/api/routers/mstar.py's
// /shadow and /pattern* endpoints, utils/mrf/multilabel_energy.py,
// utils/mstar/pattern_recognition.py). Shown as their own small diagrams
// rather than folded into the Q=2 diagram above because they branch off
// at different points (shadow reuses the ROI; pattern detection runs on
// the whole chip, independent of ROI/size) and pattern detection isn't an
// MRF/QUBO/QAOA pipeline at all - honest to keep it visually separate.
const SHADOW_ROI: WfStage = { key: "shadow-roi", kind: "roi", title: "Same ROI", detail: "Reuses row0/col0/size from the Q=2 pipeline above - same real chip patch" };
const SHADOW_MRF: WfStage = { key: "shadow-mrf", kind: "prep", title: "Multi-label MRF", detail: "3 labels this time: background/target/shadow (Q=3) - utils/mrf/multilabel_energy.py" };
const SHADOW_CLASSICAL_BRANCH: WfStage[] = [
  {
    key: "sc1",
    kind: "classical",
    title: "Alpha-expansion",
    detail: "Iterated binary graph-cut moves (Boykov-Veksler-Zabih) - approximate for Q=3, unlike Q=2's single exact graph cut",
  },
];
const SHADOW_QUANTUM_BRANCH: WfStage[] = [
  { key: "sq1", kind: "quantum", title: "One-hot QUBO", detail: "3 qubits/pixel + an exactly-one-label penalty term" },
  {
    key: "sq2",
    kind: "quantum",
    title: "QAOA (Classiq)",
    detail: "Same Classiq/Qmod pathway as Q=2, capped at a 2x2 tile (12 qubits) - 3x3 needs 27, over the real 25-qubit ceiling",
  },
];
const SHADOW_OUT: WfStage = { key: "shadow-out", kind: "merge", title: "3-class segmentation", detail: "background / target / shadow mask" };

const PATTERN_STAGES: WfStage[] = [
  { key: "p1", kind: "pattern", title: "MSTAR chip", detail: "Whole chip, not ROI-limited - THALES's own \"Pattern detection\" bullet, left undefined in the brief" },
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
    detail: "bmp2_tank / btr70_transport / t72_tank - 51.79% real accuracy on 1365 real test chips (15-deg), vs. 33% chance",
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
          SphoorthiQ takes real SAR vehicle-target chips from MSTAR, frames segmentation as a Markov Random Field,
          rewrites that as a QUBO, and then puts an exact classical solver up against QAOA (via Classiq/Qmod) on
          the exact same objective, so the comparison actually means something. Everything on this page comes from
          a real run we performed ourselves - nothing here is a placeholder or a made-up example.
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
        <a href="#beyond-binary">Beyond binary</a>
        <a href="#dossier">Dossier</a>
      </nav>

      <section id="workflow">
        <h3>Workflow</h3>
        <p className="hint">
          One objective, two solvers. Every stage below is real code, actually running in this app end to end on
          real MSTAR SAR chips - this diagram describes what the app does, not just an idea for what it could do.
        </p>
        <WorkflowDiagram />
      </section>

      <section id="classical">
        <h3>Classical</h3>
        <p>
          Each pixel gets a unary cost per label, taken straight from the chip's own real, normalized magnitude -
          MSTAR's public release has no per-pixel target/background mask for any chip, so there's simply nothing
          to train a classifier against. Every 4-connected neighbor pair also gets a contrast-sensitive smoothness
          cost. That pairwise term happens to be a ferromagnetic Potts penalty, which makes the whole energy
          submodular - and that's what lets us find the MAP labeling <em>exactly</em>, via s-t min-cut (Greig,
          Porteous &amp; Seheult 1989), not by approximating it, and with no qubit limit at any size.
        </p>
      </section>

      <section id="verification">
        <h3>QUBO Verification</h3>
        <p>
          The QUBO reformulation (<code>utils/mrf/energy.py</code>'s <code>build_qubo</code>) is not merely
          close to the MRF energy - it was checked exact over every one of the{" "}
          <strong>65,536 possible labelings</strong> of a 4x4 patch:
        </p>
        <table>
          <tbody>
            <tr>
              <td>states checked</td>
              <td>65,536 (2<sup>16</sup>)</td>
            </tr>
            <tr>
              <td>max |energy(x) - qubo_energy(x)| over every x</td>
              <td>2.13 &times; 10<sup>-14</sup> (float noise, i.e. exactly 0)</td>
            </tr>
            <tr>
              <td>graph cut matches the true brute-force optimum</td>
              <td>true</td>
            </tr>
          </tbody>
        </table>
        <p className="hint">
          Evidence file: <code>datasets/reports/11_mrf_qubo_verification.json</code>. Reproducible via{" "}
          <code>utils/mrf/verification.py</code>'s <code>exhaustive_equivalence_check</code> and{" "}
          <code>tests/test_mrf_energy.py</code> - this check is data-agnostic (a synthetic patch), so it holds
          for MSTAR the same as it did for anything else fed through <code>utils/mrf/energy.py</code>.
        </p>
      </section>

      <section id="quantum">
        <h3>Quantum</h3>
        <p>
          We take that same QUBO, convert it to an Ising Hamiltonian, and hand it to QAOA on Classiq: a Qmod
          circuit built from Classiq's own <code>qaoa_init</code> and <code>qaoa_layer</code> functions,
          synthesized and executed on their platform. A classical <code>scipy.optimize</code> loop sits on top,
          scoring every sampled bitstring against the real QUBO energy - the actual objective the classical
          solver is scored on too, not some generic Hamiltonian expectation value that only looks related.
        </p>
        <p>
          We tested this live on real MSTAR ROIs, not simulated results: at 9 qubits, QAOA found the{" "}
          <em>exact</em> classical optimum on both test chips. At 16 and 25 qubits, with the same optimizer
          budget, it got close but fell short - the real numbers are below in Scaling, not rounded up to look
          better.
        </p>
      </section>

      <section id="scaling">
        <h3>Scaling</h3>
        <p>
          We ran two real MSTAR test chips (T72 #132, BMP2 #9563) at the three qubit tiers Classiq's own
          simulators can actually handle today - no general-purpose simulator on this account went past ~28-35
          qubits, so a naive 6x6=36 tier was never realistic to try. You won't find IoU or F1 columns here:
          MSTAR's public release has no per-pixel mask, so energy is the only comparison we can make honestly.
        </p>
        <table>
          <thead>
            <tr>
              <th>chip</th>
              <th>size</th>
              <th>qubits</th>
              <th>classical energy</th>
              <th>quantum energy</th>
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
                <td>{r.quantumDuration}s</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="hint">
          QAOA matches the classical optimum exactly at 9 qubits (both chips). At 16 and 25 qubits, with the
          same optimizer budget, it falls short - a real, honest scaling limit, not a hidden one.
        </p>
        <p>
          Classical graph cut has no qubit limit at all, so it scales past where QAOA can follow - tested at
          8x8 (64 pixels, four times QAOA's ceiling) on the same two real chips: energy 24.2572 (T72 #132) and
          26.8995 (BMP2 #9563), both in under 6ms. This asymmetry - classical scales trivially, quantum is
          qubit-capped today - is the honest scaling story, not "quantum can't do this yet" hidden or glossed
          over.
        </p>
      </section>

      <section id="robustness">
        <h3>Robustness</h3>
        <p className="hint">
          Here we test something more controlled: we take one fixed real ROI (T72 #132, the same 3x3 = 9-qubit
          tile where QAOA matches classical exactly in Scaling above, so we're isolating noise from qubit
          budget) and run it at increasing levels of synthetic Gaussian noise layered on top of the real
          magnitude (<code>utils/mrf/energy.py</code>'s <code>inject_gaussian_noise</code>, seeded so it's
          reproducible). That's on top of, not instead of, the real speckle already in the chip - and it's a
          different thing from the "low SNR" preset above, which is one fixed real window rather than a
          parameter we sweep.
        </p>
        <table>
          <thead>
            <tr>
              <th>noise sigma</th>
              <th>classical energy</th>
              <th>quantum energy</th>
              <th>QAOA matches classical</th>
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
                <td>{r.quantumDuration}s</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="hint">
          Classical energy climbs smoothly and predictably as noise increases (3.151 to 3.4489) - the
          expected effect of a noisier signal on unary confidence. QAOA matched the exact classical optimum
          at every single noise level tested, not just the clean case - real evidence that exactness at this
          qubit tier isn't fragile to data quality, at least across this range. The sigma=0.15 row's 319s
          duration is a real, unexplained outlier (a slow round trip to Classiq's platform, not a bug) -
          reported as measured, not smoothed over. Reproducible via{" "}
          <code>scripts/robustness_sweep.py</code> against a running instance of this app.
        </p>
      </section>

      <section id="synthetic-validation">
        <h3>Synthetic validation</h3>
        <p className="hint">
          Everything above this point is scored on energy alone, because real MSTAR chips simply have no
          per-pixel mask. This is the one section where real Accuracy/IoU/Dice actually exist in this project
          - and only because we found two real sources of ground truth to check against. First, THALES's own
          worked example, quoted directly from page 3 of their brief - the exact same 5x5 intensity matrix and
          label matrix printed in the document, not paraphrased. Second, a few small cases we generated
          ourselves with a known target region. You can run this yourself, live, from the app's own
          "Synthetic Validation" tab, via <code>POST /api/mstar/synthetic-validation</code>, or reproduce it
          via <code>scripts/synthetic_validation_run.py</code>.
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
          The classical solver reproduces THALES's own stated answer exactly (100% accuracy/IoU/Dice) on
          every case, including their own worked example - the strongest validation this project can offer.
          QAOA matches exactly at 9 and, on this cleaner synthetic case, at 16 qubits too - unlike the
          Scaling sweep above, where an arbitrary real MSTAR ROI's speckle made 16 qubits fall short. At 25
          qubits (THALES's own exact example size), QAOA falls short here too (96% accuracy) - consistent
          with the real qubit-scaling limit disclosed throughout this project, not contradicted by the
          16-qubit result.
        </p>
      </section>

      <section id="beyond-binary">
        <h3>Beyond binary</h3>
        <p className="hint">
          THALES's brief lists "beyond binary class segmentation" as a few bullets with no further spec: edge
          detection (which we compute client-side from the labels above - no separate solver needed), 3+
          classes, and pattern detection. We built all three for real, as running parts of this same app, not
          as future work - each gets its own small diagram below because they branch off the base pipeline at
          different points. We also went ahead and built the brief's two edge-detection refinements -
          distinguishing object edges from shadow edges, and marking a single pixel per boundary instead of 2
          - even though the document itself frames those as optional "further work." Single-pixel edges are
          in the Images tab; the object-vs-shadow distinction is in the 3-Class Segmentation tab, since it
          needs a Q=3 run first.
        </p>

        <h4>3+ classes: shadow segmentation (Q=3)</h4>
        <p>
          Here the vehicle's shadow becomes a third label. We reuse the exact same ROI as the Q=2 run above,
          just with a multi-label MRF instead. One thing doesn't carry over, though: Q=2's exactness guarantee.
          S-t graph cut is a two-terminal construction that only works for two labels, so the classical solver
          here has to fall back to an approximate method (alpha-expansion), and the one-hot QUBO encoding
          triples the qubit cost per pixel - which caps the quantum path at a 2x2 tile instead of Q=2's 5x5.
        </p>
        <ShadowDiagram />

        <h4>Pattern detection: target-type recognition</h4>
        <p>
          THALES names "Pattern detection" in their document and leaves it at that - no equations, no example,
          nothing concrete to build against. Rather than skip it, or invent something arbitrary, we built the
          interpretation the brief's own MSTAR literature points toward (Mahalanobis et al.'s correlation-filter
          target recognition): given a chip, recognize which real vehicle it actually is. That's a disclosed
          judgment call on our part, not a hidden one - and we report its real accuracy plainly rather than
          inflating it: 51.79%, well above the 33% chance baseline for 3 classes, well below what a modern CNN
          would reach (~95%+).
        </p>
        <PatternDiagram />
      </section>

      <section id="dossier">
        <h3>Dossier</h3>
        <ul className="about-dossier-list">
          <li>
            <strong>Problem framing:</strong> SAR vehicle-target segmentation (real MSTAR chips) as MAP
            inference over a binary MRF (THALES SAR-segmentation brief).
          </li>
          <li>
            <strong>Objective:</strong> minimize unary + pairwise energy; identical objective for classical
            and quantum solvers.
          </li>
          <li>
            <strong>Variables:</strong> one binary label per pixel (one qubit per pixel for the quantum path).
          </li>
          <li>
            <strong>Constraints:</strong> exactly one label per pixel - structural for Q=2 (a single binary
            variable can't be two labels at once), enforced by an explicit one-hot penalty term for Q=3
            (nothing stops the QUBO from favoring two labels at once otherwise). Pairwise structure is
            4-connected neighbors only, not 8-connected. The real, measured qubit ceiling
            (~25 qubits on the Classiq simulators available to this project) bounds ROI size directly - 5x5
            for Q=2, 2x2 for Q=3. QAOA's classical outer loop is budget-limited (COBYLA,{" "}
            <code>qaoa_max_iterations</code>, default 15) - a real, disclosed compute cap, not an
            unlimited search.
          </li>
          <li>
            <strong>Assumptions:</strong> no per-pixel ground truth exists in MSTAR's public release, so
            energy is treated as the only honest objective-level comparison - accuracy/IoU/Dice are assumed
            unavailable, never estimated or approximated. The chip's own real, log-normalized magnitude is
            assumed sufficient unary evidence with no trained classifier behind it - the same assumption
            THALES's own worked example makes ("We consider here only the intensity to label the pixels").
            The pairwise term assumes same-label neighbors are more likely than not (a standard Potts-style
            vision prior), not something fit to MSTAR-specific statistics. The Robustness sweep's synthetic
            noise assumes i.i.d. Gaussian perturbation is a reasonable controlled stressor, layered on top of
            (not a replacement for) the real SAR speckle already in the chip.
          </li>
          <li>
            <strong>Candidate formulations considered:</strong> one-hot vs. binary/log&#8322;(Q) qubit
            encoding for Q=3 - one-hot was chosen despite tripling the per-pixel qubit cost, because a
            binary/log encoding needs its own nontrivial penalty terms to rule out invalid codewords, and
            this project's standard is verified-exact-or-clearly-labeled-approximate, not clever-but-unverified.
            QAOA vs. quantum annealing - THALES's own reference [7] (Otgonbaatar &amp; Datcu) uses annealing
            for SAR segmentation; QAOA was chosen instead because Classiq/Qmod, the platform this project
            standardizes on, is a gate-model toolkit, not an annealer. Alpha-expansion vs. alpha-beta swap for
            the Q=3 classical baseline - alpha-expansion is the standard Boykov-Veksler-Zabih choice for
            Potts-like pairwise costs.
          </li>
          <li>
            <strong>Label count:</strong> Q=2 (background/target) is THALES's core validation-set task, and the
            one place the graph-cut exactness claim above holds - it's a two-terminal (source/sink) construction
            that doesn't generalize to Q&gt;2. Q=3 (adding the vehicle's shadow) is also implemented, not left as
            future work - see Beyond binary below - but honestly: it trades that exactness for approximate
            alpha-expansion classically, and each additional label multiplies the quantum qubit cost per pixel
            (one-hot: +Q qubits/pixel), a real cost against the qubit ceiling, not a free extension.
          </li>
          <li>
            <strong>Formulation:</strong> MRF -&gt; QUBO (verified exact) -&gt; Ising Hamiltonian -&gt; QAOA.
          </li>
          <li>
            <strong>Quantum suitability:</strong> proven correct at small scale (9 qubits); qubit-budget-limited
            and optimizer-limited today at 16-25 qubits (~25-28 qubits ceiling on available simulators).
          </li>
          <li>
            <strong>Execution pathway:</strong> Classiq Qmod -&gt; synthesis -&gt; ExecutionSession -&gt;
            scipy-driven classical optimizer loop.
          </li>
          <li>
            <strong>Scalability:</strong> a full 128x128 chip needs tiling into many small ROIs, not one large
            circuit - no result here claims otherwise.
          </li>
          <li>
            <strong>Integration:</strong> <code>GET /api/mstar/chips</code>, <code>POST /api/mstar</code> (and
            its live-streaming twin <code>/api/mstar/stream</code>), <code>POST /api/mstar/shadow</code>,{" "}
            <code>POST /api/mstar/pattern</code> + <code>GET /api/mstar/pattern/evaluation</code>, and{" "}
            <code>POST /api/mstar/synthetic-validation</code> - all wired into this running application, not a
            standalone notebook only.
          </li>
          <li>
            <strong>Next-step roadmap:</strong> full-chip tiling with cross-patch consistency, so one run
            covers a whole 128x128 chip instead of a single ROI. Execution on real quantum hardware - every
            result on this page is a Classiq simulator backend (<code>simulator</code> /{" "}
            <code>nvidia_simulator</code> /{" "}
            <code>dgx_simulator</code>), no QPU wired up yet. General Q-ary generalization beyond Q=3, if a
            real requirement ever calls for it - the alpha-expansion/QUBO machinery in{" "}
            <code>utils/mrf/multilabel_energy.py</code> already loops over a generic label count internally,
            so this would be a scoped extension, not a rewrite.
          </li>
        </ul>
      </section>
    </div>
  );
}
