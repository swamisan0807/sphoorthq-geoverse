import { useEffect, useState } from "react";
import {
  api,
  type MstarChipInfo,
  type MstarResponse,
  type MstarShadowResponse,
  type PatternEvaluationResponse,
  type PatternResponse,
  type SyntheticValidationResponse,
} from "../api/client";

// Matches utils.qml.classiq_qaoa.MAX_QUBITS (one qubit per pixel) - a
// larger ROI still works for the classical-only baseline, which has no
// qubit budget, but QAOA needs size*size <= this.
const MAX_QAOA_SIZE = 5;
const MAX_ROI_SIZE = 12; // classical graph-cut has no qubit budget, but keep the tile small enough to stay a "patch", not a full re-tiling of the chip
type Tab = "images" | "metrics" | "logs" | "validation" | "shadow" | "pattern";

interface ChipPreviewFields {
  chip_preview_png: string;
  chip_height: number;
  chip_width: number;
  target_type: string;
  target_serial: string;
  row0: number;
  col0: number;
  size: number;
}

function ChipPreview({ result }: { result: ChipPreviewFields }) {
  const boxStyle = {
    left: `${(result.col0 / result.chip_width) * 100}%`,
    top: `${(result.row0 / result.chip_height) * 100}%`,
    width: `${(result.size / result.chip_width) * 100}%`,
    height: `${(result.size / result.chip_height) * 100}%`,
  };
  return (
    <div className="mrf-grid-block mrf-chip-block">
      <p className="mrf-grid-title">
        Input SAR chip ({result.target_type} #{result.target_serial}, {result.chip_height}x{result.chip_width})
      </p>
      <div className="mrf-chip-preview">
        <img src={`data:image/png;base64,${result.chip_preview_png}`} alt="MSTAR chip" />
        <div className="mrf-roi-box" style={boxStyle} />
      </div>
      <p className="hint">
        Real X-band radar magnitude on a decibel scale. The box marks the {result.size}x{result.size} region (row{" "}
        {result.row0}, column {result.col0}) processed by both solvers.
      </p>
    </div>
  );
}

function computeEdgeMap(labels: number[][]): number[][] {
  // THALES's own definition: "label as edge each pixel with at least one
  // neighbour of different class" - literal 4-connected implementation,
  // pure post-processing on labels the classical/quantum solver already
  // produced, no new optimization needed. Marks both sides of a boundary
  // (THALES itself calls single-sided marking an optional refinement).
  const h = labels.length;
  const w = labels[0]?.length ?? 0;
  return labels.map((row, r) =>
    row.map((v, c) => {
      const neighbors = [
        r > 0 ? labels[r - 1][c] : null,
        r < h - 1 ? labels[r + 1][c] : null,
        c > 0 ? labels[r][c - 1] : null,
        c < w - 1 ? labels[r][c + 1] : null,
      ];
      return neighbors.some((n) => n !== null && n !== v) ? 1 : 0;
    }),
  );
}

function computeSinglePixelEdgeMap(labels: number[][]): number[][] {
  // THALES's stated optional refinement: "ensure that only a single pixel
  // is labelled as edge for each edge (instead of 2, on both sides)".
  // Only checks the right/down neighbor (never left/up), so each boundary
  // is marked exactly once - at its lower-index (top/left) side - instead
  // of computeEdgeMap's both-sides marking. Every boundary is still
  // covered: a left/up neighbor's own right/down check already catches it.
  const h = labels.length;
  const w = labels[0]?.length ?? 0;
  return labels.map((row, r) =>
    row.map((v, c) => {
      const right = c < w - 1 ? labels[r][c + 1] : null;
      const down = r < h - 1 ? labels[r + 1][c] : null;
      return (right !== null && right !== v) || (down !== null && down !== v) ? 1 : 0;
    }),
  );
}

// THALES's other stated "further work": a multi-class edge detector that
// distinguishes the object's edges from the shadow's edges, applied to the
// Q=3 (background/target/shadow) labels. 0=not edge, 1=object edge
// (target/background boundary), 2=shadow edge (shadow/background
// boundary), 3=object/shadow edge - a target/shadow boundary directly, or
// a pixel touching more than one boundary type (a 3-way junction) -
// reported honestly as its own case rather than forced into object or
// shadow alone.
function computeShadowEdgeMap(labels: number[][]): number[][] {
  const h = labels.length;
  const w = labels[0]?.length ?? 0;
  return labels.map((row, r) =>
    row.map((v, c) => {
      const neighbors: number[] = [];
      if (r > 0) neighbors.push(labels[r - 1][c]);
      if (r < h - 1) neighbors.push(labels[r + 1][c]);
      if (c > 0) neighbors.push(labels[r][c - 1]);
      if (c < w - 1) neighbors.push(labels[r][c + 1]);
      const differing = neighbors.filter((n) => n !== v);
      if (differing.length === 0) return 0;
      const pairKeys = new Set(differing.map((n) => [v, n].sort().join(",")));
      if (pairKeys.size > 1) return 3;
      const [a, b] = [...pairKeys][0].split(",").map(Number);
      if ((a === 0 && b === 1) || (a === 1 && b === 0)) return 1;
      if ((a === 0 && b === 2) || (a === 2 && b === 0)) return 2;
      return 3;
    }),
  );
}

const SHADOW_EDGE_CLASS = ["mrf-cell", "mrf-cell-edge-object", "mrf-cell-edge-shadow", "mrf-cell-edge-mixed"];

function ShadowEdgeGrid({ labels, title }: { labels: number[][]; title: string }) {
  const size = labels.length;
  return (
    <div className="mrf-grid-block">
      <p className="mrf-grid-title">{title}</p>
      <div className="mrf-grid" style={{ gridTemplateColumns: `repeat(${size}, 1fr)`, gridTemplateRows: `repeat(${size}, 1fr)` }}>
        {labels.flatMap((row, r) => row.map((v, c) => <div key={`${r}-${c}`} className={SHADOW_EDGE_CLASS[v]} />))}
      </div>
    </div>
  );
}

function LabelGrid({ labels, title, itemLabel = "target" }: { labels: number[][]; title: string; itemLabel?: string }) {
  const size = labels.length;
  const flat = labels.flat();
  const nOn = flat.filter((v) => v === 1).length;
  return (
    <div className="mrf-grid-block">
      <p className="mrf-grid-title">{title}</p>
      <div className="mrf-grid" style={{ gridTemplateColumns: `repeat(${size}, 1fr)`, gridTemplateRows: `repeat(${size}, 1fr)` }}>
        {labels.flatMap((row, r) =>
          row.map((v, c) => <div key={`${r}-${c}`} className={v === 1 ? "mrf-cell mrf-cell-on" : "mrf-cell"} />),
        )}
      </div>
      <p className="hint mrf-grid-caption">
        {nOn} of {flat.length} pixels labeled {itemLabel}
        {itemLabel === "target" &&
          nOn === 0 &&
          " (expected if the tile contains only background clutter; see the boxed area on the chip)"}
      </p>
    </div>
  );
}

// 0=background, 1=target, 2=shadow
const SHADOW_LABEL_CLASS = ["mrf-cell", "mrf-cell-on", "mrf-cell-shadow"];

function ShadowGrid({ labels, title }: { labels: number[][]; title: string }) {
  const size = labels.length;
  const flat = labels.flat();
  const counts = [0, 1, 2].map((k) => flat.filter((v) => v === k).length);
  return (
    <div className="mrf-grid-block">
      <p className="mrf-grid-title">{title}</p>
      <div className="mrf-grid" style={{ gridTemplateColumns: `repeat(${size}, 1fr)`, gridTemplateRows: `repeat(${size}, 1fr)` }}>
        {labels.flatMap((row, r) => row.map((v, c) => <div key={`${r}-${c}`} className={SHADOW_LABEL_CLASS[v]} />))}
      </div>
      <p className="hint mrf-grid-caption">
        {counts[0]} background, {counts[1]} target, {counts[2]} shadow (out of {flat.length} pixels)
      </p>
    </div>
  );
}

type MetricValues = Record<string, number | null>;

function formatMetric(v: number | null | undefined): string {
  return v === null || v === undefined ? "n/a" : v.toFixed(4);
}

function MetricsTable({ classical, quantum }: { classical: MetricValues | null | undefined; quantum: MetricValues | null | undefined }) {
  const keys = Object.keys(classical ?? quantum ?? {});
  if (keys.length === 0) {
    return (
      <p className="hint">
        No reference mask is available for this chip, so IoU and precision cannot be computed. Energy is the only
comparison between the two results and is shown below. Accuracy on test images with known answers is on the
Synthetic Validation tab.
      </p>
    );
  }
  return (
    <table>
      <thead>
        <tr>
          <th>metric</th>
          <th>classical</th>
          <th>quantum</th>
        </tr>
      </thead>
      <tbody>
        {keys.map((k) => (
          <tr key={k}>
            <td>{k}</td>
            <td>{classical ? formatMetric(classical[k]) : "-"}</td>
            <td
              className={
                quantum && classical && quantum[k] != null && classical[k] != null && quantum[k]! > classical[k]!
                  ? "mrf-metric-better"
                  : undefined
              }
            >
              {quantum ? formatMetric(quantum[k]) : "-"}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

const QUANTUM_LINE_COLOR = "#7c3aed"; // matches the About page's quantum-branch color
const LOG_EPS = 1e-3; // energies can be exactly (or near) 0 for an all-agree assignment - floor before log10

function EnergyChart({
  costHistory,
  classicalEnergy,
  quantumEnergy,
}: {
  costHistory: number[];
  classicalEnergy: number;
  quantumEnergy: number;
}) {
  if (costHistory.length === 0) return null;
  const w = 480;
  const h = 200;
  const padLeft = 48;
  const padRight = 16;
  const padTop = 16;
  const padBottom = 30;

  // Log-scale y-axis for the trace itself (QUBO energies span orders of
  // magnitude across ROI sizes) - the final classical/quantum numbers are
  // shown as plain stat badges below instead of lines on this chart (two
  // near-overlapping flat lines low on a log axis were hard to read at a
  // glance - a labeled number is clearer than parsing line position).
  const logMin = Math.log10(Math.max(LOG_EPS, Math.min(...costHistory)));
  const logMax = Math.log10(Math.max(LOG_EPS, Math.max(...costHistory)) + 1e-9);
  const range = logMax - logMin || 1;
  const x = (i: number) => padLeft + (i / Math.max(1, costHistory.length - 1)) * (w - padLeft - padRight);
  const y = (v: number) => {
    const logV = Math.log10(Math.max(LOG_EPS, v));
    return h - padBottom - ((logV - logMin) / range) * (h - padTop - padBottom);
  };
  const path = costHistory.map((v, i) => `${i === 0 ? "M" : "L"} ${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(" ");

  const yTicks = [0, 0.5, 1].map((f) => {
    const logV = logMin + f * range;
    return { py: h - padBottom - f * (h - padTop - padBottom), value: Math.pow(10, logV) };
  });
  const xTickIndices = costHistory.length <= 6 ? costHistory.map((_, i) => i) : [0, Math.floor((costHistory.length - 1) / 2), costHistory.length - 1];
  const better = quantumEnergy <= classicalEnergy;

  return (
    <div className="mrf-chart-block">
      <p className="mrf-grid-title">Average energy of the sampled bitstrings at each optimizer step (the best single result is shown below)</p>
      <svg width={w} height={h} className="mrf-chart">
        {yTicks.map(({ py, value }, i) => (
          <g key={i}>
            <line x1={padLeft} y1={py} x2={w - padRight} y2={py} stroke="var(--border)" strokeWidth={1} />
            <text x={padLeft - 6} y={py + 3} textAnchor="end" fontSize="9" fill="var(--muted)">
              {value >= 10 ? value.toFixed(0) : value >= 1 ? value.toFixed(1) : value.toFixed(2)}
            </text>
          </g>
        ))}
        {xTickIndices.map((i) => (
          <text key={i} x={x(i)} y={h - padBottom + 14} textAnchor="middle" fontSize="9" fill="var(--muted)">
            {i + 1}
          </text>
        ))}
        <text x={(w + padLeft - padRight) / 2} y={h - 4} textAnchor="middle" fontSize="9" fill="var(--muted)">
          Optimizer step
        </text>
        <path d={path} fill="none" stroke="var(--accent-2)" strokeWidth={2} />
        {costHistory.map((v, i) => (
          <circle key={i} cx={x(i)} cy={y(v)} r={2.5} fill="var(--accent-2)" />
        ))}
      </svg>

      <div className="mrf-energy-badges">
        <div className="mrf-energy-badge">
          <span className="mrf-energy-badge-dot" style={{ background: "var(--accent)" }} />
          <span>Classical (exact)</span>
          <strong>{classicalEnergy}</strong>
        </div>
        <div className="mrf-energy-badge">
          <span className="mrf-energy-badge-dot" style={{ background: QUANTUM_LINE_COLOR }} />
          <span>Quantum (best result found)</span>
          <strong className={better ? "mrf-metric-better" : undefined}>{quantumEnergy}</strong>
        </div>
      </div>
    </div>
  );
}

export default function MrfPage() {
  const [mstarChips, setMstarChips] = useState<MstarChipInfo[]>([]);
  const [mstarChipId, setMstarChipId] = useState("");
  const [row0, setRow0] = useState(60);
  const [col0, setCol0] = useState(60);

  const [size, setSize] = useState(4);
  const [runQuantum, setRunQuantum] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [smoothness, setSmoothness] = useState(1.0);
  const [contrastSigma, setContrastSigma] = useState(0.15);
  const [noiseSigma, setNoiseSigma] = useState(0);
  const [qaoaReps, setQaoaReps] = useState(2);
  const [qaoaMaxIterations, setQaoaMaxIterations] = useState(15);
  const [qaoaNumShots, setQaoaNumShots] = useState(512);

  const [result, setResult] = useState<MstarResponse | null>(null);
  const [tab, setTab] = useState<Tab>("images");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 3-class (background/target/shadow) extension - THALES's own "beyond
  // binary" case. Reuses the same chip/ROI selection above; a separate,
  // smaller result since it is a genuinely different objective (Q=3, not
  // Q=2 - energies are not comparable between the two).
  const [shadowRunQuantum, setShadowRunQuantum] = useState(false);
  const [shadowLoading, setShadowLoading] = useState(false);
  const [shadowError, setShadowError] = useState<string | null>(null);
  const [shadowResult, setShadowResult] = useState<MstarShadowResponse | null>(null);

  // Pattern detection (target-type recognition) - this project's disclosed
  // interpretation of THALES's undefined "Pattern detection" requirement.
  // Reuses the same chip selection above; independent of ROI/size (it
  // classifies the whole chip's real target blob, not a small tile).
  const [patternLoading, setPatternLoading] = useState(false);
  const [patternError, setPatternError] = useState<string | null>(null);
  const [patternResult, setPatternResult] = useState<PatternResponse | null>(null);
  const [evalLoading, setEvalLoading] = useState(false);
  const [evalError, setEvalError] = useState<string | null>(null);
  const [evalResult, setEvalResult] = useState<PatternEvaluationResponse | null>(null);

  // Synthetic validation (real Accuracy/IoU/Dice) - see api/client.ts's
  // comment: the only place these metrics exist in this app, because it is
  // the only place real ground truth exists (never a real MSTAR chip).
  const [validationLoading, setValidationLoading] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [validationResult, setValidationResult] = useState<SyntheticValidationResponse | null>(null);
  const [validationRunQuantum, setValidationRunQuantum] = useState(false);

  // Live progress from POST /api/mstar/stream - real events as they
  // actually happen (each "progress" tick is a genuine ~4-5s round trip to
  // Classiq), not a simulated loading animation.
  const [liveClassical, setLiveClassical] = useState<{ energy: number; duration_s: number } | null>(null);
  const [liveIteration, setLiveIteration] = useState<{ iteration: number; maxIterations: number; bestEnergy: number } | null>(null);
  const [liveCostHistory, setLiveCostHistory] = useState<number[]>([]);
  const [elapsedS, setElapsedS] = useState(0);

  useEffect(() => {
    api.mstarChips().then((chips) => {
      setMstarChips(chips);
      if (chips.length > 0) setMstarChipId(chips[0].chip_id);
    });
  }, []);

  // Center a roiSize x roiSize ROI on a given (row, col), clamped to stay
  // in-bounds for a smaller chip (SLICY is 54x54, vehicle chips are
  // 128x128). Takes the size explicitly rather than reading the `size`
  // state, so a preset cannot center against a stale value from before a
  // size change takes effect.
  function centerRoiOn(chip: MstarChipInfo, row: number, col: number, roiSize: number) {
    const half = Math.floor(roiSize / 2);
    setRow0(Math.min(Math.max(0, row - half), Math.max(0, chip.height - roiSize)));
    setCol0(Math.min(Math.max(0, col - half), Math.max(0, chip.width - roiSize)));
  }

  function jumpToPreset(kind: "bright" | "no_object" | "low_snr") {
    const chip = mstarChips.find((c) => c.chip_id === mstarChipId);
    if (!chip) return;
    if (kind === "bright") centerRoiOn(chip, chip.bright_row, chip.bright_col, size);
    else if (kind === "no_object") centerRoiOn(chip, chip.no_object_row, chip.no_object_col, size);
    else centerRoiOn(chip, chip.low_snr_row, chip.low_snr_col, size);
  }

  // Default a newly-selected chip's ROI onto its real target blob
  // (bright_row/bright_col, from the connected-component centroid
  // utils/mrf/mstar_scenes.py computes) instead of the chip's geometric
  // center - verified those differ enough in practice that the geometric
  // center can land on plain background clutter (e.g. bmp2_9563_test_1's
  // center is background, not target).
  useEffect(() => {
    const chip = mstarChips.find((c) => c.chip_id === mstarChipId);
    if (!chip) return;
    centerRoiOn(chip, chip.bright_row, chip.bright_col, size);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mstarChipId, mstarChips]);

  const qubits = size * size;
  const quantumTooBig = runQuantum && qubits > MAX_QAOA_SIZE * MAX_QAOA_SIZE;

  async function run() {
    if (!mstarChipId) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setTab("images");
    setLiveClassical(null);
    setLiveIteration(null);
    setLiveCostHistory([]);
    setElapsedS(0);

    const startedAt = Date.now();
    const ticker = window.setInterval(() => setElapsedS((Date.now() - startedAt) / 1000), 200);

    try {
      await api.mstarStream(
        {
          chip_id: mstarChipId,
          row0,
          col0,
          size,
          smoothness,
          contrast_sigma: contrastSigma,
          noise_sigma: noiseSigma,
          run_quantum: runQuantum,
          qaoa_reps: qaoaReps,
          qaoa_max_iterations: qaoaMaxIterations,
          qaoa_num_shots: qaoaNumShots,
        },
        (event) => {
          if (event.type === "classical_done") {
            setLiveClassical({ energy: event.energy, duration_s: event.duration_s });
          } else if (event.type === "progress") {
            setLiveIteration({ iteration: event.iteration, maxIterations: event.max_iterations, bestEnergy: event.best_energy });
            setLiveCostHistory((h) => [...h, event.mean_energy]);
          } else if (event.type === "done") {
            setResult(event.result);
          } else if (event.type === "error") {
            setError(event.detail);
          }
        },
      );
    } catch (e) {
      setError(String(e));
    } finally {
      window.clearInterval(ticker);
      setLoading(false);
    }
  }

  const MAX_SHADOW_QAOA_SIZE = 2; // size*size*3 <= 25 qubits -> size<=2 (matches apps/api/routers/mstar.py's real ceiling)

  async function runShadow() {
    if (!mstarChipId) return;
    setShadowLoading(true);
    setShadowError(null);
    setShadowResult(null);
    try {
      const r = await api.mstarShadow({
        chip_id: mstarChipId,
        row0,
        col0,
        size,
        run_quantum: shadowRunQuantum,
      });
      setShadowResult(r);
    } catch (e) {
      setShadowError(String(e));
    } finally {
      setShadowLoading(false);
    }
  }

  async function runPattern() {
    if (!mstarChipId) return;
    setPatternLoading(true);
    setPatternError(null);
    setPatternResult(null);
    try {
      const r = await api.mstarPattern(mstarChipId);
      setPatternResult(r);
    } catch (e) {
      setPatternError(String(e));
    } finally {
      setPatternLoading(false);
    }
  }

  async function runEvaluation() {
    setEvalLoading(true);
    setEvalError(null);
    setEvalResult(null);
    try {
      const r = await api.mstarPatternEvaluation();
      setEvalResult(r);
    } catch (e) {
      setEvalError(String(e));
    } finally {
      setEvalLoading(false);
    }
  }

  async function runSyntheticValidation() {
    setValidationLoading(true);
    setValidationError(null);
    setValidationResult(null);
    try {
      // One case per request: a single request for all cases can outlast the browser's ~5 minute wait.
      const collected: SyntheticValidationResponse["cases"] = [];
      let total = 1;
      for (let i = 0; i < total; i++) {
        const r = await api.mstarSyntheticValidation({ run_quantum: validationRunQuantum, only_case: i });
        total = r.total_cases;
        collected.push(...r.cases);
        setValidationResult({ ...r, cases: [...collected] });
      }
    } catch (e) {
      setValidationError(String(e));
    } finally {
      setValidationLoading(false);
    }
  }

  const chipGroups = mstarChips.reduce<Record<string, MstarChipInfo[]>>((groups, c) => {
    (groups[c.target_type] ??= []).push(c);
    return groups;
  }, {});

  return (
    <div className="page">
      <div className="hero-intro">
        <h1>Classical and quantum segmentation on a common objective</h1>
        <p>
          Synthetic aperture radar (SAR) images contain speckle noise, which makes vehicles difficult to separate from the
background. This tool segments a real radar image by labelling each pixel as vehicle or background, using two
solvers on the same objective: an exact classical method (graph cut), which is guaranteed to find the optimal
labelling for two classes, and a quantum method (QAOA) executed on Classiq's simulator. The two results are
presented side by side on real radar data. The <a href="/about">About</a> page describes how the comparison is
kept fair.
        </p>
      </div>

      <div className="mrf-layout">
        <section className="mrf-panel mrf-input-panel">
          <h3>Input</h3>

          <div className="mrf-dataset-controls">
            <p className="hint">
              The chips are real radar images of vehicles from the public MSTAR dataset, released by the U.S. Air Force Research
Laboratory. MSTAR provides no per-pixel masks, so each result is scored against the SARBake reference mask where one
exists (T72, BMP2 and BTR70 chips) and is also compared on energy, a measure of solution quality (lower is better).
Exact accuracy on test images with known answers is on the Synthetic Validation tab.
            </p>
            <label>
              Chip ({mstarChips.length} real chips, {Object.keys(chipGroups).length} vehicle types)
              <select value={mstarChipId} onChange={(e) => setMstarChipId(e.target.value)}>
                {Object.entries(chipGroups).map(([targetType, chips]) => (
                  <optgroup key={targetType} label={targetType}>
                    {chips.map((c) => (
                      <option key={c.chip_id} value={c.chip_id}>
                        #{c.target_serial} - az {c.azimuth_deg}°, {c.depression_deg}° dep ({c.height}x{c.width})
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </label>
            <label>
              Region starts at row
              <input type="number" min={0} value={row0} onChange={(e) => setRow0(Number(e.target.value))} />
            </label>
            <label>
              Region starts at column
              <input type="number" min={0} value={col0} onChange={(e) => setCol0(Number(e.target.value))} />
            </label>
            <div className="mrf-roi-presets">
              <p className="hint">Preset regions for this chip:</p>
              <div className="mrf-roi-preset-buttons">
                <button type="button" onClick={() => jumpToPreset("bright")}>
                  Target
                </button>
                <button type="button" onClick={() => jumpToPreset("no_object")}>
                  No object
                </button>
                <button type="button" onClick={() => jumpToPreset("low_snr")}>
                  Low SNR
                </button>
              </div>
            </div>
          </div>

          <div className="mrf-section-divider" />

          <label>
            Tile size, in pixels per side ({qubits} qubit{qubits === 1 ? "" : "s"} if run on the quantum solver)
            <input type="number" min={1} max={MAX_ROI_SIZE} value={size} onChange={(e) => setSize(Number(e.target.value))} />
          </label>

          <label className="mrf-checkbox-row">
            <input type="checkbox" checked={runQuantum} onChange={(e) => setRunQuantum(e.target.checked)} />
            Also run the quantum solver (QAOA on Classiq)
          </label>

          <button type="button" className="mrf-advanced-toggle" onClick={() => setShowAdvanced((v) => !v)}>
            {showAdvanced ? "Hide" : "Show"} advanced options
          </button>
          {showAdvanced && (
            <div className="mrf-advanced">
              <label>
                Smoothness (strength of the neighbor agreement term)
                <input type="number" step={0.1} value={smoothness} onChange={(e) => setSmoothness(Number(e.target.value))} />
              </label>
              <label>
                Edge sensitivity (contrast sigma)
                <input type="number" step={0.01} value={contrastSigma} onChange={(e) => setContrastSigma(Number(e.target.value))} />
              </label>
              <label>
                Added noise (robustness testing; 0 = none)
                <input type="number" step={0.05} min={0} value={noiseSigma} onChange={(e) => setNoiseSigma(Number(e.target.value))} />
              </label>
              <label>
                QAOA layers
                <input type="number" min={1} value={qaoaReps} onChange={(e) => setQaoaReps(Number(e.target.value))} />
              </label>
              <label>
                Maximum optimizer steps
                <input type="number" min={1} value={qaoaMaxIterations} onChange={(e) => setQaoaMaxIterations(Number(e.target.value))} />
              </label>
              <label>
                Shots per step
                <input type="number" min={16} value={qaoaNumShots} onChange={(e) => setQaoaNumShots(Number(e.target.value))} />
              </label>
            </div>
          )}

          {quantumTooBig && (
            <p className="error">
              A {size}x{size} tile requires {qubits} qubits, which exceeds the {MAX_QAOA_SIZE * MAX_QAOA_SIZE}-qubit limit for
QAOA. Reduce the tile size or clear the quantum option.
            </p>
          )}

          <button className="mrf-run-button" onClick={run} disabled={loading || !mstarChipId || quantumTooBig}>
            {loading ? `Running... (${elapsedS.toFixed(0)}s)` : "Run pipeline"}
          </button>
          {error && <p className="error">{error}</p>}
        </section>

        <section className="mrf-panel mrf-results-panel">
          <h3>Results</h3>
          {loading && (
            <div className="mrf-live-panel">
              <p className="mrf-live-badge">
                <span className="live-dot live-dot-on" /> LIVE - {elapsedS.toFixed(1)}s so far
              </p>
              <p>
                {liveClassical ? (
                  <>Classical graph cut finished in {liveClassical.duration_s.toFixed(4)}s (energy {liveClassical.energy})</>
                ) : (
                  "Running the classical graph cut..."
                )}
              </p>
              {runQuantum && (
                <>
                  <p>
                    {liveIteration ? (
                      <>
                        QAOA step {liveIteration.iteration} of {liveIteration.maxIterations} on Classiq. Best energy so
                        far: {liveIteration.bestEnergy}
                      </>
                    ) : liveClassical ? (
                      "Sending the problem to Classiq and building the QAOA circuit..."
                    ) : null}
                  </p>
                  {liveCostHistory.length > 0 && (
                    <EnergyChart
                      costHistory={liveCostHistory}
                      classicalEnergy={liveClassical?.energy ?? 0}
                      quantumEnergy={liveIteration?.bestEnergy ?? liveClassical?.energy ?? 0}
                    />
                  )}
                </>
              )}
            </div>
          )}
          {!loading && (
            <>
              <div className="mrf-tabs">
                <button type="button" className={tab === "images" ? "active" : ""} onClick={() => setTab("images")}>
                  Images
                </button>
                <button type="button" className={tab === "metrics" ? "active" : ""} onClick={() => setTab("metrics")}>
                  Metrics
                </button>
                <button type="button" className={tab === "logs" ? "active" : ""} onClick={() => setTab("logs")}>
                  Logs
                </button>
                <button
                  type="button"
                  className={tab === "validation" ? "active" : ""}
                  onClick={() => setTab("validation")}
                >
                  Synthetic Validation
                </button>
                <button
                  type="button"
                  className={tab === "pattern" ? "active" : ""}
                  onClick={() => setTab("pattern")}
                >
                  Pattern Detection
                </button>
                <button
                  type="button"
                  className={tab === "shadow" ? "active" : ""}
                  onClick={() => setTab("shadow")}
                >
                  3-Class Segmentation
                </button>
              </div>

              {!result && !["validation", "shadow", "pattern"].includes(tab) && (
                <div className="mrf-empty-state">
                  <p>Select a chip and a region on the left, then choose Run pipeline.</p>
                  <p className="hint">
                    When the run completes, the classical and quantum results, the energy comparison, the scores against the
reference mask and the QAOA convergence chart appear here.
                  </p>
                </div>
              )}

              {tab === "images" && result && (
                <div>
                  <p className="hint mrf-grid-legend">
                    <span className="mrf-cell mrf-cell-legend-swatch" /> background (label 0) &nbsp;&nbsp;
                    <span className="mrf-cell mrf-cell-on mrf-cell-legend-swatch" /> target (label 1)
                  </p>
                  <div className="mrf-image-row">
                    <ChipPreview result={result} />
                    <LabelGrid labels={result.classical.labels} title={`Classical (energy ${result.classical.energy})`} />
                    {result.quantum ? (
                      <LabelGrid labels={result.quantum.labels} title={`Quantum (energy ${result.quantum.energy})`} />
                    ) : (
                      <div className="mrf-grid-block">
                        <p className="mrf-grid-title">Quantum</p>
                        <p className="hint">The quantum solver was not run.</p>
                      </div>
                    )}
                    {result.reference_mask && (
                      <LabelGrid labels={result.reference_mask} title="Reference (SARBake mask)" />
                    )}
                  </div>

                  <p className="hint mrf-grid-legend" style={{ marginTop: "1rem" }}>
                    Edge detection: a pixel is marked as an edge if any of its neighbors has a different label, as defined in the
THALES brief. The map is derived from the labels above and requires no further optimization. It marks both sides of
each boundary, whereas the brief's printed example marks only the object side.
                  </p>
                  <div className="mrf-image-row">
                    <LabelGrid
                      labels={computeEdgeMap(result.classical.labels)}
                      title="Classical edges"
                      itemLabel="edge"
                    />
                    {result.quantum && (
                      <LabelGrid labels={computeEdgeMap(result.quantum.labels)} title="Quantum edges" itemLabel="edge" />
                    )}
                  </div>

                  <p className="hint mrf-grid-legend" style={{ marginTop: "1rem" }}>
                    Single-pixel edges: each boundary is marked with one pixel instead of two, a refinement suggested in the brief.
                  </p>
                  <div className="mrf-image-row">
                    <LabelGrid
                      labels={computeSinglePixelEdgeMap(result.classical.labels)}
                      title="Classical edges (1px)"
                      itemLabel="edge"
                    />
                    {result.quantum && (
                      <LabelGrid
                        labels={computeSinglePixelEdgeMap(result.quantum.labels)}
                        title="Quantum edges (1px)"
                        itemLabel="edge"
                      />
                    )}
                  </div>
                </div>
              )}

              {tab === "shadow" && (
                <div>
                  <p className="hint" style={{ fontWeight: 600, color: "var(--fg)" }}>
                    Three-class segmentation: adding the vehicle's shadow
                  </p>
                  <p className="hint">
                    The THALES brief suggests extending the segmentation beyond two classes by also labelling the vehicle's shadow.
This runs on the chip and region selected on the left. With three classes the classical method (alpha-expansion)
returns a good but not provably optimal result, unlike the exact two-class method (see the About page). Shadow is
modelled as a dark, low-return area, and the result is scored against the SARBake mask where one is available. The
brief's own printed example uses a mid-level shadow, so the output will not match that example exactly.
                  </p>
                  <label className="mrf-checkbox-row">
                    <input
                      type="checkbox"
                      checked={shadowRunQuantum}
                      onChange={(e) => setShadowRunQuantum(e.target.checked)}
                    />
                    Also run the quantum solver (QAOA on Classiq). With three classes each pixel requires 3 qubits, so this option is
limited to tiles of {MAX_SHADOW_QAOA_SIZE}x{MAX_SHADOW_QAOA_SIZE} or smaller (25-qubit limit).
                  </label>
                  <p className="hint">
                    The selected tile is {size}x{size} and would require {size * size * 3} qubits.{" "}
                    {size > MAX_SHADOW_QAOA_SIZE
                      ? "This exceeds the limit for the quantum solver; leave the option cleared to obtain the classical result only."
                      : "This is within the limit for the quantum solver."}
                  </p>
                  {shadowRunQuantum && size > MAX_SHADOW_QAOA_SIZE && (
                    <p className="error">
                      Reduce the tile to {MAX_SHADOW_QAOA_SIZE}x{MAX_SHADOW_QAOA_SIZE} or smaller, or clear the quantum option.{" "}
                      <button
                        type="button"
                        className="mrf-advanced-toggle"
                        style={{ color: "inherit", textDecorationColor: "currentcolor" }}
                        onClick={() => setSize(MAX_SHADOW_QAOA_SIZE)}
                      >
                        Set tile to {MAX_SHADOW_QAOA_SIZE}x{MAX_SHADOW_QAOA_SIZE}
                      </button>
                    </p>
                  )}
                  <button
                    className="mrf-run-button"
                    onClick={runShadow}
                    disabled={shadowLoading || !mstarChipId || (shadowRunQuantum && size > MAX_SHADOW_QAOA_SIZE)}
                  >
                    {shadowLoading ? "Running..." : "Run 3-class segmentation"}
                  </button>
                  {shadowLoading && shadowRunQuantum && (
                    <p className="hint">
                      The quantum step typically takes a minute or longer because it executes on Classiq's platform.
                    </p>
                  )}
                  {shadowError && <p className="error">{shadowError}</p>}
                  {shadowResult && (
                    <div style={{ marginTop: "1rem" }}>
                      <p className="hint mrf-grid-legend">
                        <span className="mrf-cell mrf-cell-legend-swatch" /> background (label 0) &nbsp;&nbsp;
                        <span className="mrf-cell mrf-cell-on mrf-cell-legend-swatch" /> target (label 1) &nbsp;&nbsp;
                        <span className="mrf-cell mrf-cell-shadow mrf-cell-legend-swatch" /> shadow (label 2)
                      </p>
                      <div className="mrf-image-row">
                        <ChipPreview result={shadowResult} />
                        <ShadowGrid
                          labels={shadowResult.classical.labels}
                          title={`Classical (alpha-expansion, energy ${shadowResult.classical.energy})`}
                        />
                        {shadowResult.quantum ? (
                          <ShadowGrid
                            labels={shadowResult.quantum.labels}
                            title={`Quantum (energy ${shadowResult.quantum.energy})`}
                          />
                        ) : (
                          <div className="mrf-grid-block">
                            <p className="mrf-grid-title">Quantum</p>
                            <p className="hint">The quantum solver was not run.</p>
                          </div>
                        )}
                        {shadowResult.reference_mask && (
                          <ShadowGrid labels={shadowResult.reference_mask} title="Reference (SARBake mask)" />
                        )}
                      </div>
                      {shadowResult.classical.metrics && (
                        <>
                          <p className="hint">
                            Scored against the SARBake mask for this chip. The mask is derived from the vehicle's CAD model, so it is a good
reference but not hand-drawn ground truth. Scores on small tiles are low, because a few speckled pixels provide little
evidence.
                          </p>
                          <MetricsTable
                            classical={shadowResult.classical.metrics}
                            quantum={shadowResult.quantum?.metrics}
                          />
                        </>
                      )}

                      <p className="hint mrf-grid-legend" style={{ marginTop: "1rem" }}>
                        Edges by type, as suggested in the brief (object edges versus shadow edges):
                        <span className="mrf-cell mrf-cell-edge-object mrf-cell-legend-swatch" /> object edge&nbsp;&nbsp;
                        <span className="mrf-cell mrf-cell-edge-shadow mrf-cell-legend-swatch" /> shadow edge&nbsp;&nbsp;
                        <span className="mrf-cell mrf-cell-edge-mixed mrf-cell-legend-swatch" /> object/shadow edge
                      </p>
                      <div className="mrf-image-row">
                        <ShadowEdgeGrid labels={computeShadowEdgeMap(shadowResult.classical.labels)} title="Classical edges" />
                        {shadowResult.quantum && (
                          <ShadowEdgeGrid labels={computeShadowEdgeMap(shadowResult.quantum.labels)} title="Quantum edges" />
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {tab === "pattern" && (
                <div>
                  <p className="hint" style={{ fontWeight: 600, color: "var(--fg)" }}>
                    Pattern detection: vehicle type recognition
                  </p>
                  <p className="hint">
                    The THALES brief names pattern detection without defining it. We interpret it as recognizing which vehicle (BMP2,
BTR70 or T72) a chip shows, by correlating the chip against average templates built from the MSTAR training chips.
The method uses the whole chip, not only the selected tile. Unlike segmentation, the correct answer is known, because
each chip's header states the vehicle type, so the result can be verified.
                  </p>
                  <div className="mrf-checkbox-row" style={{ gap: "0.6rem" }}>
                    <button className="mrf-run-button" onClick={runPattern} disabled={patternLoading || !mstarChipId}>
                      {patternLoading ? "Running..." : "Classify this chip"}
                    </button>
                    <button type="button" className="mrf-advanced-toggle" onClick={runEvaluation} disabled={evalLoading}>
                      {evalLoading ? "Checking every test chip (about 30s)..." : "Show accuracy across all test chips"}
                    </button>
                  </div>
                  {patternError && <p className="error">{patternError}</p>}
                  {patternResult && (
                    <table style={{ marginTop: "0.75rem" }}>
                      <tbody>
                        <tr>
                          <td>chip</td>
                          <td>{patternResult.chip_id}</td>
                        </tr>
                        <tr>
                          <td>what it really is (from the chip's header)</td>
                          <td>{patternResult.actual_class}</td>
                        </tr>
                        <tr>
                          <td>what we predicted</td>
                          <td className={patternResult.correct ? "mrf-metric-better" : undefined}>
                            {patternResult.predicted_class} {patternResult.correct ? "(correct)" : "(incorrect)"}
                          </td>
                        </tr>
                        {Object.entries(patternResult.scores).map(([cls, score]) => (
                          <tr key={cls}>
                            <td>how closely it matches {cls}</td>
                            <td>{score}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                  {evalError && <p className="error">{evalError}</p>}
                  {evalResult && (
                    <div style={{ marginTop: "0.75rem" }}>
                      <p className="hint">{evalResult.note}</p>
                      <table>
                        <thead>
                          <tr>
                            <th>accuracy</th>
                            <th>test chips checked</th>
                            {evalResult.classes.map((c) => (
                              <th key={c.class_name}>{c.class_name} training chips</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          <tr>
                            <td>{(evalResult.accuracy * 100).toFixed(1)}%</td>
                            <td>{evalResult.n_test_samples}</td>
                            {evalResult.classes.map((c) => (
                              <td key={c.class_name}>{c.n_train_samples}</td>
                            ))}
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  )}

                </div>
              )}

              {tab === "validation" && (
                <div>
                  <p className="hint" style={{ fontWeight: 600, color: "var(--fg)" }}>
                    Synthetic validation: exact accuracy scores
                  </p>
                  <p className="hint">
                    These cases are not MSTAR chips. Each has a known correct answer by construction, so accuracy, IoU and Dice are
exact. One case is THALES's own worked example, reproduced from the brief; the others are small images generated for
this project with a deliberately placed target region. Scores for real chips, against the SARBake masks, are shown in
the Metrics tab and on the About page.
                  </p>
                  <label className="mrf-checkbox-row">
                    <input
                      type="checkbox"
                      checked={validationRunQuantum}
                      onChange={(e) => setValidationRunQuantum(e.target.checked)}
                    />
                    Also run the quantum solver on each case (slower; about 60 to 95 seconds each)
                  </label>
                  <button className="mrf-run-button" onClick={runSyntheticValidation} disabled={validationLoading}>
                    {validationLoading ? "Running (this can take a few minutes)" : "Run synthetic validation"}
                  </button>
                  {validationError && <p className="error">{validationError}</p>}
                  {validationResult && (
                    <div style={{ marginTop: "0.75rem" }}>
                      <p className="hint">{validationResult.note}</p>
                      {validationResult.cases.map((c) => (
                        <div key={c.name} style={{ marginTop: "0.75rem" }}>
                          <p className="mrf-grid-title">
                            {c.name.includes(`${c.size}x${c.size}`) ? c.name : `${c.name} (${c.size}x${c.size})`} - <span className="hint">{c.source}</span>
                          </p>
                          <table>
                            <thead>
                              <tr>
                                <th>metric</th>
                                <th>classical</th>
                                <th>quantum</th>
                              </tr>
                            </thead>
                            <tbody>
                              {(["accuracy", "iou", "dice"] as const).map((k) => (
                                <tr key={k}>
                                  <td>{k}</td>
                                  <td>{c.classical.metrics ? (c.classical.metrics[k] * 100).toFixed(1) + "%" : "-"}</td>
                                  <td>{c.quantum?.metrics ? (c.quantum.metrics[k] * 100).toFixed(1) + "%" : "-"}</td>
                                </tr>
                              ))}
                              <tr>
                                <td>energy</td>
                                <td>{c.classical.energy}</td>
                                <td>{c.quantum ? c.quantum.energy : "-"}</td>
                              </tr>
                              <tr>
                                <td>runtime</td>
                                <td>{c.classical.duration_s.toFixed(4)}s</td>
                                <td>{c.quantum ? `${c.quantum.duration_s.toFixed(2)}s` : "-"}</td>
                              </tr>
                            </tbody>
                          </table>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {tab === "metrics" && result && (
                <div>
                  <MetricsTable classical={result.classical.metrics} quantum={result.quantum?.metrics} />
                  {result.note && <p className="hint">{result.note}</p>}
                  {result.quantum && (
                    <EnergyChart
                      costHistory={result.quantum.cost_history}
                      classicalEnergy={result.classical.energy}
                      quantumEnergy={result.quantum.energy}
                    />
                  )}
                </div>
              )}

              {tab === "logs" && result && (
                <div>
                  <table>
                    <tbody>
                      <tr>
                        <td>classical duration</td>
                        <td>{result.classical.duration_s.toFixed(4)}s</td>
                      </tr>
                      {result.quantum && (
                        <>
                          <tr>
                            <td>quantum duration</td>
                            <td>{result.quantum.duration_s.toFixed(2)}s</td>
                          </tr>
                          <tr>
                            <td>qubits</td>
                            <td>{result.quantum.n_qubits}</td>
                          </tr>
                          <tr>
                            <td>circuit width / depth</td>
                            <td>
                              {result.quantum.circuit_width} / {result.quantum.circuit_depth ?? "?"}
                            </td>
                          </tr>
                          <tr>
                            <td>reps / shots / backend</td>
                            <td>
                              {result.quantum.reps} / {result.quantum.num_shots} / {result.quantum.backend}
                            </td>
                          </tr>
                          <tr>
                            <td>optimized params</td>
                            <td>{result.quantum.optimized_params.join(", ")}</td>
                          </tr>
                        </>
                      )}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}
        </section>
      </div>
    </div>
  );
}
