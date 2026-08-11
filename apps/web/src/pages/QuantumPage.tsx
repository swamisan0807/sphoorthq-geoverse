import { useEffect, useState } from "react";
import { api, type ConnectResponse, type EventSummary, type QuantumResponse } from "../api/client";

type Phase = "idle" | "connecting" | "connected" | "running";

export default function QuantumPage() {
  const [events, setEvents] = useState<EventSummary[] | null>(null);
  const [selectedEvent, setSelectedEvent] = useState("");
  const [chipIds, setChipIds] = useState<string[]>([]);
  const [chipId, setChipId] = useState("");
  const [backend] = useState<"ibm">("ibm");
  const [nTrain, setNTrain] = useState(12);
  const [nTest, setNTest] = useState(6);
  const [mode, setMode] = useState<"sim" | "real">("sim");
  const [ibmChannel, setIbmChannel] = useState("ibm_cloud");
  const [ibmInstance, setIbmInstance] = useState("");
  const [ibmToken, setIbmToken] = useState("");
  const [result, setResult] = useState<QuantumResponse | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [connectInfo, setConnectInfo] = useState<ConnectResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const loading = phase !== "idle";

  // Fair benchmark (multi-chip) - separate n_train/n_test defaults matching
  // notebooks 05/06, and its own result/phase/error so running the
  // single-chip explorer above doesn't clobber a benchmark result still on
  // screen (or vice versa). Shares the mode/IBM-credentials controls above,
  // since "which backend to use" is one decision either action can make.
  const [benchNTrain, setBenchNTrain] = useState(20);
  const [benchNTest, setBenchNTest] = useState(10);
  const [benchResult, setBenchResult] = useState<QuantumResponse | null>(null);
  const [benchPhase, setBenchPhase] = useState<Phase>("idle");
  const [benchError, setBenchError] = useState<string | null>(null);
  const benchLoading = benchPhase !== "idle";

  useEffect(() => {
    api.events().then((evs) => {
      setEvents(evs);
      if (evs.length > 0) setSelectedEvent(evs[0].event);
    });
  }, []);

  useEffect(() => {
    if (!selectedEvent) return;
    api.chips("train", selectedEvent).then((r) => {
      setChipIds(r.chip_ids);
      setChipId(r.chip_ids[0] ?? "");
    });
  }, [selectedEvent]);

  async function run() {
    if (!chipId) return;
    setError(null);
    setResult(null);
    setConnectInfo(null);

    if (mode === "real") {
      setPhase("connecting");
      try {
        const c = await api.quantumConnect({
          ibm_channel: ibmChannel,
          ibm_instance: ibmInstance,
          ibm_token: ibmToken,
        });
        setConnectInfo(c);
        setPhase("connected");
      } catch (e) {
        setError(String(e));
        setPhase("idle");
        return;
      }
    }

    setPhase("running");
    try {
      const r = await api.quantumKernelSvm({
        chip_id: chipId,
        backend,
        n_train: nTrain,
        n_test: nTest,
        force_simulation: mode === "sim",
        ...(mode === "real"
          ? { ibm_channel: ibmChannel, ibm_instance: ibmInstance, ibm_token: ibmToken }
          : {}),
      });
      setResult(r);
    } catch (e) {
      setError(String(e));
    } finally {
      setPhase("idle");
    }
  }

  async function runBenchmark() {
    setBenchError(null);
    setBenchResult(null);
    setConnectInfo(null);

    if (mode === "real") {
      setBenchPhase("connecting");
      try {
        const c = await api.quantumConnect({
          ibm_channel: ibmChannel,
          ibm_instance: ibmInstance,
          ibm_token: ibmToken,
        });
        setConnectInfo(c);
        setBenchPhase("connected");
      } catch (e) {
        setBenchError(String(e));
        setBenchPhase("idle");
        return;
      }
    }

    setBenchPhase("running");
    try {
      const r = await api.quantumBenchmark({
        backend,
        n_train: benchNTrain,
        n_test: benchNTest,
        force_simulation: mode === "sim",
        ...(mode === "real"
          ? { ibm_channel: ibmChannel, ibm_instance: ibmInstance, ibm_token: ibmToken }
          : {}),
      });
      setBenchResult(r);
    } catch (e) {
      setBenchError(String(e));
    } finally {
      setBenchPhase("idle");
    }
  }

  const nFit = (nTrain * (nTrain + 1)) / 2;
  const nPredict = nTrain * nTest;
  const benchNFit = (benchNTrain * (benchNTrain + 1)) / 2;
  const benchNPredict = benchNTrain * benchNTest;

  return (
    <div className="page">
      <h2>Quantum Kernel SVM</h2>
      <p className="hint">
        Trains a real quantum-kernel SVM (Havlicek et al. 2019) fresh on a small balanced pixel sample
        from the chosen chip - a ZZFeatureMap-style angle-encoding circuit computes pixel-pair similarity
        on IBM Quantum (Qiskit), a classical SVM classifies on top of that quantum-computed kernel matrix.
        There is no persisted quantum model file, unlike RF/U-Net on the Inference tab - every run here
        submits real circuits (to a local simulator by default, or to your own real IBM Quantum hardware
        if you pick "Real IBM Hardware" below and supply an account).
      </p>

      <div className="controls">
        <label>
          event
          <select value={selectedEvent} onChange={(e) => setSelectedEvent(e.target.value)}>
            {events?.map((e) => (
              <option key={e.event} value={e.event}>
                {e.event}
              </option>
            ))}
          </select>
        </label>
        <label>
          chip
          <select value={chipId} onChange={(e) => setChipId(e.target.value)}>
            {chipIds.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>
        </label>
        <label>
          backend
          <input type="text" value="IBM Quantum (Qiskit)" disabled />
        </label>
        <label>
          n_train
          <input
            type="number"
            min={4}
            max={24}
            value={nTrain}
            onChange={(e) => setNTrain(Number(e.target.value))}
          />
        </label>
        <label>
          n_test
          <input
            type="number"
            min={2}
            max={12}
            value={nTest}
            onChange={(e) => setNTest(Number(e.target.value))}
          />
        </label>
        <label>
          run on
          <select value={mode} onChange={(e) => setMode(e.target.value as "sim" | "real")}>
            <option value="sim">Qiskit Simulation</option>
            <option value="real">Real IBM Hardware</option>
          </select>
        </label>

        {mode === "real" && (
          <>
            <label>
              channel
              <input
                type="text"
                value={ibmChannel}
                onChange={(e) => setIbmChannel(e.target.value)}
                placeholder="ibm_cloud"
              />
            </label>
            <label>
              instance CRN
              <input
                type="text"
                value={ibmInstance}
                onChange={(e) => setIbmInstance(e.target.value)}
                placeholder="crn:v1:bluemix:..."
              />
            </label>
            <label>
              API token
              <input
                type="password"
                value={ibmToken}
                onChange={(e) => setIbmToken(e.target.value)}
                placeholder="hidden, sent only for this run"
                autoComplete="off"
              />
            </label>
          </>
        )}

        <button onClick={run} disabled={loading || !chipId}>
          {phase === "connecting" && "connecting to IBM Quantum..."}
          {phase === "connected" && "connected - starting job..."}
          {phase === "running" && "job running - executing circuits..."}
          {phase === "idle" && "run quantum kernel SVM"}
        </button>
      </div>

      {connectInfo && (
        <p className="hint">
          connected - {connectInfo.n_backends} real backend(s) visible on this account
          {connectInfo.n_backends > 0 && <> (e.g. {connectInfo.backend_name})</>}
        </p>
      )}

      {mode === "real" && (
        <p className="hint">
          Connects with this account for this run only - the token is sent over the request, used once to
          call <code>QiskitRuntimeService</code>, and never written to disk. Leave blank to fall back to
          IBM_QUANTUM_TOKEN/IBM_QUANTUM_INSTANCE configured on the server, if any.
        </p>
      )}

      <p className="hint">
        this run submits {nFit} fit() circuits + {nPredict} predict() circuits ={" "}
        {nFit + nPredict} total - O(n_train^2 + n_train*n_test), a real cost of the quantum kernel method
        regardless of backend
      </p>

      {error && <p className="error">{error}</p>}
      {phase === "connecting" && (
        <p className="hint">Authenticating with IBM Cloud - usually a few seconds, longer if the account is rate-limited.</p>
      )}
      {phase === "running" && (
        <p className="hint">
          Circuits executing{mode === "real" ? " on real hardware - can take from seconds to real queue time" : " on a local simulator - usually 20-60s"}.
        </p>
      )}

      {result && (
        <section>
          <h3>
            {result.chip_id} - {result.backend} ({result.backend_name})
          </h3>
          <p className={result.is_real_hardware ? "error" : "hint"}>
            {result.is_real_hardware ? "ran on REAL quantum hardware" : "ran on local simulator - no real hardware, no cost"}
          </p>
          <table>
            <thead>
              <tr>
                {Object.keys(result.metrics).map((k) => (
                  <th key={k}>{k}</th>
                ))}
                <th>duration</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                {Object.values(result.metrics).map((v, i) => (
                  <td key={i}>{v.toFixed(4)}</td>
                ))}
                <td>{result.duration_s.toFixed(1)}s</td>
              </tr>
            </tbody>
          </table>
          <p className="hint">
            n_train={result.n_train}, n_test={result.n_test}, {result.n_fit_circuits} fit circuits,{" "}
            {result.n_predict_circuits} predict circuits
          </p>
        </section>
      )}

      <h2>Fair Benchmark (multi-chip)</h2>
      <p className="hint">
        The run above trains and tests on one chosen chip - useful for exploring, but not a fair number next
        to the classical Random Forest/U-Net on the Compare page, which are evaluated across dozens of
        chips. This instead pools its train pixels across the sen1floods11 train split and its test pixels
        across this project's own event-holdout split (Nigeria + Somalia) - same sampling notebooks 05/06
        use for their fair-comparison fix - so the result is comparable. Only benchmark runs like this one
        (not the single-chip explorer above) ever show up on the Compare page's quantum series.
      </p>

      <div className="controls">
        <label>
          n_train
          <input
            type="number"
            min={4}
            max={24}
            value={benchNTrain}
            onChange={(e) => setBenchNTrain(Number(e.target.value))}
          />
        </label>
        <label>
          n_test
          <input
            type="number"
            min={2}
            max={12}
            value={benchNTest}
            onChange={(e) => setBenchNTest(Number(e.target.value))}
          />
        </label>
        <button onClick={runBenchmark} disabled={benchLoading}>
          {benchPhase === "connecting" && "connecting to IBM Quantum..."}
          {benchPhase === "connected" && "connected - starting job..."}
          {benchPhase === "running" && "job running - executing circuits..."}
          {benchPhase === "idle" && "run fair benchmark"}
        </button>
      </div>

      <p className="hint">
        this run submits {benchNFit} fit() circuits + {benchNPredict} predict() circuits ={" "}
        {benchNFit + benchNPredict} total, pooled from up to {Math.ceil(benchNTrain / 2)} train chips + up
        to {Math.ceil(benchNTest / 2)} test chips (uses "run on" / real-hardware settings above)
      </p>

      {benchError && <p className="error">{benchError}</p>}
      {benchPhase === "running" && (
        <p className="hint">
          Circuits executing{mode === "real" ? " on real hardware - can take from seconds to real queue time" : " on a local simulator - usually 20-60s"}, plus time spent reading each chip's raster.
        </p>
      )}

      {benchResult && (
        <section>
          <h3>
            fair benchmark - {benchResult.backend} ({benchResult.backend_name})
          </h3>
          <p className={benchResult.is_real_hardware ? "error" : "hint"}>
            {benchResult.is_real_hardware ? "ran on REAL quantum hardware" : "ran on local simulator - no real hardware, no cost"}
          </p>
          <table>
            <thead>
              <tr>
                {Object.keys(benchResult.metrics).map((k) => (
                  <th key={k}>{k}</th>
                ))}
                <th>duration</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                {Object.values(benchResult.metrics).map((v, i) => (
                  <td key={i}>{v.toFixed(4)}</td>
                ))}
                <td>{benchResult.duration_s.toFixed(1)}s</td>
              </tr>
            </tbody>
          </table>
          <p className="hint">
            n_train={benchResult.n_train} (across {benchResult.n_train_chips} chips), n_test=
            {benchResult.n_test} (across {benchResult.n_test_chips} chips), {benchResult.n_fit_circuits} fit
            circuits, {benchResult.n_predict_circuits} predict circuits
          </p>
        </section>
      )}
    </div>
  );
}
