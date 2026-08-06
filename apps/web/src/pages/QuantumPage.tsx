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

  const nFit = (nTrain * (nTrain + 1)) / 2;
  const nPredict = nTrain * nTest;

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
          call <code>QiskitRuntimeService</code>, and never written to disk (same channel/token/instance
          triple as <code>test_ibm_connection.py</code>). Leave blank to fall back to
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
    </div>
  );
}
