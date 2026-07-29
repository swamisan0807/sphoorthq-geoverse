import { useEffect, useState } from "react";
import { api, type EventSummary, type QuantumResponse } from "../api/client";

export default function QuantumPage() {
  const [events, setEvents] = useState<EventSummary[] | null>(null);
  const [selectedEvent, setSelectedEvent] = useState("");
  const [chipIds, setChipIds] = useState<string[]>([]);
  const [chipId, setChipId] = useState("");
  const [backend, setBackend] = useState<"ibm" | "braket">("ibm");
  const [nTrain, setNTrain] = useState(12);
  const [nTest, setNTest] = useState(6);
  const [forceSimulation, setForceSimulation] = useState(true);
  const [result, setResult] = useState<QuantumResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const r = await api.quantumKernelSvm({
        chip_id: chipId,
        backend,
        n_train: nTrain,
        n_test: nTest,
        force_simulation: forceSimulation,
      });
      setResult(r);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
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
        on IBM Quantum (Qiskit) or AWS Braket, a classical SVM classifies on top of that quantum-computed
        kernel matrix. There is no persisted quantum model file, unlike RF/U-Net on the Inference tab -
        every run here submits real circuits (to a local simulator by default, or real hardware if you
        uncheck simulation and have credentials configured).
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
          <select value={backend} onChange={(e) => setBackend(e.target.value as "ibm" | "braket")}>
            <option value="ibm">IBM Quantum (Qiskit)</option>
            <option value="braket">AWS Braket</option>
          </select>
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
          <input
            type="checkbox"
            checked={forceSimulation}
            onChange={(e) => setForceSimulation(e.target.checked)}
          />
          {" "}force simulation (no real hardware)
        </label>
        <button onClick={run} disabled={loading || !chipId}>
          {loading ? "running circuits..." : "run quantum kernel SVM"}
        </button>
      </div>

      <p className="hint">
        this run submits {nFit} fit() circuits + {nPredict} predict() circuits ={" "}
        {nFit + nPredict} total - O(n_train^2 + n_train*n_test), a real cost of the quantum kernel method
        regardless of backend
      </p>

      {error && <p className="error">{error}</p>}
      {loading && <p className="hint">This can take 20-60s on a local simulator, longer on real hardware queue time.</p>}

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
