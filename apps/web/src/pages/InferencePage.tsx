import { useEffect, useState } from "react";
import { api, type EventSummary, type InferResponse } from "../api/client";
import { useAuthedImage } from "../api/useAuthedImage";

export default function InferencePage() {
  const [events, setEvents] = useState<EventSummary[] | null>(null);
  const [selectedEvent, setSelectedEvent] = useState("");
  const [split, setSplit] = useState("valid");
  const [chipIds, setChipIds] = useState<string[]>([]);
  const [chipId, setChipId] = useState("");
  const [model, setModel] = useState<"rf" | "unet">("rf");
  const [result, setResult] = useState<InferResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { url: previewUrl, error: previewError } = useAuthedImage(
    result ? api.inferencePreviewPath(result.chip_id, result.model) : null,
  );

  useEffect(() => {
    api.events().then((evs) => {
      setEvents(evs);
      if (evs.length > 0) setSelectedEvent(evs[0].event);
    });
  }, []);

  useEffect(() => {
    if (!selectedEvent) return;
    api.chips(split, selectedEvent).then((r) => {
      setChipIds(r.chip_ids);
      setChipId(r.chip_ids[0] ?? "");
    });
  }, [split, selectedEvent]);

  async function runInference() {
    if (!chipId) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const r = await api.infer(chipId, model);
      setResult(r);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <h2>Inference</h2>
      <p className="hint">Runs the real trained models against a chosen sen1floods11 chip - not a mock.</p>

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
          split
          <select value={split} onChange={(e) => setSplit(e.target.value)}>
            <option value="train">train</option>
            <option value="valid">valid</option>
            <option value="test">test</option>
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
          model
          <select value={model} onChange={(e) => setModel(e.target.value as "rf" | "unet")}>
            <option value="rf">Random Forest (pixel-wise)</option>
            <option value="unet">U-Net (patch-wise)</option>
          </select>
        </label>
        <button onClick={runInference} disabled={loading || !chipId}>
          {loading ? "running..." : "run inference"}
        </button>
      </div>

      {error && <p className="error">{error}</p>}

      {result && (
        <section>
          <h3>
            {result.chip_id} - {result.model}
          </h3>
          {result.note && <p className="hint">{result.note}</p>}
          {result.metrics && (
            <table>
              <thead>
                <tr>
                  {Object.keys(result.metrics).map((k) => (
                    <th key={k}>{k}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr>
                  {Object.values(result.metrics).map((v, i) => (
                    <td key={i}>{v.toFixed(4)}</td>
                  ))}
                </tr>
              </tbody>
            </table>
          )}
          <p className="hint">
            {result.n_valid_pixels.toLocaleString()} valid px, {result.n_nodata_pixels.toLocaleString()} S1
            nodata px excluded
          </p>
          <p className="hint">left: input · center: ground truth (blue = water, gray = nodata) · right: prediction</p>
          {previewError && <p className="error">{previewError}</p>}
          {previewUrl && <img className="preview" src={previewUrl} alt="inference preview" />}
        </section>
      )}
    </div>
  );
}
