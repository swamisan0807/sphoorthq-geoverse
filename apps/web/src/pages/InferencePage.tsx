import { FormEvent, useState } from "react";

import { api, InferenceJob } from "../api/client";

export default function InferencePage() {
  const [bbox, setBbox] = useState("67.0,44.0,68.0,45.0");
  const [date, setDate] = useState("");
  const [model, setModel] = useState<"classic-unet" | "quantum-hybrid">(
    "classic-unet",
  );
  const [job, setJob] = useState<InferenceJob | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const aoi_bbox = bbox.split(",").map(Number) as [
        number,
        number,
        number,
        number,
      ];
      const result = await api.runInference({
        aoi_bbox,
        date: date || undefined,
        model,
        objective: "flood-segmentation",
      });
      setJob(result);
    } catch (e) {
      setError(String(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <>
      <div className="page-title">Run Inference</div>
      <div className="page-subtitle">
        Submits a job against /api/inference/run. Currently returns a stub
        "queued" job — wire src/ai training/inference to complete it.
      </div>
      <div className="card">
        <form onSubmit={handleSubmit}>
          <div className="form-row">
            <label>AOI bbox (minx,miny,maxx,maxy)</label>
            <input value={bbox} onChange={(e) => setBbox(e.target.value)} />
          </div>
          <div className="form-row">
            <label>Date (optional)</label>
            <input
              type="date"
              value={date}
              onChange={(e) => setDate(e.target.value)}
            />
          </div>
          <div className="form-row">
            <label>Model</label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value as typeof model)}
            >
              <option value="classic-unet">Classic U-Net</option>
              <option value="quantum-hybrid">Quantum Hybrid</option>
            </select>
          </div>
          <button className="primary" type="submit" disabled={submitting}>
            {submitting ? "Submitting…" : "Run"}
          </button>
        </form>
      </div>
      {error && <div className="card">{error}</div>}
      {job && (
        <div className="card">
          <div>Job: {job.job_id}</div>
          <div>
            Status: <span className={`badge ${job.status === "completed" ? "processed" : "raw"}`}>{job.status}</span>
          </div>
        </div>
      )}
    </>
  );
}
