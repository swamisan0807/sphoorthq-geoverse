import { useEffect, useState } from "react";
import { api } from "../api/client";
import { useAuthedImage } from "../api/useAuthedImage";

export default function ComparePage() {
  const [data, setData] = useState<Record<string, Record<string, number>> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { url: plotUrl, error: plotError } = useAuthedImage(api.comparePlotPath());

  useEffect(() => {
    api.compareData().then(setData).catch((e) => setError(String(e)));
  }, []);

  const metricNames = data ? Array.from(new Set(Object.values(data).flatMap((m) => Object.keys(m)))) : [];

  return (
    <div className="page">
      <h2>Classical vs Quantum</h2>
      <p className="hint">
        Server-side (Python/matplotlib) observability plot comparing the current registry-pointed RF and
        U-Net versions against the most recent successful quantum kernel SVM run. Quantum runs use a small
        sample (tens of points, not full-image pixel counts) - a perfect score there reflects a tiny test
        set, not proof of outright superiority over the classical models trained on the full split.
      </p>
      {error && <p className="error">{error}</p>}
      {plotError && <p className="error">{plotError}</p>}
      {plotUrl && <img className="preview" src={plotUrl} alt="classical vs quantum comparison" />}

      {data && (
        <table>
          <thead>
            <tr>
              <th>model</th>
              {metricNames.map((m) => (
                <th key={m}>{m}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.entries(data).map(([model, metrics]) => (
              <tr key={model}>
                <td>{model}</td>
                {metricNames.map((m) => (
                  <td key={m}>{metrics[m] !== undefined ? metrics[m].toFixed(4) : "-"}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
