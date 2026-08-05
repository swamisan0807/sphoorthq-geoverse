import { useEffect, useState } from "react";
import { api } from "../api/client";
import { useAuthedImage } from "../api/useAuthedImage";
import CompareChart from "../components/CompareChart";

export default function ComparePage() {
  const [data, setData] = useState<Record<string, Record<string, number>> | null>(null);
  const [error, setError] = useState<string | null>(null);
  // still fetched for the "download static PNG" link below the chart - the
  // interactive chart is the primary view since a flat image can't show
  // values on hover.
  const { url: plotUrl } = useAuthedImage(api.comparePlotPath());

  useEffect(() => {
    api.compareData().then(setData).catch((e) => setError(String(e)));
  }, []);

  const metricNames = data ? Array.from(new Set(Object.values(data).flatMap((m) => Object.keys(m)))) : [];

  return (
    <div className="page">
      <h2>Classical vs Quantum</h2>
      <p className="hint">
        Comparing the current registry-pointed RF and U-Net versions against the most recent successful
        quantum kernel SVM run. Hover (or tab to) a metric to see every model's value at once. Quantum runs
        use a small sample (tens of points, not full-image pixel counts) - a perfect score there reflects a
        tiny test set, not proof of outright superiority over the classical models trained on the full
        split.
      </p>
      {error && <p className="error">{error}</p>}
      {data && <CompareChart data={data} />}
      {plotUrl && (
        <p className="hint">
          <a href={plotUrl} download="classical_vs_quantum.png">
            Download static PNG (server-rendered)
          </a>
        </p>
      )}

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
