import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import { useAuthedImage } from "../api/useAuthedImage";
import CompareChart from "../components/CompareChart";

const POLL_MS = 5000;

export default function ComparePage() {
  const [data, setData] = useState<Record<string, Record<string, number>> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [live, setLive] = useState(true);
  // still fetched for the "download static PNG" link below the chart - the
  // interactive chart is the primary view since a flat image can't show
  // values on hover.
  const { url: plotUrl } = useAuthedImage(api.comparePlotPath());

  const refresh = useCallback(() => {
    api
      .compareData()
      .then((d) => {
        setData(d);
        setLastUpdated(new Date());
        setError(null);
      })
      .catch((e) => setError(String(e)));
  }, []);

  // Live polling: this page reflects whatever's newest in the model
  // registry + job history, and both change from elsewhere in the app
  // (Jobs tab, Quantum tab) - poll instead of requiring a manual reload so
  // a run finishing anywhere shows up here within a few seconds.
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  useEffect(() => {
    refresh();
    if (live) {
      intervalRef.current = setInterval(refresh, POLL_MS);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [refresh, live]);

  const metricNames = data ? Array.from(new Set(Object.values(data).flatMap((m) => Object.keys(m)))) : [];

  return (
    <div className="page">
      <div className="page-header-row">
        <h2>Classical vs Quantum</h2>
        <div className="live-status">
          <button className="live-toggle" onClick={() => setLive((v) => !v)} aria-pressed={live}>
            <span className={live ? "live-dot live-dot-on" : "live-dot"} aria-hidden="true" />
            {live ? "live" : "paused"}
          </button>
          {lastUpdated && (
            <span className="hint">updated {lastUpdated.toLocaleTimeString()}</span>
          )}
        </div>
      </div>
      <p className="hint">
        Comparing the current registry-pointed RF and U-Net versions against the most recent successful
        quantum kernel SVM run of each kind - a simulator run and a real-hardware run are tracked as
        separate series, never averaged together. Hover (or tab to) a metric to see every model's value at
        once. This view refreshes itself every {POLL_MS / 1000}s, so a job finishing anywhere else in the
        app (Jobs, Quantum) shows up here without a reload. Quantum runs use a small sample (tens of
        points, not full-image pixel counts) - a perfect score there reflects a tiny test set, not proof of
        outright superiority over the classical models trained on the full split.
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
