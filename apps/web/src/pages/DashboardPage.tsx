import { Fragment, useEffect, useRef, useState } from "react";
import mermaid from "mermaid";
import { api, type ModelInfo, type RunRecord } from "../api/client";
import StatusBadge from "../components/StatusBadge";

mermaid.initialize({ startOnLoad: false, theme: "dark", securityLevel: "loose" });

function formatBytes(n: number): string {
  if (n > 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n > 1e3) return `${(n / 1e3).toFixed(1)} KB`;
  return `${n} B`;
}

function formatTime(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString();
}

function DiagramView() {
  const [source, setSource] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.architecture().then((r) => setSource(r.mermaid)).catch((e) => setSource(`%% error: ${e}`));
  }, []);

  useEffect(() => {
    if (!source || !containerRef.current) return;
    const el = containerRef.current;
    mermaid
      .render("architecture-diagram", source)
      .then(({ svg }) => {
        el.innerHTML = svg;
      })
      .catch((e) => {
        el.textContent = `diagram render failed: ${e}`;
      });
  }, [source]);

  return <div className="diagram" ref={containerRef} />;
}

export default function DashboardPage() {
  const [runs, setRuns] = useState<RunRecord[] | null>(null);
  const [models, setModels] = useState<ModelInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    api.runs(20).then(setRuns).catch((e) => setError(String(e)));
    api.models().then(setModels).catch((e) => setError(String(e)));
  }, []);

  const totalModelSize = models?.reduce((sum, m) => sum + m.size_bytes, 0) ?? null;
  const fullyOkRuns = runs?.filter((r) => r.stages.length > 0 && r.stages.every((s) => s.status === "ok")).length ?? null;
  const successRate = runs && runs.length > 0 && fullyOkRuns !== null ? Math.round((fullyOkRuns / runs.length) * 100) : null;
  const lastRun = runs && runs.length > 0 ? runs[0] : null;
  const lastRunOk = lastRun ? lastRun.stages.every((s) => s.status === "ok") : null;

  return (
    <div className="page">
      <h2>Observability Dashboard</h2>
      {error && <p className="error">{error}</p>}

      {(models || runs) && (
        <section className="stat-grid">
          <div className="stat-tile">
            <span className="stat-label">Trained models</span>
            <span className="stat-value">{models ? models.length : "—"}</span>
            {totalModelSize !== null && <span className="stat-sub">{formatBytes(totalModelSize)} total</span>}
          </div>
          <div className="stat-tile">
            <span className="stat-label">Recent runs</span>
            <span className="stat-value">{runs ? runs.length : "—"}</span>
            <span className="stat-sub">most recently logged</span>
          </div>
          <div className="stat-tile">
            <span className="stat-label">Success rate</span>
            <span className="stat-value">{successRate !== null ? `${successRate}%` : "—"}</span>
            <span className="stat-sub">
              {fullyOkRuns ?? 0}/{runs?.length ?? 0} runs fully ok
            </span>
          </div>
          <div className="stat-tile">
            <span className="stat-label">Last run</span>
            {lastRun ? (
              <>
                <span className="stat-value stat-value-sm">{lastRun.run_name}</span>
                <span className="stat-sub">
                  <StatusBadge status={lastRunOk ? "ok" : "failed"} />
                </span>
              </>
            ) : (
              <span className="stat-value">—</span>
            )}
          </div>
        </section>
      )}

      <section>
        <h3>Saved models</h3>
        {!models && <p>loading...</p>}
        {models && models.length === 0 && <p>No models trained yet - run notebooks 04 / 09.</p>}
        {models && models.length > 0 && (
          <table>
            <thead>
              <tr>
                <th>name</th>
                <th>size</th>
                <th>last modified</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.name}>
                  <td>{m.name}</td>
                  <td>{formatBytes(m.size_bytes)}</td>
                  <td>{formatTime(m.modified_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section>
        <h3>Recent pipeline runs</h3>
        {!runs && <p>loading...</p>}
        {runs && runs.length === 0 && <p>No runs logged yet - execute a notebook first.</p>}
        {runs && runs.length > 0 && (
          <table>
            <thead>
              <tr>
                <th>notebook</th>
                <th>started</th>
                <th>duration</th>
                <th>stages</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => {
                const duration = r.ended_at ? (r.ended_at - r.started_at).toFixed(1) : "running";
                const isOpen = expanded === r.run_id;
                const okCount = r.stages.filter((s) => s.status === "ok").length;
                return (
                  <Fragment key={r.run_id}>
                    <tr className="run-row" onClick={() => setExpanded(isOpen ? null : r.run_id)}>
                      <td>{r.run_name}</td>
                      <td>{formatTime(r.started_at)}</td>
                      <td>{duration}s</td>
                      <td>
                        <StatusBadge status={okCount === r.stages.length ? "ok" : "failed"} /> {okCount}/{r.stages.length}
                      </td>
                      <td>{isOpen ? "▲" : "▼"}</td>
                    </tr>
                    {isOpen && (
                      <tr>
                        <td colSpan={5}>
                          <table className="nested">
                            <thead>
                              <tr>
                                <th>stage</th>
                                <th>status</th>
                                <th>duration</th>
                                <th>metrics</th>
                              </tr>
                            </thead>
                            <tbody>
                              {r.stages.map((s) => (
                                <tr key={s.name}>
                                  <td>{s.name}</td>
                                  <td>
                                    <StatusBadge status={s.status} />
                                  </td>
                                  <td>{s.duration_s ?? "-"}s</td>
                                  <td>
                                    <code>{JSON.stringify(s.metrics)}</code>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        )}
      </section>

      <section>
        <h3>Architecture</h3>
        <DiagramView />
      </section>
    </div>
  );
}
