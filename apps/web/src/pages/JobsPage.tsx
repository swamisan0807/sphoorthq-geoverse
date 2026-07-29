import { Fragment, useEffect, useRef, useState } from "react";
import { api, type JobRecord, type NewDataStatus } from "../api/client";

function formatTime(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString();
}

export default function JobsPage() {
  const [notebooks, setNotebooks] = useState<string[]>([]);
  const [selected, setSelected] = useState("");
  const [jobs, setJobs] = useState<JobRecord[]>([]);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [newData, setNewData] = useState<NewDataStatus | null>(null);
  const [checking, setChecking] = useState(false);
  const pollRef = useRef<number | null>(null);

  function refreshJobs() {
    api.jobs(30).then(setJobs).catch((e) => setError(String(e)));
  }

  useEffect(() => {
    api.jobNotebooks().then((nbs) => {
      setNotebooks(nbs);
      if (nbs.length > 0) setSelected(nbs[0]);
    });
    refreshJobs();
  }, []);

  useEffect(() => {
    const hasRunning = jobs.some((j) => j.status === "running" || j.status === "queued");
    if (hasRunning && pollRef.current === null) {
      pollRef.current = window.setInterval(refreshJobs, 4000);
    } else if (!hasRunning && pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
    return () => {
      if (pollRef.current !== null) window.clearInterval(pollRef.current);
    };
  }, [jobs]);

  async function runJob() {
    setError(null);
    try {
      await api.runNotebookJob(selected);
      refreshJobs();
    } catch (e) {
      setError(String(e));
    }
  }

  async function checkNewData() {
    setChecking(true);
    setError(null);
    try {
      setNewData(await api.checkNewData());
    } catch (e) {
      setError(String(e));
    } finally {
      setChecking(false);
    }
  }

  async function retrainIfNew() {
    setChecking(true);
    setError(null);
    try {
      const result = await api.retrainIfNew();
      setNewData(result);
      refreshJobs();
    } catch (e) {
      setError(String(e));
    } finally {
      setChecking(false);
    }
  }

  return (
    <div className="page">
      <h2>Jobs</h2>
      <p className="hint">
        Trigger real notebook executions (like a Databricks job run) and watch status. There's no
        cell-by-cell live stream - nbconvert only writes the notebook back once it finishes - so status
        goes queued -&gt; running -&gt; success/failed with a final log tail.
      </p>

      <div className="controls">
        <label>
          notebook
          <select value={selected} onChange={(e) => setSelected(e.target.value)}>
            {notebooks.map((nb) => (
              <option key={nb} value={nb}>
                {nb}
              </option>
            ))}
          </select>
        </label>
        <button onClick={runJob} disabled={!selected}>
          run notebook
        </button>
      </div>

      <section>
        <h3>Auto-retrain on new data</h3>
        <p className="hint">
          Diffs the on-disk chip set against the last-known baseline. If new chips appear (e.g. from a
          fresh ingestion run), triggers real retraining jobs for both classical models.
        </p>
        <div className="controls">
          <button onClick={checkNewData} disabled={checking}>
            check for new data
          </button>
          <button onClick={retrainIfNew} disabled={checking}>
            retrain if new
          </button>
        </div>
        {newData && (
          <p className="hint">
            {newData.total_chips} total chips on disk.{" "}
            {newData.baseline_just_created
              ? "Baseline just created (first check) - nothing to compare against yet."
              : newData.has_new_data
                ? `${newData.new_chip_ids.length} new chip(s): ${newData.new_chip_ids.join(", ")}`
                : "no new data since last check."}
          </p>
        )}
      </section>

      {error && <p className="error">{error}</p>}

      <section>
        <h3>Job history</h3>
        <table>
          <thead>
            <tr>
              <th>label</th>
              <th>kind</th>
              <th>status</th>
              <th>triggered by</th>
              <th>started</th>
              <th>duration</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {jobs.map((j) => {
              const isOpen = expanded === j.job_id;
              const duration = j.ended_at ? `${(j.ended_at - j.started_at).toFixed(1)}s` : "running...";
              return (
                <Fragment key={j.job_id}>
                  <tr className="run-row" onClick={() => setExpanded(isOpen ? null : j.job_id)}>
                    <td>{j.notebook}</td>
                    <td>{j.kind}</td>
                    <td className={j.status === "success" ? "ok" : j.status === "failed" ? "failed" : ""}>
                      {j.status}
                    </td>
                    <td>{j.triggered_by}</td>
                    <td>{formatTime(j.started_at)}</td>
                    <td>{duration}</td>
                    <td>{isOpen ? "▲" : "▼"}</td>
                  </tr>
                  {isOpen && (
                    <tr>
                      <td colSpan={7}>
                        {j.kind === "quantum" ? (
                          <code>{JSON.stringify(j.extra, null, 2)}</code>
                        ) : (
                          <pre className="log-tail">{j.log_tail || "(no output captured)"}</pre>
                        )}
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </section>
    </div>
  );
}
