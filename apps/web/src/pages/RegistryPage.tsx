import { useEffect, useState } from "react";
import { api, type ModelVersion, type RegistrySummary } from "../api/client";

function formatTime(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString();
}

export default function RegistryPage() {
  const [summary, setSummary] = useState<RegistrySummary[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [versions, setVersions] = useState<ModelVersion[]>([]);
  const [currentVersion, setCurrentVersion] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  function refresh() {
    api.registrySummary().then(setSummary).catch((e) => setError(String(e)));
  }

  useEffect(refresh, []);

  useEffect(() => {
    if (!selected) return;
    api
      .registryVersions(selected)
      .then((r) => {
        setVersions(r.versions);
        setCurrentVersion(r.current_version);
      })
      .catch((e) => setError(String(e)));
  }, [selected]);

  async function restore(version: number) {
    if (!selected) return;
    setBusy(true);
    setError(null);
    try {
      await api.restoreVersion(selected, version);
      const r = await api.registryVersions(selected);
      setVersions(r.versions);
      setCurrentVersion(r.current_version);
      refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <h2>Model Registry</h2>
      <p className="hint">
        Every successful training job registers a new immutable version. "Restore" points inference at a
        prior version instantly - no retraining - the Inference tab re-resolves the current pointer on its
        next request.
      </p>
      {error && <p className="error">{error}</p>}

      <table>
        <thead>
          <tr>
            <th>model</th>
            <th>current version</th>
            <th># versions</th>
          </tr>
        </thead>
        <tbody>
          {summary?.map((s) => (
            <tr key={s.model_name} className={`clickable ${selected === s.model_name ? "selected" : ""}`} onClick={() => setSelected(s.model_name)}>
              <td>{s.model_name}</td>
              <td>{s.current_version ?? "none"}</td>
              <td>{s.n_versions}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {selected && (
        <section>
          <h3>{selected} versions</h3>
          <table>
            <thead>
              <tr>
                <th>version</th>
                <th>notebook</th>
                <th>registered by</th>
                <th>created</th>
                <th>metrics</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {versions.map((v) => (
                <tr key={v.version}>
                  <td>
                    v{v.version} {v.version === currentVersion && <strong>(current)</strong>}
                  </td>
                  <td>{v.notebook}</td>
                  <td>{v.registered_by}</td>
                  <td>{formatTime(v.created_at)}</td>
                  <td>
                    <code>{Object.entries(v.metrics).map(([k, val]) => `${k}=${val.toFixed(3)}`).join(" ")}</code>
                  </td>
                  <td>
                    <button disabled={busy || v.version === currentVersion} onClick={() => restore(v.version)}>
                      restore
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}
