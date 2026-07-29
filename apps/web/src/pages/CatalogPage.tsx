import { useEffect, useState } from "react";

import { api, DatasetSummary } from "../api/client";

export default function CatalogPage() {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.listDatasets().then(setDatasets).catch((e) => setError(String(e)));
  }, []);

  return (
    <>
      <div className="page-title">Dataset Catalog</div>
      <div className="page-subtitle">
        Sources scanned from datasets/raw. Status flips to "processed" once
        src/processing writes to datasets/processed.
      </div>
      {error && <div className="card">Failed to load: {error}</div>}
      <div className="card">
        <table className="data-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Source</th>
              <th>Acquisition Date</th>
              <th>CRS</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {datasets.map((d) => (
              <tr key={d.id}>
                <td>{d.id}</td>
                <td>{d.source}</td>
                <td>{d.acquisition_date ?? "—"}</td>
                <td>{d.crs ?? "—"}</td>
                <td>
                  <span className={`badge ${d.status}`}>{d.status}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
