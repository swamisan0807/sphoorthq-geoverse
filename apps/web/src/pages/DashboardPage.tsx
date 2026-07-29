import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { api, ExperimentSummary } from "../api/client";

export default function DashboardPage() {
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.listExperiments().then(setExperiments).catch((e) => setError(String(e)));
  }, []);

  const chartData = experiments.map((e) => ({
    name: e.id,
    iou: e.metrics.iou ?? 0,
    f1: e.metrics.f1 ?? 0,
    boundary_f1: e.metrics.boundary_f1 ?? 0,
  }));

  return (
    <>
      <div className="page-title">Experiments</div>
      <div className="page-subtitle">
        Classic vs quantum model comparison. Populated once src/ai training
        loops write results to datasets/reports.
      </div>
      {error && <div className="card">Failed to load: {error}</div>}
      <div className="card" style={{ height: 320 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#262b36" />
            <XAxis dataKey="name" stroke="#8b93a3" fontSize={12} />
            <YAxis stroke="#8b93a3" fontSize={12} domain={[0, 1]} />
            <Tooltip
              contentStyle={{ background: "#151922", border: "1px solid #262b36" }}
            />
            <Legend />
            <Bar dataKey="iou" fill="#3b6fe0" />
            <Bar dataKey="f1" fill="#4dd07c" />
            <Bar dataKey="boundary_f1" fill="#f0b64d" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}
