export interface DatasetSummary {
  id: string;
  source: string;
  acquisition_date?: string;
  footprint_bbox?: [number, number, number, number];
  crs?: string;
  thumbnail_url?: string;
  status: "raw" | "processed" | "missing";
}

export interface InferenceRequest {
  aoi_bbox: [number, number, number, number];
  date?: string;
  model: "classic-unet" | "quantum-hybrid";
  objective: string;
}

export interface InferenceJob {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  result_tile_url?: string;
  metrics?: Record<string, number>;
}

export interface ExperimentSummary {
  id: string;
  model_type: "classic" | "quantum";
  objective: string;
  metrics: Record<string, number>;
  created_at: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`);
  }
  return res.json();
}

export const api = {
  listDatasets: () => request<DatasetSummary[]>("/datasets"),
  getDataset: (id: string) => request<DatasetSummary>(`/datasets/${id}`),
  runInference: (req: InferenceRequest) =>
    request<InferenceJob>("/inference/run", {
      method: "POST",
      body: JSON.stringify(req),
    }),
  getJob: (jobId: string) => request<InferenceJob>(`/inference/${jobId}`),
  listExperiments: () => request<ExperimentSummary[]>("/experiments"),
};
