const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";
const TOKEN_STORAGE_KEY = "geoverse_token";
const USERNAME_STORAGE_KEY = "geoverse_username";

let _token: string | null = localStorage.getItem(TOKEN_STORAGE_KEY);
let _username: string | null = localStorage.getItem(USERNAME_STORAGE_KEY);

export function getToken(): string | null {
  return _token;
}

export function getUsername(): string | null {
  return _username;
}

export function setSession(token: string, username: string): void {
  _token = token;
  _username = username;
  localStorage.setItem(TOKEN_STORAGE_KEY, token);
  localStorage.setItem(USERNAME_STORAGE_KEY, username);
}

export function clearSession(): void {
  _token = null;
  _username = null;
  localStorage.removeItem(TOKEN_STORAGE_KEY);
  localStorage.removeItem(USERNAME_STORAGE_KEY);
}

function authHeaders(): Record<string, string> {
  return _token ? { Authorization: `Bearer ${_token}` } : {};
}

let _onUnauthorized: (() => void) | null = null;

/** Called whenever a request comes back 401 (expired/invalid stored
 * token) - clears the session and lets App.tsx bounce back to the login
 * screen instead of every page silently failing with "401 Unauthorized". */
export function setUnauthorizedHandler(fn: () => void): void {
  _onUnauthorized = fn;
}

function handleUnauthorized(status: number): void {
  if (status === 401) {
    clearSession();
    _onUnauthorized?.();
  }
}

export interface EventSummary {
  event: string;
  train: number;
  valid: number;
  test: number;
  total: number;
}

export interface ChipList {
  split: string;
  event: string | null;
  chip_ids: string[];
}

export interface StageRecord {
  name: string;
  status: string;
  duration_s: number | null;
  metrics: Record<string, unknown>;
  error: string | null;
}

export interface RunRecord {
  run_id: string;
  run_name: string;
  started_at: number;
  ended_at: number | null;
  stages: StageRecord[];
  metrics: Record<string, unknown>;
}

export interface ModelInfo {
  name: string;
  size_bytes: number;
  modified_at: number;
}

export interface InferResponse {
  chip_id: string;
  model: string;
  metrics: Record<string, number> | null;
  n_valid_pixels: number;
  n_nodata_pixels: number;
  note: string | null;
}

export interface QuantumRequest {
  chip_id: string;
  backend: "ibm" | "braket";
  n_train: number;
  n_test: number;
  force_simulation: boolean;
}

export interface QuantumResponse {
  chip_id: string;
  backend: string;
  is_real_hardware: boolean;
  backend_name: string;
  n_train: number;
  n_test: number;
  n_fit_circuits: number;
  n_predict_circuits: number;
  metrics: Record<string, number>;
  duration_s: number;
}

export interface JobRecord {
  job_id: string;
  notebook: string;
  status: "queued" | "running" | "success" | "failed";
  triggered_by: string;
  started_at: number;
  ended_at: number | null;
  returncode: number | null;
  log_tail: string;
  kind: "notebook" | "quantum";
  extra: Record<string, unknown>;
}

export interface ModelVersion {
  model_name: string;
  version: number;
  created_at: number;
  notebook: string;
  registered_by: string;
  metrics: Record<string, number>;
  file: string;
  size_bytes: number;
}

export interface RegistrySummary {
  model_name: string;
  current_version: number | null;
  n_versions: number;
}

export interface NewDataStatus {
  has_new_data: boolean;
  new_chip_ids: string[];
  total_chips: number;
  baseline_just_created: boolean;
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { headers: authHeaders() });
  if (!res.ok) {
    handleUnauthorized(res.status);
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<T>;
}

async function postJSON<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    // login/signup themselves can 401/409 on bad credentials - that's not
    // an expired session, so don't treat it as one
    if (path !== "/api/auth/login" && path !== "/api/auth/signup") {
      handleUnauthorized(res.status);
    }
    const body2 = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body2}`);
  }
  return res.json() as Promise<T>;
}

/** <img src=...> can't send an Authorization header - fetch the image as an
 * authenticated blob instead and hand back an object URL. Caller is
 * responsible for revoking it (see useAuthedImage hook) when done. */
export async function fetchAuthedImageUrl(path: string): Promise<string> {
  const res = await fetch(`${API_BASE}${path}`, { headers: authHeaders() });
  if (!res.ok) {
    handleUnauthorized(res.status);
    throw new Error(`${res.status} ${res.statusText}`);
  }
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

export const api = {
  health: () => getJSON<{ status: string }>("/api/health"),

  signup: (username: string, password: string) =>
    postJSON<{ token: string; username: string }>("/api/auth/signup", { username, password }),
  login: (username: string, password: string) =>
    postJSON<{ token: string; username: string }>("/api/auth/login", { username, password }),
  me: () => getJSON<{ username: string; created_at: number }>("/api/auth/me"),

  events: () => getJSON<EventSummary[]>("/api/catalog/events"),
  chips: (split: string, event?: string) =>
    getJSON<ChipList>(`/api/catalog/chips?split=${split}${event ? `&event=${event}` : ""}`),
  chipPreviewPath: (chipId: string) => `/api/catalog/chips/${chipId}/preview.png`,

  runs: (limit = 20) => getJSON<RunRecord[]>(`/api/runs?limit=${limit}`),
  models: () => getJSON<ModelInfo[]>("/api/models"),
  architecture: () => getJSON<{ mermaid: string }>("/api/architecture"),

  infer: (chipId: string, model: string) =>
    postJSON<InferResponse>("/api/inference", { chip_id: chipId, model }),
  inferencePreviewPath: (chipId: string, model: string) =>
    `/api/inference/${chipId}/preview.png?model=${model}`,

  quantumKernelSvm: (req: QuantumRequest) => postJSON<QuantumResponse>("/api/quantum/kernel-svm", req),

  jobNotebooks: () => getJSON<string[]>("/api/jobs/notebooks"),
  jobs: (limit = 30) => getJSON<JobRecord[]>(`/api/jobs?limit=${limit}`),
  jobDetail: (jobId: string) => getJSON<JobRecord>(`/api/jobs/${jobId}`),
  runNotebookJob: (notebook: string) => postJSON<JobRecord>("/api/jobs", { notebook }),

  registrySummary: () => getJSON<RegistrySummary[]>("/api/registry"),
  registryVersions: (modelName: string) =>
    getJSON<{ model_name: string; current_version: number | null; versions: ModelVersion[] }>(
      `/api/registry/${modelName}/versions`,
    ),
  restoreVersion: (modelName: string, version: number) =>
    postJSON<ModelVersion>(`/api/registry/${modelName}/restore`, { version }),

  compareData: () => getJSON<Record<string, Record<string, number>>>("/api/compare/data"),
  comparePlotPath: () => "/api/compare/plot.png",

  graphMermaid: () => getJSON<{ mermaid: string; n_nodes: number; n_edges: number }>("/api/graph/mermaid"),

  checkNewData: () => getJSON<NewDataStatus>("/api/pipeline/check-new-data"),
  retrainIfNew: () =>
    postJSON<NewDataStatus & { triggered: boolean; job_ids: string[] }>("/api/pipeline/retrain-if-new"),
};

export { API_BASE };
