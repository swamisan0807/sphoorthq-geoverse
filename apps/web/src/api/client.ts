// Explicit VITE_API_BASE always wins. Otherwise: in `npm run dev` (two
// separate servers - Vite on :5173, uvicorn on :8000) default to the local
// API. In a production build (`npm run build`), default to "" - a
// same-origin relative path - because apps/api/main.py serves this built
// UI itself, so the deployed app is one process on one port and the
// browser never needs a different host at all.
//
// "localhost", not "127.0.0.1": apps/api/routers/auth.py's Auth0 redirect_uri
// is derived from the actual incoming request's own host (request.base_url),
// so whatever hostname this constant uses is exactly the hostname Auth0 gets
// told to redirect back to - it has to match a URL registered in Auth0's
// Allowed Callback URLs byte-for-byte, and "127.0.0.1" != "localhost" to
// that exact-match check even though both reach the same machine.
const API_BASE = import.meta.env.VITE_API_BASE ?? (import.meta.env.DEV ? "http://localhost:8000" : "");
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

export interface MrfSolverResult {
  labels: number[][];
  energy: number;
  metrics: Record<string, number> | null;
  duration_s: number;
}

export interface MrfQuantumResult extends MrfSolverResult {
  n_qubits: number;
  reps: number;
  backend: string;
  circuit_width: number;
  circuit_depth: number | null;
  num_shots: number;
  optimized_params: number[];
  cost_history: number[]; // shot-weighted mean QUBO energy per COBYLA iteration - the energy-convergence trace
}

// Real MSTAR SAR vehicle-target chips (utils/mrf/mstar_scenes.py). The public
// release has no per-pixel mask, so results are scored against the SARBake
// reference mask when one is available (reference_mask, metrics), and
// compared on energy only otherwise. See MstarResponse.note.
export interface MstarChipInfo {
  chip_id: string;
  target_type: string;
  target_serial: string;
  depression_deg: number;
  azimuth_deg: number;
  height: number;
  width: number;
  bright_row: number; // centroid of the real target blob - default the ROI here, not the geometric center
  bright_col: number;
  no_object_row: number; // real, verified target-free window - THALES's "images without object" case
  no_object_col: number;
  low_snr_row: number; // real, verified low-contrast target-edge window - THALES's "low SNR" case
  low_snr_col: number;
}

export interface MstarRequest {
  chip_id: string;
  row0: number;
  col0: number;
  size: number;
  smoothness?: number;
  contrast_sigma?: number;
  noise_sigma?: number; // synthetic Gaussian noise added on top of the real magnitude - robustness testing
  run_quantum: boolean;
  qaoa_reps?: number;
  qaoa_max_iterations?: number;
  qaoa_num_shots?: number;
  qaoa_backend?: string;
}

export interface MstarResponse {
  chip_id: string;
  target_type: string;
  target_serial: string;
  chip_height: number;
  chip_width: number;
  chip_preview_png: string; // base64 PNG of the FULL chip's real magnitude
  row0: number;
  col0: number;
  size: number;
  roi_intensity: number[][]; // the calibrated intensity the unary term was built from
  reference_mask: number[][] | null; // SARBake mask for this window (1 = target), when available
  classical: MrfSolverResult;
  quantum: MrfQuantumResult | null;
  note: string | null;
}

// Real per-iteration events from POST /api/mstar/stream - each "progress"
// event corresponds to one genuine ExecutionSession.sample() round trip to
// Classiq's platform (~4-5s), not a client-side simulated tick.
export type MstarStreamEvent =
  | { type: "classical_done"; energy: number; duration_s: number }
  | { type: "progress"; iteration: number; max_iterations: number; mean_energy: number; best_energy: number }
  | { type: "done"; result: MstarResponse }
  | { type: "error"; detail: string };

// 3-class (background/target/shadow) extension - THALES's own "beyond
// binary" case. See apps/api/routers/mstar.py's /shadow endpoint and
// utils/mrf/multilabel_energy.py for what's genuinely different here
// (approximate alpha-expansion classical solver, not exact graph-cut;
// one-hot QUBO tripling the qubit cost per pixel).
export interface MstarShadowRequest {
  chip_id: string;
  row0: number;
  col0: number;
  size: number;
  smoothness?: number;
  contrast_sigma?: number;
  run_quantum: boolean;
  qaoa_reps?: number;
  qaoa_max_iterations?: number;
  qaoa_num_shots?: number;
  qaoa_backend?: string;
}

export interface ShadowSolverResult {
  labels: number[][]; // 0=background, 1=target, 2=shadow
  energy: number;
  duration_s: number;
  metrics?: Record<string, number> | null; // scored against the SARBake mask when available
}

export interface ShadowQuantumSolverResult extends ShadowSolverResult {
  n_qubits: number;
  backend: string;
  circuit_width: number;
  circuit_depth: number | null;
  cost_history: number[];
}

export interface MstarShadowResponse {
  chip_id: string;
  target_type: string;
  target_serial: string;
  chip_height: number;
  chip_width: number;
  chip_preview_png: string;
  row0: number;
  col0: number;
  size: number;
  background_mu: number;
  reference_mask: number[][] | null; // SARBake mask for this window (0/1/2), when available
  classical: ShadowSolverResult;
  quantum: ShadowQuantumSolverResult | null;
  note: string | null;
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

/** Reads POST /api/mstar/stream's newline-delimited JSON response,
 * invoking onEvent for each real event as it arrives (not buffered until
 * the whole response completes) - the browser's fetch streaming body
 * reader, no SSE/EventSource needed since this is a POST with a JSON
 * body. */
async function mstarStream(req: MstarRequest, onEvent: (e: MstarStreamEvent) => void): Promise<void> {
  const res = await fetch(`${API_BASE}/api/mstar/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(req),
  });
  if (!res.ok || !res.body) {
    handleUnauthorized(res.status);
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newlineIndex: number;
    while ((newlineIndex = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newlineIndex).trim();
      buffer = buffer.slice(newlineIndex + 1);
      if (line) onEvent(JSON.parse(line) as MstarStreamEvent);
    }
  }
  const tail = buffer.trim();
  if (tail) onEvent(JSON.parse(tail) as MstarStreamEvent);
}

// Pattern detection (target-type recognition) - this project's disclosed
// interpretation of THALES's undefined "Pattern detection" requirement.
// Real correlation-filter classifier, real MSTAR train/test chips, real
// accuracy (see apps/api/routers/mstar.py's /pattern* endpoints).
export interface PatternClassInfo {
  class_name: string;
  n_train_samples: number;
}

export interface PatternEvaluationResponse {
  accuracy: number;
  n_test_samples: number;
  confusion: Record<string, Record<string, number>>;
  classes: PatternClassInfo[];
  note: string;
}

export interface PatternResponse {
  chip_id: string;
  actual_class: string;
  predicted_class: string;
  correct: boolean;
  scores: Record<string, number>;
}

// Real Accuracy/IoU/Dice - the one place they exist in this app, because
// it's the one place real ground truth exists (THALES's own worked
// example, quoted directly from their brief, plus small generated cases
// with a known target region) - see apps/api/routers/mstar.py's
// /synthetic-validation endpoint. Never computed against a real MSTAR chip.
export interface SyntheticValidationRequest {
  run_quantum?: boolean;
  qaoa_reps?: number;
  qaoa_max_iterations?: number;
  qaoa_num_shots?: number;
  qaoa_backend?: string;
  only_case?: number;
}

export interface ValidationCaseResult {
  name: string;
  source: string;
  size: number;
  intensity: number[][];
  true_labels: number[][];
  classical: MrfSolverResult; // .metrics holds {accuracy, iou, dice}
  quantum: MrfQuantumResult | null;
}

export interface SyntheticValidationResponse {
  cases: ValidationCaseResult[];
  note: string;
  total_cases: number;
}

export const api = {
  health: () => getJSON<{ status: string }>("/api/health"),

  signup: (username: string, password: string) =>
    postJSON<{ token: string; username: string }>("/api/auth/signup", { username, password }),
  login: (username: string, password: string) =>
    postJSON<{ token: string; username: string }>("/api/auth/login", { username, password }),
  me: () => getJSON<{ username: string; created_at: number }>("/api/auth/me"),
  forgotPassword: (username: string) =>
    postJSON<{ detail: string; dev_reset_link: string | null }>("/api/auth/forgot-password", { username }),
  resetPassword: (token: string, new_password: string) =>
    postJSON<{ token: string; username: string }>("/api/auth/reset-password", { token, new_password }),
  // Full navigation, not a fetch - this starts a real redirect to Auth0's Universal Login page
  // (apps/api/routers/auth.py's auth0_login), not an API call that returns JSON.
  auth0LoginUrl: () => `${API_BASE}/api/auth/auth0/login`,

  mstarChips: () => getJSON<MstarChipInfo[]>("/api/mstar/chips"),
  mstar: (req: MstarRequest) => postJSON<MstarResponse>("/api/mstar", req),
  mstarStream,
  mstarShadow: (req: MstarShadowRequest) => postJSON<MstarShadowResponse>("/api/mstar/shadow", req),
  mstarPattern: (chip_id: string) => postJSON<PatternResponse>("/api/mstar/pattern", { chip_id }),
  mstarPatternEvaluation: () => getJSON<PatternEvaluationResponse>("/api/mstar/pattern/evaluation"),
  mstarSyntheticValidation: (req: SyntheticValidationRequest = {}) =>
    postJSON<SyntheticValidationResponse>("/api/mstar/synthetic-validation", req),
};

export { API_BASE };
