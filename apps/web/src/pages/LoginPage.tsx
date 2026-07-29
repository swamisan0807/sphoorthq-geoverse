import { useState } from "react";
import { api, setSession } from "../api/client";

export default function LoginPage({ onAuthed }: { onAuthed: () => void }) {
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const resp = mode === "login" ? await api.login(username, password) : await api.signup(username, password);
      setSession(resp.token, resp.username);
      onAuthed();
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="login-page">
      <div className="login-card">
        <img src="/logo.png" alt="company logo" className="login-logo" onError={(e) => (e.currentTarget.style.display = "none")} />
        <h1>geoverse</h1>
        <p className="subtitle">SAR flood segmentation - classical + quantum ML platform</p>

        <div className="login-tabs">
          <button className={mode === "login" ? "active" : ""} onClick={() => setMode("login")} type="button">
            Log in
          </button>
          <button className={mode === "signup" ? "active" : ""} onClick={() => setMode("signup")} type="button">
            Sign up
          </button>
        </div>

        <form onSubmit={submit}>
          <label>
            username
            <input
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
              required
              minLength={3}
            />
          </label>
          <label>
            password
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete={mode === "login" ? "current-password" : "new-password"}
              required
              minLength={8}
            />
          </label>
          {mode === "signup" && <p className="hint">min 8 characters. Any username creates a new account - no invite needed.</p>}
          {error && <p className="error">{error}</p>}
          <button type="submit" disabled={loading} className="submit">
            {loading ? "..." : mode === "login" ? "Log in" : "Create account"}
          </button>
        </form>
      </div>
    </div>
  );
}
