import { useState } from "react";
import { api, setSession } from "../api/client";
import BrandMark from "../components/BrandMark";

export default function LoginPage({ onAuthed }: { onAuthed: () => void }) {
  const [mode, setMode] = useState<"login" | "signup" | "forgot">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [forgotResult, setForgotResult] = useState<{ detail: string; dev_reset_link: string | null } | null>(null);
  // apps/web/public/logo.png doesn't exist yet - falls back to the
  // recreated BrandMark below until the real SphoorthiQ asset is dropped
  // in (see the img's onError). No code change needed when it lands.
  const [logoMissing, setLogoMissing] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      if (mode === "forgot") {
        const resp = await api.forgotPassword(username);
        setForgotResult(resp);
        return;
      }
      const resp = mode === "login" ? await api.login(username, password) : await api.signup(username, password);
      setSession(resp.token, resp.username);
      onAuthed();
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  function switchMode(next: "login" | "signup" | "forgot") {
    setMode(next);
    setError(null);
    setForgotResult(null);
  }

  return (
    <div className="login-page">
      <div className="login-card">
        <div className="brand-badge">
          {logoMissing ? (
            <BrandMark size={40} />
          ) : (
            <img src="/logo.png" alt="SphoorthiQ logo" className="login-logo" onError={() => setLogoMissing(true)} />
          )}
        </div>
        <h1>
          Sphoorthi<span className="brand-q">Q</span>
        </h1>
        <p className="subtitle">SAR Flood Segmentation — Classical + Quantum ML Platform</p>

        {mode !== "forgot" && (
          <div className="login-tabs">
            <button className={mode === "login" ? "active" : ""} onClick={() => switchMode("login")} type="button">
              Log in
            </button>
            <button className={mode === "signup" ? "active" : ""} onClick={() => switchMode("signup")} type="button">
              Sign up
            </button>
          </div>
        )}

        {mode === "forgot" && forgotResult ? (
          <div className="forgot-result">
            <p>{forgotResult.detail}</p>
            {forgotResult.dev_reset_link && (
              <>
                <p className="hint">
                  SMTP isn't configured on this server (see <code>config/platform.yaml</code>: <code>SMTP_HOST</code>/
                  <code>SMTP_USER</code>/<code>SMTP_PASSWORD</code>), so no email was actually sent. Local/dev link
                  instead:
                </p>
                <a className="dev-reset-link" href={forgotResult.dev_reset_link}>
                  {forgotResult.dev_reset_link}
                </a>
              </>
            )}
            <button type="button" className="submit" onClick={() => switchMode("login")}>
              Back to log in
            </button>
          </div>
        ) : (
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
            {mode !== "forgot" && (
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
            )}
            {mode === "signup" && <p className="hint">min 8 characters. Any username creates a new account - no invite needed.</p>}
            {mode === "forgot" && <p className="hint">enter the username (or email) you signed up with - we'll send a reset link.</p>}
            {mode === "login" && (
              <button type="button" className="link-button" onClick={() => switchMode("forgot")}>
                Forgot password?
              </button>
            )}
            {error && <p className="error">{error}</p>}
            <button type="submit" disabled={loading} className="submit">
              {loading ? "..." : mode === "login" ? "Log in" : mode === "signup" ? "Create account" : "Send reset link"}
            </button>
            {mode === "forgot" && (
              <button type="button" className="link-button" onClick={() => switchMode("login")}>
                Back to log in
              </button>
            )}
          </form>
        )}

        {mode !== "forgot" && (
          <>
            <div className="login-divider">or</div>
            <a className="submit auth0-button" href={api.auth0LoginUrl()}>
              Log in with Auth0
            </a>
          </>
        )}
      </div>
    </div>
  );
}
