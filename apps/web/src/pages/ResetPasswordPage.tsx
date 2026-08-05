import { useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api, setSession } from "../api/client";
import BrandMark from "../components/BrandMark";

export default function ResetPasswordPage({ onAuthed }: { onAuthed: () => void }) {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const token = searchParams.get("token") ?? "";
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [logoMissing, setLogoMissing] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (password !== confirm) {
      setError("passwords don't match");
      return;
    }
    setLoading(true);
    try {
      const resp = await api.resetPassword(token, password);
      setSession(resp.token, resp.username);
      onAuthed();
      navigate("/dashboard", { replace: true });
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
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
        <p className="subtitle">Reset your password</p>

        {!token ? (
          <p className="error">
            No reset token in this link. Use the "Forgot password?" link on the login page to request a new one.
          </p>
        ) : (
          <form onSubmit={submit}>
            <label>
              new password
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete="new-password"
                required
                minLength={8}
              />
            </label>
            <label>
              confirm password
              <input
                type="password"
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                autoComplete="new-password"
                required
                minLength={8}
              />
            </label>
            <p className="hint">min 8 characters. This link expires 15 minutes after it was requested.</p>
            {error && <p className="error">{error}</p>}
            <button type="submit" disabled={loading} className="submit">
              {loading ? "..." : "Reset password"}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
