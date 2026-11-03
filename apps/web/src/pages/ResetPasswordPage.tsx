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
      setError("Those two passwords don't match - try typing them again.");
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
            This reset link looks incomplete. Use "Forgot password?" on the login page to get a fresh one.
          </p>
        ) : (
          <form onSubmit={submit}>
            <label>
              New password
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
              Confirm new password
              <input
                type="password"
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                autoComplete="new-password"
                required
                minLength={8}
              />
            </label>
            <p className="hint">At least 8 characters. This link expires 15 minutes after you asked for it.</p>
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
