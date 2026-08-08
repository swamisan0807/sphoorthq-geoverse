import { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { setSession } from "../api/client";
import BrandMark from "../components/BrandMark";

/** Lands here after apps/api/routers/auth.py's auth0_callback redirects back
 * from Auth0 with this app's own session token already minted - same query-
 * string-carries-the-token pattern ResetPasswordPage already uses for reset
 * links, just one hop earlier. Stores it exactly like a normal password
 * login (setSession) and moves on - every other page can't tell the
 * difference from here. */
export default function Auth0CallbackPage({ onAuthed }: { onAuthed: () => void }) {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const token = searchParams.get("token");
    const username = searchParams.get("username");
    if (!token || !username) {
      setError("No session token in this link - the Auth0 login flow didn't complete as expected.");
      return;
    }
    setSession(token, username);
    onAuthed();
    navigate("/dashboard", { replace: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (!error) return null;

  return (
    <div className="login-page">
      <div className="login-card">
        <div className="brand-badge">
          <BrandMark size={40} />
        </div>
        <h1>
          Sphoorthi<span className="brand-q">Q</span>
        </h1>
        <p className="error">{error}</p>
        <a className="submit" href="/login">
          Back to log in
        </a>
      </div>
    </div>
  );
}
