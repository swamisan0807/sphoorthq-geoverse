import { NavLink, Outlet } from "react-router-dom";
import { API_BASE } from "../api/client";
import BrandMark from "./BrandMark";

type Props = {
  backendUp: boolean | null;
  username: string | null;
  onLogout: () => void;
  showcase?: boolean;
};

export default function Layout({ backendUp, username, onLogout, showcase = false }: Props) {
  return (
    <div className="shell-single">
      <header className="topbar-single">
        <div className="topbar-brand">
          <BrandMark size={30} />
          <div>
            <span className="brand-name">
              Sphoorthi<span className="brand-q">Q</span>
            </span>
            <span className="brand-sub">Quantum Optimization for SAR Vehicle Segmentation</span>
          </div>
        </div>
        <nav className="topbar-nav">
          {!showcase && (
            <NavLink to="/" end className={({ isActive }) => (isActive ? "active" : "")}>
              Home
            </NavLink>
          )}
          <NavLink to="/about" className={({ isActive }) => (isActive ? "active" : "")}>
            About
          </NavLink>
        </nav>
        <span className="user-info">
          {showcase ? (
            <NavLink to="/login">Sign in</NavLink>
          ) : (
            <>
              {username}{" "}
              <button className="logout" onClick={onLogout}>
                log out
              </button>
            </>
          )}
        </span>
      </header>
      <main className="content">
        {showcase && (
          <p className="hint">
            This public copy shows the About page, which holds the measured results. The interactive tools run on the
            project's own server and need a sign-in.
          </p>
        )}
        {backendUp === false && (
          <p className="error">
            We can't reach the server{API_BASE ? ` at ${API_BASE}` : ""}. Start it with:
            <br />
            <code>.venv\Scripts\uvicorn apps.api.main:app --reload --port 8000</code>
          </p>
        )}
        <Outlet />
      </main>
    </div>
  );
}
