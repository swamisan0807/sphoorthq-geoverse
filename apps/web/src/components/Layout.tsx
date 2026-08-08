import { Outlet } from "react-router-dom";
import { API_BASE } from "../api/client";
import Sidebar from "./Sidebar";
import Topbar from "./Topbar";

type Props = {
  backendUp: boolean | null;
  username: string | null;
  onLogout: () => void;
};

export default function Layout({ backendUp, username, onLogout }: Props) {
  return (
    <div className="shell">
      <Sidebar />
      <div className="shell-main">
        <Topbar username={username} onLogout={onLogout} />
        <main className="content">
          {backendUp === false && (
            <p className="error">
              Cannot reach the API{API_BASE ? ` at ${API_BASE}` : ""}. Start it with:
              <br />
              <code>.venv\Scripts\uvicorn apps.api.main:app --reload --port 8000</code>
            </p>
          )}
          <Outlet />
        </main>
      </div>
    </div>
  );
}
