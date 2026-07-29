import { Outlet } from "react-router-dom";
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
        <Topbar backendUp={backendUp} username={username} onLogout={onLogout} />
        <main className="content">
          {backendUp === false && (
            <p className="error">
              Cannot reach the API at {import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000"}. Start it with:
              <br />
              <code>.venv\Scripts\uvicorn src.api.main:app --reload --port 8000</code>
            </p>
          )}
          <Outlet />
        </main>
      </div>
    </div>
  );
}
