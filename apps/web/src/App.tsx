import { useEffect, useState } from "react";
import { api, clearSession, getToken, getUsername, setUnauthorizedHandler } from "./api/client";
import LoginPage from "./pages/LoginPage";
import DashboardPage from "./pages/DashboardPage";
import CatalogPage from "./pages/CatalogPage";
import InferencePage from "./pages/InferencePage";
import QuantumPage from "./pages/QuantumPage";
import JobsPage from "./pages/JobsPage";
import RegistryPage from "./pages/RegistryPage";
import ComparePage from "./pages/ComparePage";
import GraphPage from "./pages/GraphPage";
import "./App.css";

type Tab = "dashboard" | "catalog" | "inference" | "quantum" | "jobs" | "registry" | "compare" | "graph";

function App() {
  const [authed, setAuthed] = useState(() => getToken() !== null);
  const [username, setUsername] = useState(() => getUsername());
  const [tab, setTab] = useState<Tab>("dashboard");
  const [backendUp, setBackendUp] = useState<boolean | null>(null);

  useEffect(() => {
    api
      .health()
      .then(() => setBackendUp(true))
      .catch(() => setBackendUp(false));
  }, []);

  useEffect(() => {
    setUnauthorizedHandler(() => setAuthed(false));
  }, []);

  if (!authed) {
    return (
      <LoginPage
        onAuthed={() => {
          setAuthed(true);
          setUsername(getUsername());
        }}
      />
    );
  }

  function logout() {
    clearSession();
    setAuthed(false);
  }

  return (
    <div className="app">
      <header>
        <h1>geoverse</h1>
        <span className="subtitle">SAR flood segmentation - classical + quantum ML hybrid</span>
        <span className={`status ${backendUp ? "up" : "down"}`}>
          {backendUp === null ? "checking API..." : backendUp ? "API connected" : "API unreachable"}
        </span>
        <span className="user-info">
          {username} <button className="logout" onClick={logout}>log out</button>
        </span>
      </header>

      <nav>
        <button className={tab === "dashboard" ? "active" : ""} onClick={() => setTab("dashboard")}>
          Dashboard
        </button>
        <button className={tab === "catalog" ? "active" : ""} onClick={() => setTab("catalog")}>
          Catalog
        </button>
        <button className={tab === "inference" ? "active" : ""} onClick={() => setTab("inference")}>
          Inference
        </button>
        <button className={tab === "quantum" ? "active" : ""} onClick={() => setTab("quantum")}>
          Quantum
        </button>
        <button className={tab === "jobs" ? "active" : ""} onClick={() => setTab("jobs")}>
          Jobs
        </button>
        <button className={tab === "registry" ? "active" : ""} onClick={() => setTab("registry")}>
          Registry
        </button>
        <button className={tab === "compare" ? "active" : ""} onClick={() => setTab("compare")}>
          Compare
        </button>
        <button className={tab === "graph" ? "active" : ""} onClick={() => setTab("graph")}>
          Knowledge Graph
        </button>
      </nav>

      <main>
        {!backendUp && backendUp !== null && (
          <p className="error">
            Cannot reach the API at {import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000"}. Start it with:
            <br />
            <code>.venv\Scripts\uvicorn src.api.main:app --reload --port 8000</code>
          </p>
        )}
        {tab === "dashboard" && <DashboardPage />}
        {tab === "catalog" && <CatalogPage />}
        {tab === "inference" && <InferencePage />}
        {tab === "quantum" && <QuantumPage />}
        {tab === "jobs" && <JobsPage />}
        {tab === "registry" && <RegistryPage />}
        {tab === "compare" && <ComparePage />}
        {tab === "graph" && <GraphPage />}
      </main>
    </div>
  );
}

export default App;
