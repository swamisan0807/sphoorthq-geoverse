import { useEffect, useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { api, clearSession, getToken, getUsername, setUnauthorizedHandler } from "./api/client";
import LoginPage from "./pages/LoginPage";
import ResetPasswordPage from "./pages/ResetPasswordPage";
import Auth0CallbackPage from "./pages/Auth0CallbackPage";
import DashboardPage from "./pages/DashboardPage";
import CatalogPage from "./pages/CatalogPage";
import InferencePage from "./pages/InferencePage";
import QuantumPage from "./pages/QuantumPage";
import JobsPage from "./pages/JobsPage";
import RegistryPage from "./pages/RegistryPage";
import ComparePage from "./pages/ComparePage";
import GraphPage from "./pages/GraphPage";
import Layout from "./components/Layout";
import "./App.css";

function App() {
  const [authed, setAuthed] = useState(() => getToken() !== null);
  const [username, setUsername] = useState(() => getUsername());
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

  function logout() {
    clearSession();
    setAuthed(false);
  }

  return (
    <Routes>
      <Route
        path="/login"
        element={
          authed ? (
            <Navigate to="/dashboard" replace />
          ) : (
            <LoginPage
              onAuthed={() => {
                setAuthed(true);
                setUsername(getUsername());
              }}
            />
          )
        }
      />
      <Route
        path="/reset-password"
        element={
          <ResetPasswordPage
            onAuthed={() => {
              setAuthed(true);
              setUsername(getUsername());
            }}
          />
        }
      />
      <Route
        path="/auth0/callback"
        element={
          <Auth0CallbackPage
            onAuthed={() => {
              setAuthed(true);
              setUsername(getUsername());
            }}
          />
        }
      />
      <Route
        element={
          authed ? (
            <Layout backendUp={backendUp} username={username} onLogout={logout} />
          ) : (
            <Navigate to="/login" replace />
          )
        }
      >
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/catalog" element={<CatalogPage />} />
        <Route path="/inference" element={<InferencePage />} />
        <Route path="/quantum" element={<QuantumPage />} />
        <Route path="/jobs" element={<JobsPage />} />
        <Route path="/registry" element={<RegistryPage />} />
        <Route path="/compare" element={<ComparePage />} />
        <Route path="/graph" element={<GraphPage />} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Route>
      <Route path="*" element={<Navigate to={authed ? "/dashboard" : "/login"} replace />} />
    </Routes>
  );
}

export default App;
