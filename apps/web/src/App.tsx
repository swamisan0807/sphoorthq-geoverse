import { useEffect, useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { api, clearSession, getToken, getUsername, setUnauthorizedHandler } from "./api/client";
import LoginPage from "./pages/LoginPage";
import ResetPasswordPage from "./pages/ResetPasswordPage";
import Auth0CallbackPage from "./pages/Auth0CallbackPage";
import MrfPage from "./pages/MrfPage";
import AboutPage from "./pages/AboutPage";
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
            <Navigate to="/" replace />
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
        <Route path="/" element={<MrfPage />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
      <Route path="*" element={<Navigate to={authed ? "/" : "/login"} replace />} />
    </Routes>
  );
}

export default App;
