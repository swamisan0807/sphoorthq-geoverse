import { NavLink, Route, Routes } from "react-router-dom";

import CatalogPage from "./pages/CatalogPage";
import DashboardPage from "./pages/DashboardPage";
import InferencePage from "./pages/InferencePage";
import MapPage from "./pages/MapPage";

const NAV_ITEMS = [
  { to: "/", label: "Map", end: true },
  { to: "/catalog", label: "Dataset Catalog" },
  { to: "/inference", label: "Run Inference" },
  { to: "/dashboard", label: "Experiments" },
];

export default function App() {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <h1>Geoverse</h1>
        <nav>
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) => (isActive ? "active" : "")}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main className="content">
        <Routes>
          <Route path="/" element={<MapPage />} />
          <Route path="/catalog" element={<CatalogPage />} />
          <Route path="/inference" element={<InferencePage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
        </Routes>
      </main>
    </div>
  );
}
