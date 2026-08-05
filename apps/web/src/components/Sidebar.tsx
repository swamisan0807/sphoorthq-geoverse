import { NavLink } from "react-router-dom";
import { NAV_ITEMS } from "../nav";
import BrandMark from "./BrandMark";

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-row">
          <BrandMark size={30} />
          <span className="brand-name">
            Sphoorthi<span className="brand-q">Q</span>
          </span>
        </div>
        <span className="brand-sub">SAR Flood Segmentation — Classical + Quantum ML Platform</span>
      </div>
      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item) => (
          <NavLink key={item.to} to={item.to} className={({ isActive }) => (isActive ? "active" : "")}>
            {item.icon}
            {item.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
