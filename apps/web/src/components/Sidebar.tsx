import { NavLink } from "react-router-dom";
import { NAV_ITEMS } from "../nav";

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <span className="brand-name">geoverse</span>
        <span className="brand-sub">SAR flood segmentation — classical + quantum ML</span>
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
