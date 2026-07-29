import { useLocation } from "react-router-dom";
import { NAV_ITEMS } from "../nav";

type Props = {
  backendUp: boolean | null;
  username: string | null;
  onLogout: () => void;
};

export default function Topbar({ backendUp, username, onLogout }: Props) {
  const { pathname } = useLocation();
  const current = NAV_ITEMS.find((item) => pathname.startsWith(item.to));

  return (
    <header className="topbar">
      <h1>{current?.label ?? "geoverse"}</h1>
      <span className={`status ${backendUp ? "up" : "down"}`}>
        {backendUp === null ? "checking API..." : backendUp ? "API connected" : "API unreachable"}
      </span>
      <span className="user-info">
        {username}{" "}
        <button className="logout" onClick={onLogout}>
          log out
        </button>
      </span>
    </header>
  );
}
