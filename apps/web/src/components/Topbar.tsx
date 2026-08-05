import { useLocation } from "react-router-dom";
import { NAV_ITEMS } from "../nav";

type Props = {
  username: string | null;
  onLogout: () => void;
};

export default function Topbar({ username, onLogout }: Props) {
  const { pathname } = useLocation();
  const current = NAV_ITEMS.find((item) => pathname.startsWith(item.to));

  return (
    <header className="topbar">
      <h1>{current?.label ?? "SphoorthiQ"}</h1>
      <span className="user-info">
        {username}{" "}
        <button className="logout" onClick={onLogout}>
          log out
        </button>
      </span>
    </header>
  );
}
