import type { ReactNode } from "react";

export type NavItem = { to: string; label: string; icon: ReactNode };

function Icon({ children }: { children: ReactNode }) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {children}
    </svg>
  );
}

export const NAV_ITEMS: NavItem[] = [
  {
    to: "/dashboard",
    label: "Dashboard",
    icon: (
      <Icon>
        <rect x="1.5" y="1.5" width="5.5" height="5.5" rx="1" />
        <rect x="9" y="1.5" width="5.5" height="5.5" rx="1" />
        <rect x="1.5" y="9" width="5.5" height="5.5" rx="1" />
        <rect x="9" y="9" width="5.5" height="5.5" rx="1" />
      </Icon>
    ),
  },
  {
    to: "/catalog",
    label: "Catalog",
    icon: (
      <Icon>
        <ellipse cx="8" cy="3.2" rx="5.5" ry="1.8" />
        <path d="M2.5 3.2v9.6c0 1 2.46 1.8 5.5 1.8s5.5-.8 5.5-1.8V3.2" />
        <path d="M2.5 8c0 1 2.46 1.8 5.5 1.8s5.5-.8 5.5-1.8" />
      </Icon>
    ),
  },
  {
    to: "/inference",
    label: "Inference",
    icon: (
      <Icon>
        <circle cx="8" cy="8" r="6" />
        <circle cx="8" cy="8" r="2.2" />
      </Icon>
    ),
  },
  {
    to: "/quantum",
    label: "Quantum",
    icon: (
      <Icon>
        <circle cx="8" cy="8" r="1.3" fill="currentColor" stroke="none" />
        <ellipse cx="8" cy="8" rx="6.2" ry="2.6" />
        <ellipse cx="8" cy="8" rx="6.2" ry="2.6" transform="rotate(60 8 8)" />
        <ellipse cx="8" cy="8" rx="6.2" ry="2.6" transform="rotate(120 8 8)" />
      </Icon>
    ),
  },
  {
    to: "/jobs",
    label: "Jobs",
    icon: (
      <Icon>
        <circle cx="8" cy="8" r="6" />
        <path d="M8 4.6V8l2.4 1.4" />
      </Icon>
    ),
  },
  {
    to: "/registry",
    label: "Registry",
    icon: (
      <Icon>
        <path d="M8 1.5l5.5 2.2v4.1c0 3.1-2.2 5-5.5 5.9-3.3-.9-5.5-2.8-5.5-5.9V3.7L8 1.5z" />
        <path d="M5.3 8l1.7 1.7L10.7 6" />
      </Icon>
    ),
  },
  {
    to: "/compare",
    label: "Compare",
    icon: (
      <Icon>
        <rect x="2" y="8" width="3" height="6" />
        <rect x="6.5" y="4" width="3" height="10" />
        <rect x="11" y="6" width="3" height="8" />
      </Icon>
    ),
  },
  {
    to: "/graph",
    label: "Knowledge Graph",
    icon: (
      <Icon>
        <circle cx="4" cy="4" r="1.8" />
        <circle cx="12" cy="4" r="1.8" />
        <circle cx="8" cy="12.5" r="1.8" />
        <path d="M5.6 5.1 6.6 11M10.4 5.1 9.4 11M5.8 4h4.4" />
      </Icon>
    ),
  },
];
