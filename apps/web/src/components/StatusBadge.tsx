/** Shared status indicator (stage/run/job status) - icon + label, never
 * color alone: a bare red/green pair fails colorblind-safe separation (run
 * `dataviz` skill's validator on --status-good/--status-critical to see
 * why), so the check/cross glyph carries the distinction, color reinforces
 * it. Terminal states (ok/success, failed) get the reserved status colors;
 * in-flight states (running, queued) get neutral/brand colors since they
 * aren't a good/bad outcome yet. */
function CheckIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M3 8.5 L6.5 12 L13 4" />
    </svg>
  );
}

function CrossIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
      <path d="M4 4 L12 12 M12 4 L4 12" />
    </svg>
  );
}

export default function StatusBadge({ status }: { status: string }) {
  const s = status.toLowerCase();
  const isGood = s === "ok" || s === "success";
  const isCritical = s === "failed" || s === "error";
  const isRunning = s === "running";
  const kind = isGood ? "good" : isCritical ? "critical" : isRunning ? "running" : "queued";

  return (
    <span className={`status-badge status-${kind}`}>
      {isGood && <CheckIcon />}
      {isCritical && <CrossIcon />}
      {!isGood && !isCritical && <span className={`status-dot${isRunning ? " pulse" : ""}`} />}
      {status}
    </span>
  );
}
