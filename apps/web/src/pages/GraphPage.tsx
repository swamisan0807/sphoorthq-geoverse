import { useEffect, useRef, useState } from "react";
import mermaid from "mermaid";
import { api } from "../api/client";

export default function GraphPage() {
  const [meta, setMeta] = useState<{ n_nodes: number; n_edges: number } | null>(null);
  const [source, setSource] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api
      .graphMermaid()
      .then((r) => {
        setSource(r.mermaid);
        setMeta({ n_nodes: r.n_nodes, n_edges: r.n_edges });
      })
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (!source || !containerRef.current) return;
    const el = containerRef.current;
    mermaid
      .render("knowledge-graph", source)
      .then(({ svg }) => {
        el.innerHTML = svg;
      })
      .catch((e) => {
        el.textContent = `graph render failed: ${e}`;
      });
  }, [source]);

  return (
    <div className="page">
      <h2>Knowledge Graph</h2>
      <p className="hint">
        Built only from relationships the platform actually recorded: dataset -&gt; flood events (real
        catalog), notebook -&gt; registered model version (real registry manifests, with real metrics), and
        event -&gt; quantum run (real, since every quantum job logs the exact chip it used). Nothing here is
        inferred - if a relationship isn't precisely tracked (e.g. exactly which chips a given RF/U-Net
        training run sampled), it isn't drawn.
      </p>
      {error && <p className="error">{error}</p>}
      {meta && (
        <p className="hint">
          {meta.n_nodes} nodes, {meta.n_edges} edges
        </p>
      )}
      <div className="diagram" ref={containerRef} />
    </div>
  );
}
