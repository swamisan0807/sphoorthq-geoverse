import { useEffect, useState } from "react";
import { api, type EventSummary } from "../api/client";
import { useAuthedImage } from "../api/useAuthedImage";

export default function CatalogPage() {
  const [events, setEvents] = useState<EventSummary[] | null>(null);
  const [split, setSplit] = useState("train");
  const [selectedEvent, setSelectedEvent] = useState<string | null>(null);
  const [chipIds, setChipIds] = useState<string[]>([]);
  const [selectedChip, setSelectedChip] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { url: previewUrl, error: previewError } = useAuthedImage(
    selectedChip ? api.chipPreviewPath(selectedChip) : null,
  );

  useEffect(() => {
    api.events().then(setEvents).catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (!selectedEvent) {
      setChipIds([]);
      return;
    }
    api
      .chips(split, selectedEvent)
      .then((r) => {
        setChipIds(r.chip_ids);
        setSelectedChip(null);
      })
      .catch((e) => setError(String(e)));
  }, [split, selectedEvent]);

  return (
    <div className="page">
      <h2>Catalog</h2>
      <p className="hint">
        {events ? `${events.length} flood events, ${events.reduce((a, e) => a + e.total, 0)} hand-labeled chips total (sen1floods11)` : "loading..."}
      </p>
      {error && <p className="error">{error}</p>}

      <table>
        <thead>
          <tr>
            <th>event</th>
            <th>train</th>
            <th>valid</th>
            <th>test</th>
            <th>total</th>
          </tr>
        </thead>
        <tbody>
          {events?.map((e) => (
            <tr
              key={e.event}
              className={`clickable ${selectedEvent === e.event ? "selected" : ""}`}
              onClick={() => setSelectedEvent(e.event)}
            >
              <td>{e.event}</td>
              <td>{e.train}</td>
              <td>{e.valid}</td>
              <td>{e.test}</td>
              <td>{e.total}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {selectedEvent && (
        <section>
          <h3>
            {selectedEvent} chips
            <select value={split} onChange={(ev) => setSplit(ev.target.value)}>
              <option value="train">train</option>
              <option value="valid">valid</option>
              <option value="test">test</option>
            </select>
          </h3>
          <div className="chip-grid">
            {chipIds.map((id) => (
              <button
                key={id}
                className={selectedChip === id ? "selected" : ""}
                onClick={() => setSelectedChip(id)}
              >
                {id}
              </button>
            ))}
          </div>
        </section>
      )}

      {selectedChip && (
        <section>
          <h3>{selectedChip}</h3>
          <p className="hint">left: VV/VH false-color · right: hand-labeled mask (blue = water, gray = nodata)</p>
          {previewError && <p className="error">{previewError}</p>}
          {previewUrl && <img className="preview" src={previewUrl} alt={selectedChip} />}
        </section>
      )}
    </div>
  );
}
