import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { useEffect, useRef } from "react";

const BASEMAP_STYLE = "https://demotiles.maplibre.org/style.json";

export default function MapPage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    mapRef.current = new maplibregl.Map({
      container: containerRef.current,
      style: BASEMAP_STYLE,
      center: [67, 45],
      zoom: 3,
    });
    mapRef.current.addControl(new maplibregl.NavigationControl(), "top-right");

    return () => {
      mapRef.current?.remove();
      mapRef.current = null;
    };
  }, []);

  return (
    <>
      <div className="page-title">Map</div>
      <div className="page-subtitle">
        AOI footprints and segmentation results overlay here once{" "}
        <code>/api/tiles</code> is wired to processed COGs.
      </div>
      <div className="map-container" ref={containerRef} />
    </>
  );
}
