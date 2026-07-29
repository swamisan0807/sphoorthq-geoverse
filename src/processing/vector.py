"""OSM extract -> rasterized weak labels (water, buildings, roads) for validation."""

import geopandas as gpd
import numpy as np
import osmium
import shapely.wkb as swkb
from rasterio.features import rasterize
from shapely.geometry import Polygon

WATER_TAGS = {"natural": ["water"], "waterway": ["riverbank"], "landuse": ["reservoir"]}
BUILDING_TAGS = {"building": None}
ROAD_TAGS = {"highway": None}


class _TagFilterHandler(osmium.SimpleHandler):
    """Streams an .osm.pbf and collects geometries matching a tag filter.

    as_lines=False collects closed ways as polygons (water/buildings);
    as_lines=True collects ways as linestrings (roads). A single `way`
    callback branches on this flag - pyosmium's apply_file resolves
    callbacks by inspecting the handler's class, so the branch has to live
    inside one real method rather than being swapped in after construction.
    Kept intentionally simple (no osmium.geom area assembler for
    multipolygons) - sufficient for weak-label rasterization, not exact
    cadastral geometry.
    """

    def __init__(self, tag_filter: dict[str, list[str] | None], as_lines: bool = False):
        super().__init__()
        self.tag_filter = tag_filter
        self.as_lines = as_lines
        self.geometries: list = []
        self._wkbfab = osmium.geom.WKBFactory()

    def _matches(self, tags) -> bool:
        for key, allowed_values in self.tag_filter.items():
            if key in tags:
                if allowed_values is None or tags[key] in allowed_values:
                    return True
        return False

    def way(self, w):
        if not self._matches(w.tags):
            return
        if not self.as_lines and not w.is_closed():
            return
        try:
            wkb = self._wkbfab.create_linestring(w)
        except RuntimeError:
            return

        line = swkb.loads(wkb, hex=True)
        if self.as_lines:
            self.geometries.append(line)
        elif len(line.coords) >= 4:
            self.geometries.append(Polygon(line.coords))


def extract_geometries(pbf_path: str, tag_filter: dict, as_lines: bool = False) -> gpd.GeoDataFrame:
    handler = _TagFilterHandler(tag_filter, as_lines=as_lines)
    handler.apply_file(pbf_path, locations=True)
    return gpd.GeoDataFrame(geometry=handler.geometries, crs="EPSG:4326")


def rasterize_layer(
    gdf: gpd.GeoDataFrame, out_shape: tuple[int, int], transform, fill: int = 0, value: int = 1
) -> np.ndarray:
    if gdf.empty:
        return np.full(out_shape, fill, dtype=np.uint8)
    shapes = ((geom, value) for geom in gdf.geometry if geom is not None and not geom.is_empty)
    return rasterize(
        shapes=shapes, out_shape=out_shape, transform=transform, fill=fill, dtype="uint8"
    )


def build_weak_labels(
    pbf_path: str, out_shape: tuple[int, int], transform
) -> dict[str, np.ndarray]:
    water = extract_geometries(pbf_path, WATER_TAGS)
    buildings = extract_geometries(pbf_path, BUILDING_TAGS)
    roads = extract_geometries(pbf_path, ROAD_TAGS, as_lines=True)

    return {
        "water": rasterize_layer(water, out_shape, transform),
        "buildings": rasterize_layer(buildings, out_shape, transform),
        "roads": rasterize_layer(roads, out_shape, transform),
    }
