#!/usr/bin/env python3
"""
Build a drive-only graph pickle from an OSM PBF file clipped to a bbox.

The output is a NetworkX MultiDiGraph that can be loaded by this project via:
  meta.graph.pkl_path in input.json

Example:
  python u_bbox_visualisation/build_drive_pkl_from_pbf.py \
    --pbf /data/egypt-latest.osm.pbf \
    --bbox 29.925630 31.229084 30.051972 31.338611 \
    --write-run-cache

Bbox format:
  [min_lat, min_lon, max_lat, max_lon]
"""

import argparse
import datetime
import hashlib
import math
import os
import pickle
import sys

import networkx as nx

try:
    import osmium
except ImportError as exc:
    sys.stderr.write(
        "ERROR: Missing dependency 'osmium'. Install with:\n"
        "  pip install osmium\n"
    )
    raise

# Keep only drive-relevant highway classes.
NON_DRIVABLE_HIGHWAYS = {
    "footway", "path", "cycleway", "steps", "bridleway", "corridor", "pedestrian",
    "proposed", "construction", "platform", "track", "raceway",
}


def _haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters."""
    rlat1 = math.radians(lat1)
    rlon1 = math.radians(lon1)
    rlat2 = math.radians(lat2)
    rlon2 = math.radians(lon2)
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.asin(math.sqrt(a))
    return 6371000.0 * c


def _in_bbox(lat, lon, bbox):
    min_lat, min_lon, max_lat, max_lon = bbox
    return (min_lat <= lat <= max_lat) and (min_lon <= lon <= max_lon)


def _bbox_hash(bbox):
    # Match run_algorithm.setup_graph cache hashing.
    return hashlib.md5(str(list(bbox)).encode()).hexdigest()[:8]


class DriveBBoxHandler(osmium.SimpleHandler):
    def __init__(self, bbox):
        super().__init__()
        self.bbox = tuple(float(v) for v in bbox)
        self.nodes = {}
        self.edges = []
        self.ways_seen = 0
        self.ways_kept = 0

    def way(self, w):
        self.ways_seen += 1

        highway = w.tags.get("highway")
        if not highway:
            return
        if highway in NON_DRIVABLE_HIGHWAYS:
            return
        if w.tags.get("area") == "yes":
            return

        # Basic access filtering.
        access = (w.tags.get("access") or "").lower()
        motor_vehicle = (w.tags.get("motor_vehicle") or "").lower()
        if access in {"no", "private"} or motor_vehicle in {"no", "private"}:
            return

        refs = []
        for n in w.nodes:
            try:
                lat = float(n.location.lat)
                lon = float(n.location.lon)
            except Exception:
                continue
            refs.append((int(n.ref), lat, lon))

        if len(refs) < 2:
            return

        oneway_raw = (w.tags.get("oneway") or "").lower()
        is_roundabout = (w.tags.get("junction") or "").lower() == "roundabout"
        is_oneway = is_roundabout or (oneway_raw in {"yes", "1", "true"}) or (oneway_raw == "-1")
        reverse_only = oneway_raw == "-1"

        common_attrs = {
            "osmid": int(w.id),
            "highway": highway,
            "name": w.tags.get("name"),
            "maxspeed": w.tags.get("maxspeed"),
            "oneway": is_oneway,
        }

        kept_any = False
        for i in range(len(refs) - 1):
            u_id, u_lat, u_lon = refs[i]
            v_id, v_lat, v_lon = refs[i + 1]

            if not (_in_bbox(u_lat, u_lon, self.bbox) and _in_bbox(v_lat, v_lon, self.bbox)):
                continue

            self.nodes[u_id] = {"y": u_lat, "x": u_lon}
            self.nodes[v_id] = {"y": v_lat, "x": v_lon}

            length_m = _haversine_m(u_lat, u_lon, v_lat, v_lon)
            edge_attrs = dict(common_attrs)
            edge_attrs["length"] = float(length_m)

            if reverse_only:
                self.edges.append((v_id, u_id, edge_attrs))
            else:
                self.edges.append((u_id, v_id, edge_attrs))
                if not is_oneway:
                    self.edges.append((v_id, u_id, edge_attrs))

            kept_any = True

        if kept_any:
            self.ways_kept += 1


def build_graph_from_pbf(pbf_path, bbox):
    handler = DriveBBoxHandler(bbox=bbox)
    handler.apply_file(pbf_path, locations=True)

    g = nx.MultiDiGraph()
    g.graph["crs"] = "epsg:4326"
    g.graph["network_type"] = "drive"
    g.graph["simplified"] = False
    g.graph["bbox"] = list(bbox)
    g.graph["source"] = "pbf_bbox_extract"
    g.graph["created_utc"] = datetime.datetime.utcnow().isoformat() + "Z"

    for node_id, data in handler.nodes.items():
        g.add_node(node_id, **data)

    for u, v, attrs in handler.edges:
        g.add_edge(u, v, **attrs)

    return g, handler


def main():
    parser = argparse.ArgumentParser(description="Build bbox-clipped drive graph pickle from OSM PBF")
    parser.add_argument("--pbf", required=True, help="Path to .osm.pbf file")
    parser.add_argument(
        "--bbox",
        required=True,
        nargs=4,
        type=float,
        metavar=("MIN_LAT", "MIN_LON", "MAX_LAT", "MAX_LON"),
        help="Bounding box coordinates in [min_lat min_lon max_lat max_lon]",
    )
    parser.add_argument("--output", default=None, help="Output pickle path")
    parser.add_argument(
        "--write-run-cache",
        action="store_true",
        help="Write to cache/graph_<bbox_hash>.pkl compatible with run_algorithm cache naming",
    )
    parser.add_argument(
        "--payload",
        action="store_true",
        help="Store dict payload {'graph': graph, 'meta': ...} instead of raw graph object",
    )
    args = parser.parse_args()

    pbf_path = os.path.abspath(args.pbf)
    if not os.path.exists(pbf_path):
        raise FileNotFoundError(f"PBF not found: {pbf_path}")

    bbox = tuple(args.bbox)
    if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        raise ValueError("Invalid bbox: expected min_lat < max_lat and min_lon < max_lon")

    if args.write_run_cache:
        os.makedirs("cache", exist_ok=True)
        cache_name = f"graph_{_bbox_hash(bbox)}.pkl"
        output_path = os.path.abspath(os.path.join("cache", cache_name))
    elif args.output:
        output_path = os.path.abspath(args.output)
    else:
        raise ValueError("Provide either --output or --write-run-cache")

    print("Building drive graph from PBF...")
    print(f"  pbf   : {pbf_path}")
    print(f"  bbox  : {list(bbox)}")
    print(f"  out   : {output_path}")

    graph, stats = build_graph_from_pbf(pbf_path=pbf_path, bbox=bbox)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as fh:
        if args.payload:
            payload = {
                "graph": graph,
                "meta": {
                    "bbox": list(bbox),
                    "pbf": pbf_path,
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                    "ways_seen": stats.ways_seen,
                    "ways_kept": stats.ways_kept,
                    "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
                },
            }
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            pickle.dump(graph, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done.")
    print(f"  Nodes: {graph.number_of_nodes():,}")
    print(f"  Edges: {graph.number_of_edges():,}")
    print(f"  Ways seen/kept: {stats.ways_seen:,}/{stats.ways_kept:,}")
    print("\ninput.json snippet:")
    print("{")
    print('  "graph": {')
    print(f'    "bbox": {list(bbox)},')
    print(f'    "pkl_path": "{output_path}",')
    print('    "cache": {"enabled": false}')
    print("  }")
    print("}")


if __name__ == "__main__":
    main()
