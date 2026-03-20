"""
Fast isolated preview for named-opposite secondary/tertiary synthetic crossings.

This script only builds drive + walk graphs, synthesizes crossings, and writes
an HTML map. It skips ALNS and route solving for rapid iteration.

Usage:
  source .venv/bin/activate && python3 experiments/experiment4_crossings/preview_named_crossings_2.py
  source .venv/bin/activate && python3 experiments/experiment4_crossings/preview_named_crossings_2.py --input experiments/experiment4_crossings/input.json
  source .venv/bin/activate && python3 experiments/experiment4_crossings/preview_named_crossings_2.py --debug
"""

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys

import folium
from folium import FeatureGroup

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR, os.pardir, os.pardir))
_COMPARISON_DIR = os.path.join(_ROOT, "experiments", "comparison")

for _p in (_ROOT, _COMPARISON_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import detour_engine as _eng
from run_algorithm import setup_graph, setup_walk_graph


def _strip_comments(obj):
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items() if k != "_comment"}
    if isinstance(obj, list):
        return [_strip_comments(v) for v in obj]
    return obj


def _input_hash(input_path: str) -> str:
    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)
    canonical = json.dumps(_strip_comments(raw), sort_keys=True, separators=(",", ":"))
    return hashlib.md5(canonical.encode()).hexdigest()[:8]


def _extract_walk_segments(walk_graph):
    segs = []
    for u, v, k, data in walk_graph.edges(keys=True, data=True):
        if data.get("synthetic_crossing", False):
            continue
        if "geometry" in data:
            coords = [(lat, lon) for lon, lat in data["geometry"].coords]
        else:
            coords = [
                (walk_graph.nodes[u]["y"], walk_graph.nodes[u]["x"]),
                (walk_graph.nodes[v]["y"], walk_graph.nodes[v]["x"]),
            ]
        segs.append(coords)
    return segs


def _extract_synth_segments(walk_graph):
    segs = []
    for u, v, k, data in walk_graph.edges(keys=True, data=True):
        if not data.get("synthetic_crossing", False):
            continue
        if "geometry" in data:
            coords = [(lat, lon) for lon, lat in data["geometry"].coords]
        else:
            coords = [
                (walk_graph.nodes[u]["y"], walk_graph.nodes[u]["x"]),
                (walk_graph.nodes[v]["y"], walk_graph.nodes[v]["x"]),
            ]
        segs.append(coords)
    return segs


def _extract_secondary_tertiary_drive_nodes_in_bbox(drive_graph, bbox=None):
    """Extract unique drive nodes from secondary/tertiary road edges within bounding box.

    Args:
        drive_graph: The drive graph from OSMnx
        bbox: Bounding box as [west, south, east, north]. If None, uses default bbox.

    Returns:
        List of node dictionaries with lat, lon, highway, road_name, node_id
    """
    # Default bbox from run_algorithm.py:78
    if bbox is None:
        bbox = [31.229084, 29.925630, 31.331909, 29.991682]

    west, south, east, north = bbox[0], bbox[1], bbox[2], bbox[3]
    allowed_hw = {"secondary", "tertiary"}
    nodes = {}  # node_id -> {lat, lon, highway, road_name}

    for u, v, key, data in drive_graph.edges(keys=True, data=True):
        hw = data.get("highway", "")
        if isinstance(hw, list):
            hw = hw[0] if hw else ""
        hw = str(hw).lower().strip()

        if hw not in allowed_hw:
            continue

        road_name = data.get("name", f"Unnamed {hw}")
        if isinstance(road_name, list):
            road_name = road_name[0] if road_name else f"Unnamed {hw}"
        road_name = str(road_name).strip()

        # Add both endpoint nodes if they're within the bounding box
        for node_id in [u, v]:
            if node_id not in nodes:
                try:
                    lat = drive_graph.nodes[node_id]['y']
                    lon = drive_graph.nodes[node_id]['x']

                    # Check if node is within bounding box
                    if west <= lon <= east and south <= lat <= north:
                        nodes[node_id] = {
                            'lat': lat,
                            'lon': lon,
                            'highway': hw,
                            'road_name': road_name,
                            'node_id': node_id
                        }
                except KeyError:
                    continue  # Skip nodes without coordinates

    return list(nodes.values())


def _build_crossings_injection_payload(crossings, synth_cfg, input_path, run_dir):
    nodes_by_id = {}
    edge_pairs = []

    def _upsert_node(node_id, lat, lon, node_kind):
        key = node_id
        existing = nodes_by_id.get(key)
        if existing is None:
            nodes_by_id[key] = {
                "node_id": node_id,
                "lat": float(lat),
                "lon": float(lon),
                "node_kind": node_kind,
            }
            return
        if existing.get("node_kind") != "synthetic" and node_kind == "synthetic":
            existing["node_kind"] = "synthetic"

    for crossing in crossings:
        node_a = crossing.get("node_a")
        node_b = crossing.get("node_b")
        if node_a is None or node_b is None:
            continue

        crossing_type = crossing.get("crossing_type", "real_to_real")
        kind_a = "real"
        kind_b = "synthetic" if crossing_type == "real_to_synthetic" else "real"

        _upsert_node(node_a, crossing["lat_a"], crossing["lon_a"], kind_a)
        _upsert_node(node_b, crossing["lat_b"], crossing["lon_b"], kind_b)

        edge_pairs.append({
            "node_a": node_a,
            "node_b": node_b,
            "length_m": float(crossing.get("length_m", 0.0)),
            "crossing_type": crossing_type,
            "road_name": crossing.get("road_name", "?"),
        })

    payload = {
        "metadata": {
            "source": "preview_named_crossings_2",
            "input_path": input_path,
            "run_dir": run_dir,
            "strategy": str(synth_cfg.get("strategy", "named_opposite_secondary_tertiary")),
            "min_dist_m": float(synth_cfg.get("min_dist_m", 0.0)),
            "max_dist_m": float(synth_cfg.get("max_dist_m", 60.0)),
            "min_opposite_bearing_deg": float(synth_cfg.get("min_opposite_bearing_deg", 150.0)),
            "max_crossing_angle_delta_deg": float(synth_cfg.get("max_crossing_angle_delta_deg", 30.0)),
            "min_spacing_m": float(synth_cfg.get("min_spacing_m", 100.0)),
            "min_spacing_per_road_m": float(synth_cfg.get("min_spacing_per_road_m", 100.0)),
            "guaranteed_connection_distance_m": float(synth_cfg.get("guaranteed_connection_distance_m", 6.0)),
            "fallback_mode": str(synth_cfg.get("fallback_mode", "adaptive")),
            "node_count": len(nodes_by_id),
            "edge_count": len(edge_pairs),
        },
        "nodes": list(nodes_by_id.values()),
        "crossings": [
            {
                "node_a": c.get("node_a"),
                "node_b": c.get("node_b"),
                "lat_a": float(c.get("lat_a", 0.0)),
                "lon_a": float(c.get("lon_a", 0.0)),
                "lat_b": float(c.get("lat_b", 0.0)),
                "lon_b": float(c.get("lon_b", 0.0)),
                "length_m": float(c.get("length_m", 0.0)),
                "crossing_type": c.get("crossing_type", "real_to_real"),
                "road_name": c.get("road_name", "?"),
            }
            for c in crossings
            if c.get("node_a") is not None and c.get("node_b") is not None
        ],
        "edge_pairs": edge_pairs,
    }
    return payload


def main():
    parser = argparse.ArgumentParser(description="Preview named-opposite crossings only")
    parser.add_argument("--input", default=os.path.join(_DIR, "input.json"), help="Path to input JSON")
    parser.add_argument("--debug", action="store_true", help="Show drive edges used for crossing generation")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        sys.exit(f"[ERROR] config file not found: {input_path}")

    with open(input_path, encoding="utf-8") as f:
        meta = json.load(f)

    school = meta.get("school", {})
    center = (float(school["latitude"]), float(school["longitude"]))
    walk_cfg = meta.get("walk_graph", {})
    synth_cfg = meta.get("synthetic_crossings", {})
    if not synth_cfg:
        synth_cfg = {"enabled": True, "strategy": "named_opposite_secondary_tertiary"}

    # Generate unique folder name with timestamp
    import time
    ts = int(time.time())
    h = _input_hash(input_path)
    run_dir = os.path.join(_DIR, f"preview_{h}_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    # Save snapshot of input config for this run
    input_snapshot = os.path.join(run_dir, "input_snapshot.json")
    shutil.copy(input_path, input_snapshot)

    out_html = os.path.join(run_dir, "named_crossings_preview.html")
    out_json = os.path.join(run_dir, "named_crossings_diagnostics.json")
    out_pkl = os.path.join(run_dir, "crossings_nodes_injection.pkl")

    print("Preview named-opposite crossings")
    print(f"  Config hash : {h}")
    print(f"  Run folder  : {run_dir}")
    if args.debug:
        print("  Debug mode  : ON (showing drive edges)")

    print("[1/3] Loading constrained drive graph...")
    G_drive = setup_graph(meta, unconstrained=False)

    print("[2/3] Loading walk graph...")
    walk_radius_km = float(walk_cfg.get("radius_km", 2.0))
    G_walk = setup_walk_graph(meta, center=center, radius_m=walk_radius_km * 1000.0)

    print("[3/3] Building drive node crossings...")
    crossings, diag, debug_info = _eng.build_drive_node_crossings(
        walk_graph=G_walk,
        drive_graph=G_drive,
        min_dist_m=float(synth_cfg.get("min_dist_m", 6.0)),
        max_dist_m=float(synth_cfg.get("max_dist_m", 60.0)),
        min_opposite_bearing_deg=float(synth_cfg.get("min_opposite_bearing_deg", 150.0)),
        max_crossing_angle_delta_deg=float(synth_cfg.get("max_crossing_angle_delta_deg", 30.0)),
        min_perpendicular_deg=(
            float(synth_cfg["min_perpendicular_deg"])
            if synth_cfg.get("min_perpendicular_deg") is not None
            else None
        ),
        max_perpendicular_deg=(
            float(synth_cfg["max_perpendicular_deg"])
            if synth_cfg.get("max_perpendicular_deg") is not None
            else None
        ),
        min_spacing_m=float(synth_cfg.get("min_spacing_m", 100.0)),
        min_spacing_per_road_m=float(synth_cfg.get("min_spacing_per_road_m", 100.0)),
        center_lat=center[0],
        center_lon=center[1],
        radius_km=walk_radius_km,
        enable_guaranteed_connection=bool(synth_cfg.get("enable_guaranteed_connection", True)),
        guaranteed_connection_distance_m=float(synth_cfg.get("guaranteed_connection_distance_m", 6.0)),
        fallback_mode=str(synth_cfg.get("fallback_mode", "adaptive")),
    )

    # Extract drive nodes and synthetic nodes from debug info
    drive_nodes_a = [n for n in debug_info.get("drive_nodes", []) if n["direction"] == "A"]
    drive_nodes_b = [n for n in debug_info.get("drive_nodes", []) if n["direction"] == "B"]
    synthetic_nodes_a = [n for n in debug_info.get("synthetic_nodes", []) if n["direction"] == "A"]
    synthetic_nodes_b = [n for n in debug_info.get("synthetic_nodes", []) if n["direction"] == "B"]
    real_crossings = [c for c in crossings if c.get("crossing_type") == "real_to_real"]
    synth_crossings = [c for c in crossings if c.get("crossing_type") == "real_to_synthetic"]

    # Extract secondary/tertiary drive nodes for visualization (filtered by default bbox)
    drive_nodes_sec_tert = _extract_secondary_tertiary_drive_nodes_in_bbox(G_drive)

    injection_payload = _build_crossings_injection_payload(
        crossings=crossings,
        synth_cfg=synth_cfg,
        input_path=input_path,
        run_dir=run_dir,
    )
    with open(out_pkl, "wb") as f:
        pickle.dump(injection_payload, f)

    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")
    folium.Marker(location=center, popup="School", tooltip="School").add_to(m)

    fg_walk = FeatureGroup(name="Walk Network", show=False)
    for seg in _extract_walk_segments(G_walk):
        folium.PolyLine(seg, color="#4aa3df", weight=1.5, opacity=0.5).add_to(fg_walk)
    fg_walk.add_to(m)

    # Real crossings (gold)
    fg_real = FeatureGroup(name=f"Real Crossings (gold) [{len(real_crossings)}]", show=True)
    for i, crossing in enumerate(real_crossings):
        lat_a, lon_a = crossing["lat_a"], crossing["lon_a"]
        lat_b, lon_b = crossing["lat_b"], crossing["lon_b"]
        road_name = crossing.get("road_name", "?")
        dist = crossing.get("length_m", 0)
        tooltip = f"Real Crossing #{i+1}<br>Road: {road_name}<br>Distance: {dist:.1f}m"
        folium.PolyLine(
            [(lat_a, lon_a), (lat_b, lon_b)],
            color="gold", weight=4, opacity=0.9, tooltip=tooltip
        ).add_to(fg_real)
        # Add numbered marker at midpoint
        mid_lat = (lat_a + lat_b) / 2
        mid_lon = (lon_a + lon_b) / 2
        folium.Marker(
            location=(mid_lat, mid_lon),
            icon=folium.DivIcon(
                html=f'<div style="font-size:10px;font-weight:bold;color:#c00;background:white;padding:1px 3px;border-radius:3px;border:1px solid #c00;">{i+1}</div>',
                icon_anchor=(8, 8),
            ),
        ).add_to(fg_real)
    fg_real.add_to(m)

    # Synthetic crossings (darkorange)
    fg_synth = FeatureGroup(name=f"Synthetic Crossings (darkorange) [{len(synth_crossings)}]", show=True)
    for i, crossing in enumerate(synth_crossings):
        lat_a, lon_a = crossing["lat_a"], crossing["lon_a"]
        lat_b, lon_b = crossing["lat_b"], crossing["lon_b"]
        road_name = crossing.get("road_name", "?")
        dist = crossing.get("length_m", 0)
        tooltip = f"Synthetic Crossing #{i+1}<br>Road: {road_name}<br>Distance: {dist:.1f}m"
        folium.PolyLine(
            [(lat_a, lon_a), (lat_b, lon_b)],
            color="darkorange", weight=4, opacity=0.9, tooltip=tooltip
        ).add_to(fg_synth)
        # Add numbered marker at midpoint
        mid_lat = (lat_a + lat_b) / 2
        mid_lon = (lon_a + lon_b) / 2
        folium.Marker(
            location=(mid_lat, mid_lon),
            icon=folium.DivIcon(
                html=f'<div style="font-size:10px;font-weight:bold;color:#ff6600;background:white;padding:1px 3px;border-radius:3px;border:1px solid #ff6600;">{i+1+len(real_crossings)}</div>',
                icon_anchor=(8, 8),
            ),
        ).add_to(fg_synth)
    fg_synth.add_to(m)

    # Debug: show drive edges, walk nodes, and synthetic nodes
    if args.debug and debug_info:
        edges = debug_info.get("edges", [])
        walk_nodes = debug_info.get("walk_nodes", [])
        synthetic_nodes = debug_info.get("synthetic_nodes", [])

        fg_edges_a = FeatureGroup(name="Drive Edges A (blue)", show=True)
        fg_edges_b = FeatureGroup(name="Drive Edges B (orange)", show=True)

        for edge in edges:
            u_lat, u_lon = edge["u_lat"], edge["u_lon"]
            v_lat, v_lon = edge["v_lat"], edge["v_lon"]
            direction = edge.get("direction", "?")
            name = edge.get("name", "?")
            hw = edge.get("highway", "?")
            bearing = edge.get("bearing", 0)

            tooltip = f"{name} ({hw})<br>bearing: {bearing:.0f}°<br>dir: {direction}"

            if direction == "A":
                folium.PolyLine(
                    [(u_lat, u_lon), (v_lat, v_lon)],
                    color="blue", weight=3, opacity=0.8, tooltip=tooltip
                ).add_to(fg_edges_a)
            else:
                folium.PolyLine(
                    [(u_lat, u_lon), (v_lat, v_lon)],
                    color="orange", weight=3, opacity=0.8, tooltip=tooltip
                ).add_to(fg_edges_b)

        fg_edges_a.add_to(m)
        fg_edges_b.add_to(m)

        # Show drive nodes used in crossings
        fg_dnodes_a = FeatureGroup(name=f"Drive Nodes A (blue) [{len(drive_nodes_a)}]", show=True)
        fg_dnodes_b = FeatureGroup(name=f"Drive Nodes B (orange) [{len(drive_nodes_b)}]", show=True)

        for dn in drive_nodes_a:
            lat, lon = dn["lat"], dn["lon"]
            road_name = dn.get("road_name", "?")
            node_id = str(dn.get("node_id", "?"))[:10]
            highway = dn.get("highway", "?")

            tooltip = f"Drive Node A<br>ID: {node_id}<br>Road: {road_name}<br>Type: {highway}"

            folium.CircleMarker(
                location=(lat, lon),
                radius=5,
                color="blue",
                fill=True,
                fillColor="blue",
                fillOpacity=0.7,
                tooltip=tooltip
            ).add_to(fg_dnodes_a)

        for dn in drive_nodes_b:
            lat, lon = dn["lat"], dn["lon"]
            road_name = dn.get("road_name", "?")
            node_id = str(dn.get("node_id", "?"))[:10]
            highway = dn.get("highway", "?")

            tooltip = f"Drive Node B<br>ID: {node_id}<br>Road: {road_name}<br>Type: {highway}"

            folium.CircleMarker(
                location=(lat, lon),
                radius=5,
                color="orange",
                fill=True,
                fillColor="orange",
                fillOpacity=0.7,
                tooltip=tooltip
            ).add_to(fg_dnodes_b)

        fg_dnodes_a.add_to(m)
        fg_dnodes_b.add_to(m)

        # Show synthetic nodes
        fg_synth_nodes_a = FeatureGroup(name=f"Synthetic Nodes A (purple) [{len(synthetic_nodes_a)}]", show=True)
        fg_synth_nodes_b = FeatureGroup(name=f"Synthetic Nodes B (green) [{len(synthetic_nodes_b)}]", show=True)

        for sn in synthetic_nodes_a:
            lat, lon = sn["lat"], sn["lon"]
            road_name = sn.get("road_name", "?")
            node_id = str(sn.get("node_id", "?"))[:20]

            tooltip = f"SYNTHETIC A<br>Node: {node_id}<br>Road: {road_name}"

            folium.CircleMarker(
                location=(lat, lon),
                radius=6,
                color="purple",
                fill=True,
                fillColor="purple",
                fillOpacity=0.9,
                tooltip=tooltip
            ).add_to(fg_synth_nodes_a)

        for sn in synthetic_nodes_b:
            lat, lon = sn["lat"], sn["lon"]
            road_name = sn.get("road_name", "?")
            node_id = str(sn.get("node_id", "?"))[:20]

            tooltip = f"SYNTHETIC B<br>Node: {node_id}<br>Road: {road_name}"

            folium.CircleMarker(
                location=(lat, lon),
                radius=6,
                color="green",
                fill=True,
                fillColor="green",
                fillOpacity=0.9,
                tooltip=tooltip
            ).add_to(fg_synth_nodes_b)

        fg_synth_nodes_a.add_to(m)
        fg_synth_nodes_b.add_to(m)

        # Show drive nodes from secondary/tertiary roads
        fg_drive_nodes = FeatureGroup(name=f"Drive Nodes Sec/Tert (teal) [{len(drive_nodes_sec_tert)}]", show=True)

        for dn in drive_nodes_sec_tert:
            lat, lon = dn["lat"], dn["lon"]
            highway = dn["highway"]
            road_name = dn["road_name"]
            node_id = str(dn["node_id"])[:12]  # Truncate for display

            tooltip = f"DRIVE NODE<br>ID: {node_id}<br>Road: {road_name}<br>Type: {highway}"

            folium.CircleMarker(
                location=(lat, lon),
                radius=4,
                color="teal",
                fill=True,
                fillColor="teal",
                fillOpacity=0.8,
                tooltip=tooltip
            ).add_to(fg_drive_nodes)

        fg_drive_nodes.add_to(m)

        print(f"  Debug: {len(edges)} drive edges shown")
        a_count = sum(1 for e in edges if e.get("direction") == "A")
        b_count = sum(1 for e in edges if e.get("direction") == "B")
        print(f"  Debug: {a_count} in direction A, {b_count} in direction B")
        print(f"  Debug: {len(drive_nodes_a)} drive nodes A, {len(drive_nodes_b)} drive nodes B")
        print(f"  Debug: {len(synthetic_nodes_a)} synthetic nodes A, {len(synthetic_nodes_b)} synthetic nodes B")
        print(f"  Debug: {len(real_crossings)} real crossings, {len(synth_crossings)} synthetic crossings")
        print(f"  Debug: {len(drive_nodes_sec_tert)} secondary/tertiary drive nodes shown (teal layer)")

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_html)

    payload = {
        "input": input_path,
        "output_html": out_html,
        "crossings_injection_pkl": out_pkl,
        "crossings_count": len(crossings),
        "crossings_injection": {
            "nodes": injection_payload["metadata"]["node_count"],
            "edge_pairs": injection_payload["metadata"]["edge_count"],
        },
        "diagnostics": diag,
    }
    if args.debug:
        payload["debug_edges_count"] = len(debug_info.get("edges", []))

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"  Crossings generated: {len(crossings)}")
    print(f"  Injection PKL      : {out_pkl}")
    print(f"  Diagnostics JSON   : {out_json}")
    print(f"  Preview HTML       : {out_html}")


if __name__ == "__main__":
    main()
