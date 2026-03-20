"""
Fast isolated preview for named-opposite secondary/tertiary synthetic crossings.

This script only builds drive + walk graphs, synthesizes crossings, and writes
an HTML map. It skips ALNS and route solving for rapid iteration.

Usage:
  source .venv/bin/activate && python3 experiments/experiment4_crossings/preview_named_crossings.py
  source .venv/bin/activate && python3 experiments/experiment4_crossings/preview_named_crossings.py --input experiments/experiment4_crossings/input.json
  source .venv/bin/activate && python3 experiments/experiment4_crossings/preview_named_crossings.py --debug
"""

import argparse
import hashlib
import json
import os
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
    out_html = os.path.join(run_dir, "named_crossings_preview.html")
    out_json = os.path.join(run_dir, "named_crossings_diagnostics.json")

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

    print("[3/3] Synthesizing named-opposite crossings...")
    synth_cfg = dict(synth_cfg)
    synth_cfg["enabled"] = True
    synth_cfg["strategy"] = "named_opposite_secondary_tertiary"
    synth_cfg["center_lat"] = center[0]
    synth_cfg["center_lon"] = center[1]
    synth_cfg.setdefault("radius_km", walk_radius_km)
    _eng.set_walk_graph(G_walk, synthetic_cfg=synth_cfg, drive_graph=G_drive)

    diag = _eng.get_synthetic_diagnostics()
    crossings = _eng.get_synthetic_crossings()
    debug_info = _eng.get_crossing_debug_candidates()

    # Extract secondary/tertiary drive nodes for visualization (filtered by default bbox)
    drive_nodes_sec_tert = _extract_secondary_tertiary_drive_nodes_in_bbox(G_drive)

    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")
    folium.Marker(location=center, popup="School", tooltip="School").add_to(m)

    fg_walk = FeatureGroup(name="Walk Network", show=False)
    for seg in _extract_walk_segments(G_walk):
        folium.PolyLine(seg, color="#4aa3df", weight=1.5, opacity=0.5).add_to(fg_walk)
    fg_walk.add_to(m)

    # Crossings with numbers
    fg_syn = FeatureGroup(name=f"Crossings ({len(crossings)})", show=True)
    for i, seg in enumerate(_extract_synth_segments(G_walk)):
        folium.PolyLine(seg, color="#ff00aa", weight=4, opacity=0.95).add_to(fg_syn)
        # Add numbered marker at midpoint
        mid_lat = (seg[0][0] + seg[-1][0]) / 2
        mid_lon = (seg[0][1] + seg[-1][1]) / 2
        folium.Marker(
            location=(mid_lat, mid_lon),
            icon=folium.DivIcon(
                html=f'<div style="font-size:10px;font-weight:bold;color:#c00;background:white;padding:1px 3px;border-radius:3px;border:1px solid #c00;">{i+1}</div>',
                icon_anchor=(8, 8),
            ),
        ).add_to(fg_syn)
    fg_syn.add_to(m)

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

        # Show walk nodes used in crossings
        fg_nodes_a = FeatureGroup(name=f"Walk Nodes A (blue) [{len([n for n in walk_nodes if n.get('direction')=='A'])}]", show=True)
        fg_nodes_b = FeatureGroup(name=f"Walk Nodes B (orange) [{len([n for n in walk_nodes if n.get('direction')=='B'])}]", show=True)

        for wn in walk_nodes:
            lat, lon = wn["lat"], wn["lon"]
            direction = wn.get("direction", "?")
            road_name = wn.get("road_name", "?")
            node_id = str(wn.get("node", "?"))[:10]

            tooltip = f"Node: {node_id}<br>Road: {road_name}<br>Side: {direction}"

            if direction == "A":
                folium.CircleMarker(
                    location=(lat, lon),
                    radius=5,
                    color="blue",
                    fill=True,
                    fillColor="blue",
                    fillOpacity=0.7,
                    tooltip=tooltip
                ).add_to(fg_nodes_a)
            else:
                folium.CircleMarker(
                    location=(lat, lon),
                    radius=5,
                    color="orange",
                    fill=True,
                    fillColor="orange",
                    fillOpacity=0.7,
                    tooltip=tooltip
                ).add_to(fg_nodes_b)

        fg_nodes_a.add_to(m)
        fg_nodes_b.add_to(m)

        # Show synthetic nodes
        fg_synth_nodes_a = FeatureGroup(name=f"Synthetic Nodes A (purple) [{len([n for n in synthetic_nodes if n.get('direction')=='A'])}]", show=True)
        fg_synth_nodes_b = FeatureGroup(name=f"Synthetic Nodes B (green) [{len([n for n in synthetic_nodes if n.get('direction')=='B'])}]", show=True)

        for sn in synthetic_nodes:
            lat, lon = sn["lat"], sn["lon"]
            direction = sn.get("direction", "?")
            road_name = sn.get("road_name", "?")
            node_id = str(sn.get("node", "?"))[:20]

            tooltip = f"SYNTHETIC<br>Node: {node_id}<br>Road: {road_name}<br>Side: {direction}"

            if direction == "A":
                folium.CircleMarker(
                    location=(lat, lon),
                    radius=6,
                    color="purple",
                    fill=True,
                    fillColor="purple",
                    fillOpacity=0.9,
                    tooltip=tooltip
                ).add_to(fg_synth_nodes_a)
            else:
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
        print(f"  Debug: {len(walk_nodes)} walk nodes shown (used in crossings)")
        print(f"  Debug: {len(synthetic_nodes)} synthetic nodes created")
        print(f"  Debug: {len(drive_nodes_sec_tert)} secondary/tertiary drive nodes shown")

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_html)

    payload = {
        "input": input_path,
        "output_html": out_html,
        "crossings_count": len(crossings),
        "diagnostics": diag,
    }
    if args.debug:
        payload["debug_edges_count"] = len(debug_info.get("edges", []))

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"  Crossings generated: {len(crossings)}")
    print(f"  Diagnostics JSON   : {out_json}")
    print(f"  Preview HTML       : {out_html}")


if __name__ == "__main__":
    main()
