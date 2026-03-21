"""
experiments/comparison/run_comparison.py
========================================
Reads ``meta.json`` from this folder, generates a dataset with the specified
stage distribution, then runs three routing modes and produces an interactive
Folium comparison map with directional arrow-heads on every bus route.

Key design: the **bus** always drives on the full (unconstrained) road network.
Safety constraints only affect the **student walking BFS** — which stops the
student can reach on foot without crossing a dangerous road.

Modes
-----
  A  Constrained   – walking BFS avoids primary/trunk/secondary; same walk radius as B
  B  Unconstrained – walking BFS uses all edges; same walk radius as A
  C  Door-to-Door  – walk_radius=0 for all (bus visits every home)

Usage (from the repo root):
    python -m experiments.comparison.run_comparison          # uses meta.json defaults
    python -m experiments.comparison.run_comparison --iterations 50
"""

import os, sys, json, time, copy, math, argparse, datetime, statistics, random, pickle

# ── path fix: ensure repo root is on sys.path ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import folium
from folium import plugins, FeatureGroup

import detour_engine as _eng
import alns_engine   as _alns

from run_algorithm import (
    setup_graph, setup_walk_graph, precompute_matrix, run_algorithm, find_minimum_fleet,
    DEFAULT_STAGE_WALK_LIMITS,
)
from data_loader   import load_mode1_input
from solution_state import ServiceSolution
from detour_engine  import (
    calculate_route_path_and_stats,
    calculate_route_time_from_matrix,
    walk_path_on_roads,
    walk_distance_on_roads,
    find_shortest_path_with_turns,
    compute_student_tmax,
    compute_direct_time,
    calculate_walk_penalty,
    _MATRIX_CACHE,
)

# Patch: fast snap for large graphs
_eng._FAST_SNAP_MODE = True

# ────────────────────────────────────────────────────────────────────
# Load input.json
# ────────────────────────────────────────────────────────────────────
_INPUT_PATH = os.path.join(_SCRIPT_DIR, "input.json")

def _load_meta(path=None):
    with open(path or _INPUT_PATH) as f:
        return json.load(f)


def _resolve_injection_pkl_path(injection_cfg, input_path):
    if not isinstance(injection_cfg, dict):
        return None
    raw = injection_cfg.get("pkl_path")
    if not raw:
        return None
    if os.path.isabs(raw):
        return raw
    bases = []
    if input_path:
        input_dir = os.path.dirname(input_path)
        bases.append(input_dir)
        bases.append(os.path.dirname(input_dir))
    bases.extend([_SCRIPT_DIR, _ROOT])
    for base in bases:
        if not base:
            continue
        candidate = os.path.abspath(os.path.join(base, raw))
        if os.path.exists(candidate):
            return candidate
    base = os.path.dirname(input_path) if input_path else _SCRIPT_DIR
    return os.path.abspath(os.path.join(base, raw))


def _load_crossings_injection_payload(pkl_path):
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Crossings injection payload must be a dict")
    for key in ("nodes", "edge_pairs"):
        if key not in payload:
            raise ValueError(f"Crossings injection payload missing required key: {key}")
    return payload


def _validate_crossings_injection_payload(payload, synth_cfg):
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    expected_strategy = str(synth_cfg.get("strategy", "")) if isinstance(synth_cfg, dict) else ""
    actual_strategy = str(metadata.get("strategy", ""))
    if expected_strategy and actual_strategy and expected_strategy != actual_strategy:
        raise ValueError(
            f"Crossings injection strategy mismatch: expected '{expected_strategy}', got '{actual_strategy}'"
        )


def _inject_crossings_into_walk_graph(walk_graph, payload):
    nodes = payload.get("nodes", []) if isinstance(payload, dict) else []
    edge_pairs = payload.get("edge_pairs", []) if isinstance(payload, dict) else []
    crossings = payload.get("crossings", []) if isinstance(payload, dict) else []

    node_lookup = {}
    for node in nodes:
        node_id = node.get("node_id")
        if node_id is None:
            continue
        lat = float(node.get("lat", 0.0))
        lon = float(node.get("lon", 0.0))
        node_lookup[node_id] = node
        if node_id not in walk_graph:
            walk_graph.add_node(node_id, x=lon, y=lat)
        else:
            walk_graph.nodes[node_id].setdefault("x", lon)
            walk_graph.nodes[node_id].setdefault("y", lat)

    added_edges = 0
    markers = []

    for crossing in crossings:
        try:
            lat_a = float(crossing.get("lat_a", 0.0))
            lon_a = float(crossing.get("lon_a", 0.0))
            lat_b = float(crossing.get("lat_b", 0.0))
            lon_b = float(crossing.get("lon_b", 0.0))
            markers.append({
                "lat": (lat_a + lat_b) / 2.0,
                "lon": (lon_a + lon_b) / 2.0,
                "length_m": float(crossing.get("length_m", 0.0)),
                "crossing_type": crossing.get("crossing_type", "real_to_real"),
                "road_name": crossing.get("road_name", "?"),
                "node_a": crossing.get("node_a"),
                "node_b": crossing.get("node_b"),
            })
        except Exception:
            continue

    for edge in edge_pairs:
        node_a = edge.get("node_a")
        node_b = edge.get("node_b")
        if node_a is None or node_b is None:
            continue
        if node_a not in walk_graph:
            info = node_lookup.get(node_a, {})
            walk_graph.add_node(node_a, x=float(info.get("lon", 0.0)), y=float(info.get("lat", 0.0)))
        if node_b not in walk_graph:
            info = node_lookup.get(node_b, {})
            walk_graph.add_node(node_b, x=float(info.get("lon", 0.0)), y=float(info.get("lat", 0.0)))

        if walk_graph.has_edge(node_a, node_b):
            continue

        length_m = float(edge.get("length_m", 0.0))
        walk_graph.add_edge(
            node_a,
            node_b,
            length=length_m,
            travel_time=(length_m / 80.0) if length_m > 0 else 0.0,
            synthetic_crossing=True,
            is_safe_to_cross=True,
            crossing_rule="injected_crossings",
            crossing_subtype=edge.get("crossing_type", "real_to_real"),
            road_name=edge.get("road_name", "?"),
            injected_crossing=True,
        )
        added_edges += 1

    return {
        "markers": markers,
        "nodes": nodes,
        "edge_pairs": edge_pairs,
        "added_edges": added_edges,
    }


# ────────────────────────────────────────────────────────────────────
# Dataset generation (delegates to experiments.generate_dataset)
# ────────────────────────────────────────────────────────────────────
def _generate_dataset(meta):
    """Call the refactored generate_dataset() with meta.json values."""
    sys.path.insert(0, os.path.join(_ROOT, "experiments"))
    from generate_dataset import generate_dataset

    return generate_dataset(
        n_students=meta["n_students"],
        seed=meta["seed"],
        school=meta["school"],
        stage_dist=meta["stage_distribution"],
        annulus=meta.get("annulus"),
        buses_count=meta.get("buses", {}).get("count", 4),
        bus_capacity=meta.get("buses", {}).get("capacity", 60),
        constraints=meta.get("constraints"),
        iterations=meta.get("algorithm", {}).get("iterations", 30),
    )


# ────────────────────────────────────────────────────────────────────
# Cache helpers
# ────────────────────────────────────────────────────────────────────
def _reset_caches(keep_matrix=False, keep_walk=False):
    """Clear caches between modes.

    keep_matrix=True : retain _MATRIX_CACHE / _MATRIX_CACHE_LENGTH / _path_cache.
        All three modes use G_unc for bus routing so matrix entries are reusable
        across the full pipeline without recomputing Dijkstra.
    keep_walk=True   : retain _WALK_DIST_CACHE / _WALK_GRAPH / _safe_nodes_cache.
        Safe when the walking graph is unchanged (Mode B→C both use G_unc).
    """
    _alns._student_candidate_cache.clear()
    _alns._student_candidate_dist.clear()
    if not keep_matrix:
        _eng._MATRIX_CACHE.clear()
        _eng._MATRIX_CACHE_LENGTH.clear()
        _eng._path_cache.clear()
        _eng._DIJKSTRA_DONE.clear()
    if not keep_walk:
        _eng._WALK_DIST_CACHE.clear()
        _eng._safe_nodes_cache.clear()
    _eng._STUDENT_NODE_CACHE.clear()


def _prebuild_ball_tree(G):
    print("  Pre-building BallTree …", end="", flush=True)
    t0 = time.time()
    _eng._get_or_build_ball_tree(G)
    print(f" {time.time()-t0:.1f}s")


# ────────────────────────────────────────────────────────────────────
# Input mutators  (same logic as visualize_comparison.py)
# ────────────────────────────────────────────────────────────────────
def _relax_ride_constraints(d):
    """Disable per-route ride-time caps so only the walking variable differs."""
    d["meta"].setdefault("constraints", {}).update(
        {"ride_time_multiplier": 999, "floor_minutes": 999, "ceiling_minutes": 999}
    )
    return d


def _make_constrained(data):
    """Mode A: safety constraints ON, ride-time constraints from meta.json."""
    return copy.deepcopy(data)


def _make_unconstrained(data):
    """Mode B: all-safe walking, ride-time constraints from meta.json."""
    d = copy.deepcopy(data)
    for s in d["data"]["students"]:
        s["walk_radius_override"] = 400
    return d


def _make_door_to_door(data):
    """Mode C: no walking (walk_radius=0), ride-time constraints from meta.json."""
    d = copy.deepcopy(data)
    for s in d["data"]["students"]:
        s["walk_radius_override"] = 0
    return d


# ────────────────────────────────────────────────────────────────────
# Crossing detection  — FIXED
# ────────────────────────────────────────────────────────────────────
#
# A "dangerous crossing" means the student's walking path passes
# through an edge whose highway type is a major road (primary, trunk,
# secondary, motorway).  Residential and tertiary roads are NOT
# dangerous — MID/HIGH students can cross them freely.
#
# The run-length heuristic:
#   • A consecutive run of dangerous edges < 150 m that is bounded by
#     non-dangerous edges on both sides = TRUE crossing (perpendicular).
#   • A longer run or one that starts/ends the path = walking ALONGSIDE
#     — not counted.
# ────────────────────────────────────────────────────────────────────

_CROSSING_RUN_MAX_M = 150.0

# Only these highway types count as genuinely dangerous to cross on foot.
_DANGEROUS_HW_TYPES = {'motorway', 'motorway_link', 'trunk', 'trunk_link',
                       'primary', 'primary_link', 'secondary', 'secondary_link'}


def _edge_is_dangerous(data_dict):
    """Return True if the edge represents a road dangerous to cross on foot."""
    hw = data_dict.get('highway', '')
    if isinstance(hw, list):
        hw = hw[0] if hw else ''
    return hw in _DANGEROUS_HW_TYPES


def _classify_walk_path(walk_path, G_con):
    """Detect true dangerous road crossings along a walk path.

    Uses the *constrained* graph as ground truth.  An edge is 'dangerous'
    only if its highway type is in _DANGEROUS_HW_TYPES (primary, trunk,
    secondary, motorway).  Tertiary and residential are safe to cross.
    """
    if len(walk_path) < 2:
        return []

    # Build per-edge metadata: (is_dangerous, length_m, u, v, highway)
    edges = []
    for i in range(len(walk_path) - 1):
        u, v = walk_path[i], walk_path[i + 1]
        ed = G_con.get_edge_data(u, v) or G_con.get_edge_data(v, u)
        if ed is None:
            edges.append((False, 0.0, u, v, ''))
            continue
        d = ed[0] if 0 in ed else list(ed.values())[0]
        dangerous = _edge_is_dangerous(d)
        hw = d.get('highway', '')
        if isinstance(hw, list):
            hw = hw[0] if hw else ''
        edges.append((dangerous, float(d.get('length', 0.0)), u, v, hw))

    crossings = []
    i = 0
    while i < len(edges):
        if not edges[i][0]:   # not dangerous — skip
            i += 1
            continue

        # Start of a dangerous run
        run_start   = i
        run_total_m = 0.0
        run_hws     = set()
        while i < len(edges) and edges[i][0]:
            run_total_m += edges[i][1]
            run_hws.add(edges[i][4])
            i += 1
        run_end = i  # exclusive

        # Find midpoint for marker
        accumulated = 0.0
        mid_u, mid_v = edges[run_start][2], edges[run_start][3]
        for j in range(run_start, run_end):
            accumulated += edges[j][1]
            if accumulated >= run_total_m / 2:
                mid_u, mid_v = edges[j][2], edges[j][3]
                break

        came_from_safe = (run_start == 0) or not edges[run_start - 1][0]
        goes_to_safe   = (run_end >= len(edges)) or not edges[run_end][0]

        if run_total_m < _CROSSING_RUN_MAX_M and came_from_safe and goes_to_safe:
            mid_lat = (G_con.nodes[mid_u]['y'] + G_con.nodes[mid_v]['y']) / 2
            mid_lon = (G_con.nodes[mid_u]['x'] + G_con.nodes[mid_v]['x']) / 2
            crossings.append({
                'u': mid_u, 'v': mid_v,
                'lat': mid_lat, 'lon': mid_lon,
                'run_length_m': round(run_total_m, 1),
                'road_types': ', '.join(sorted(run_hws)),
            })
    return crossings


def _count_unsafe_crossings(sol, G_con, G_solve):
    """Detect dangerous crossings for every served student.

    Walks on G_solve (the graph used for that mode's routing), but checks
    edge danger using G_con (the constrained ground-truth graph).
    """
    crossings = []
    for route in sol.routes:
        for stop in route.stops:
            if stop.stop_type == 'school':
                continue
            for student in stop.students:
                s_node = _eng.fast_nearest_node(G_solve, student.coords[1], student.coords[0])
                wp = walk_path_on_roads(G_solve, s_node, stop.node_id)
                for cx in _classify_walk_path(wp, G_con):
                    cx['student_id'] = student.id
                    crossings.append(cx)
    return crossings


# ────────────────────────────────────────────────────────────────────
# Dangerous / unclassified road helpers
# ────────────────────────────────────────────────────────────────────
def _extract_segments(G_con, center_lat, center_lon, kind="dangerous"):
    """Return coord lists for dangerous OR unclassified road segments within 5 km."""
    segments = []
    seen = set()
    for u, v, k, data in G_con.edges(keys=True, data=True):
        if kind == "dangerous":
            if data.get("is_safe_to_cross", True):
                continue
        elif kind == "unclassified":
            hw = data.get("highway", "")
            if isinstance(hw, list):
                hw = hw[0]
            if hw != "unclassified":
                continue
        if data.get("length", 0) < 20:
            continue
        ek = (min(u, v), max(u, v))
        if ek in seen:
            continue
        seen.add(ek)
        mid_lat = (G_con.nodes[u]["y"] + G_con.nodes[v]["y"]) / 2
        mid_lon = (G_con.nodes[u]["x"] + G_con.nodes[v]["x"]) / 2
        dlat = abs(mid_lat - center_lat) * 111.0
        dlon = abs(mid_lon - center_lon) * 111.0 * math.cos(math.radians(center_lat))
        if math.sqrt(dlat**2 + dlon**2) > 5.0:
            continue
        if "geometry" in data:
            coords = [(lat, lon) for lon, lat in data["geometry"].coords]
        else:
            coords = [(G_con.nodes[u]["y"], G_con.nodes[u]["x"]),
                       (G_con.nodes[v]["y"], G_con.nodes[v]["x"])]
        segments.append(coords)
    return segments


def _extract_walk_segments(G_walk, center_lat, center_lon, radius_km=4.0, safe_only=True):
    """Return coord lists for walk-network segments within radius_km of center.

    When safe_only=True, skip major-road edges that are unsafe to cross.
    """
    segments = []
    seen = set()
    for u, v, k, data in G_walk.edges(keys=True, data=True):
        if safe_only:
            hw = data.get("highway", "")
            if isinstance(hw, list):
                hw = hw[0] if hw else ""
            if hw in _DANGEROUS_HW_TYPES:
                continue
        if data.get("length", 0) < 15:
            continue
        ek = (min(u, v), max(u, v))
        if ek in seen:
            continue
        seen.add(ek)
        mid_lat = (G_walk.nodes[u]["y"] + G_walk.nodes[v]["y"]) / 2
        mid_lon = (G_walk.nodes[u]["x"] + G_walk.nodes[v]["x"]) / 2
        dlat = abs(mid_lat - center_lat) * 111.0
        dlon = abs(mid_lon - center_lon) * 111.0 * math.cos(math.radians(center_lat))
        if math.sqrt(dlat**2 + dlon**2) > radius_km:
            continue
        if "geometry" in data:
            coords = [(lat, lon) for lon, lat in data["geometry"].coords]
        else:
            coords = [(G_walk.nodes[u]["y"], G_walk.nodes[u]["x"]),
                      (G_walk.nodes[v]["y"], G_walk.nodes[v]["x"]) ]
        segments.append(coords)
    return segments


# ────────────────────────────────────────────────────────────────────
# MAP BUILDING  (with PolyLineTextPath arrows, like visualization.py)
# ────────────────────────────────────────────────────────────────────
_ROUTE_COLORS = {
    "A": ["#2196F3", "#1565C0", "#0D47A1", "#82B1FF"],
    "B": ["#4CAF50", "#2E7D32", "#1B5E20", "#A5D6A7"],
    "C": ["#FF9800", "#E65100", "#BF360C", "#FFCC80"],
}
# Matching folium-valid named colors for Icon markers (same order as _ROUTE_COLORS)
_ICON_COLORS = {
    "A": ["blue",   "darkblue",  "darkblue",  "lightblue"],
    "B": ["green",  "darkgreen", "darkgreen", "lightgreen"],
    "C": ["orange", "red",       "darkred",   "beige"],
}
_MODE_NAMES = {
    "A": "Constrained (Safe Walking)",
    "B": "Unconstrained (Any Walking)",
    "C": "Door-to-Door (No Walking)",
}

import networkx as nx
from detour_engine import (
    find_shortest_path_with_turns,
    get_bearing_of_path,
    _candidate_points as _cand_pts,
    get_crossing_bfs_stats,
    reset_crossing_bfs_stats,
)

# Arrow text template — spaces pad between arrow glyphs
_ARROW_TEXT = "          \u27A4          "


def _compute_route_path(G, stops):
    """Compute the full node-level path between consecutive stops.

    Uses ``find_shortest_path_with_turns`` (bearing-aware A*) so that
    U-turns are penalised / banned — matching the solver's routing logic.

    The matrix cache only stores *times* (no paths), which makes the
    standard helper return ``(None, time)`` and breaks rendering.
    We work around this by chaining bearings across segments: when
    ``initial_bearing`` is not None the function skips the matrix
    shortcut and either hits the path cache or runs a full A*.
    """
    if len(stops) < 2:
        return []

    full_path = []
    last_bearing = None          # chain across segments

    for i in range(len(stops) - 1):
        u = stops[i].node_id
        v = stops[i + 1].node_id
        if u == v:
            if not full_path:
                full_path.append(u)
            continue

        # Use the turn-aware pathfinder.
        # Passing initial_bearing (even 0.0 on first call) bypasses
        # the matrix-only shortcut so we always get an actual path.
        bearing_arg = last_bearing if last_bearing is not None else 0.0
        seg, _ = find_shortest_path_with_turns(
            G, u, v, weight='travel_time', initial_bearing=bearing_arg,
        )

        if seg is None or len(seg) < 2:
            # Fallback: plain Dijkstra (at least draws *something*)
            try:
                seg = nx.shortest_path(G, u, v, weight='travel_time')
            except Exception:
                try:
                    seg = nx.shortest_path(G, u, v, weight='length')
                except Exception:
                    continue

        if not full_path:
            full_path.extend(seg)
        else:
            full_path.extend(seg[1:])

        last_bearing = get_bearing_of_path(G, seg)

    return full_path


def _build_path_coords(G, full_path, offset=0.0):
    """Convert a list of node IDs to (lat, lon) tuples following edge geometries."""
    coords = []
    for i in range(len(full_path) - 1):
        u, v = full_path[i], full_path[i + 1]
        ed = G.get_edge_data(u, v)
        if not ed:
            continue
        d = ed[0] if 0 in ed else list(ed.values())[0]
        if "geometry" in d:
            for lon, lat in d["geometry"].coords:
                coords.append((lat + offset, lon + offset))
        else:
            coords.append((G.nodes[u]["y"] + offset, G.nodes[u]["x"] + offset))
    last = full_path[-1]
    coords.append((G.nodes[last]["y"] + offset, G.nodes[last]["x"] + offset))
    return coords


def _build_walk_coords(G, wp):
    walk_g = getattr(_eng, "_WALK_GRAPH", None)

    def _node_coords(nid):
        if walk_g is not None and nid in walk_g.nodes:
            return (walk_g.nodes[nid]["y"], walk_g.nodes[nid]["x"])
        if nid in G.nodes:
            return (G.nodes[nid]["y"], G.nodes[nid]["x"])
        return None

    wcoords = []
    for wi in range(len(wp) - 1):
        u2, v2 = wp[wi], wp[wi + 1]
        ed2 = None
        if walk_g is not None:
            ed2 = walk_g.get_edge_data(u2, v2) or walk_g.get_edge_data(v2, u2)
        if not ed2:
            ed2 = G.get_edge_data(u2, v2) or G.get_edge_data(v2, u2)
        if ed2:
            dd = ed2[0] if 0 in ed2 else list(ed2.values())[0]
            if "geometry" in dd:
                for lon, lat in dd["geometry"].coords:
                    wcoords.append((lat, lon))
            else:
                c = _node_coords(u2)
                if c is not None:
                    wcoords.append(c)
        else:
            c = _node_coords(u2)
            if c is not None:
                wcoords.append(c)
    c_last = _node_coords(wp[-1])
    if c_last is not None:
        wcoords.append(c_last)
    return wcoords


def _compute_pm_ride_time(route, stop, G):
    """Afternoon (school→home-stop) ride time using the reversed route sequence."""
    interior = [s for s in route.stops if s.stop_type != 'school']
    afternoon = [route.stops[0]] + interior[::-1] + [route.stops[-1]]
    target_idx = next((i for i, s in enumerate(afternoon) if s is stop), -1)
    if target_idx <= 0:
        return 0.0
    total = 0.0
    for i in range(target_idx):
        u = afternoon[i].node_id
        v = afternoon[i + 1].node_id
        t = _MATRIX_CACHE.get((u, v), None)
        if t is None:
            _, t = find_shortest_path_with_turns(G, u, v)
        if not math.isfinite(t):
            return float('inf')
        total += t
    return total


def _dir_cap_html(label, ride, direct, cap, k):
    """Compact per-direction ride-cap block with progress bar."""
    if direct is None or direct <= 0 or not math.isfinite(ride):
        ride_str = f"{ride:.1f}" if math.isfinite(ride) else "∞"
        return (
            f'<div style="margin:3px 0;font-size:11px;">'
            f'  <b>{label}:</b> ride&nbsp;<b>{ride_str}&nbsp;min</b> — direct N/A'
            f'</div>'
        )
    ratio = ride / direct
    ratio_color = 'green' if ratio <= 1.5 else ('darkorange' if ratio <= k else 'red')
    cap_safe = cap if math.isfinite(cap) else 999
    usage_pct = min(100, int(ride / cap_safe * 100)) if cap_safe > 0 else 0
    status = '✖ over cap' if ride > cap else '✔ ok'
    status_color = 'red' if ride > cap else 'green'
    return (
        f'<div style="margin:3px 0;font-size:11px;border-left:3px solid {ratio_color};padding-left:4px;">'
        f'  <b>{label}:</b> '
        f'  ride <b style="color:{ratio_color};">{ride:.1f}</b> / cap <b>{cap:.1f}</b> min'
        f'  &nbsp;<span style="color:{ratio_color};">({ratio:.2f}×)</span>'
        f'  <span style="color:{status_color};float:right;">{status}</span><br>'
        f'  direct {direct:.1f} min'
        f'  <div style="background:#eee;border-radius:3px;height:5px;margin-top:2px;">'
        f'    <div style="background:{ratio_color};width:{usage_pct}%;height:5px;border-radius:3px;"></div>'
        f'  </div>'
        f'</div>'
    )


def _add_route_layer(m, G, sol, mode_key, G_con, constraints=None):
    """Add route + walk FeatureGroups for one mode.  Returns (fg_routes, fg_walks, crossings, occupancies)."""
    con = constraints or {}
    ride_k       = float(con.get("ride_time_multiplier", 2.5))
    floor_min    = float(con.get("floor_minutes",        45))
    ceiling_min  = float(con.get("ceiling_minutes",      60))
    caps_enabled = bool(con.get("enabled",              True))

    show = mode_key in ("A", "B")   # show constrained + unconstrained by default
    fg_routes = FeatureGroup(name=f"{_MODE_NAMES[mode_key]} – Routes",        show=show)
    fg_walks  = FeatureGroup(name=f"{_MODE_NAMES[mode_key]} – Walking Paths", show=show)
    colors      = _ROUTE_COLORS[mode_key]
    icon_colors = _ICON_COLORS[mode_key]
    active  = [r for r in sol.routes if r.get_student_count() > 0]
    occupancies = []

    for ri, route in enumerate(active):
        c  = colors[ri % len(colors)]
        ic = icon_colors[ri % len(icon_colors)]

        # ── Bus route polyline with arrow-heads ──
        if len(route.stops) > 1:
            full_path = _compute_route_path(G, route.stops)
            if full_path and len(full_path) >= 2:
                offset = 0.00003 * (ri - 0.5)
                coords = _build_path_coords(G, full_path, offset)

                pl = folium.PolyLine(
                    coords, color=c, weight=5, opacity=0.8,
                    popup=f"Mode {mode_key} Route {route.route_id} "
                          f"({route.get_student_count()} students, "
                          f"{route.total_time:.0f} min, "
                          f"{route.total_distance:.1f} km)",
                )
                pl.add_to(fg_routes)

                # Directional arrows (same technique as visualization.py)
                plugins.PolyLineTextPath(
                    pl, _ARROW_TEXT, repeat=True, offset=6,
                    attributes={"fill": c, "font-weight": "bold", "font-size": "24"},
                ).add_to(fg_routes)

        # ── Stop markers ──
        student_count = 0
        for si, stop in enumerate(route.stops):
            if stop.stop_type == "school":
                continue
            n_stu = len(stop.students)
            student_count += n_stu
            student_ids = [s.id for s in stop.students]
            student_ids_html = ", ".join(student_ids) if student_ids else "—"
            folium.CircleMarker(
                location=stop.coords, radius=7,
                color=c, fill=True, fillColor=c, fillOpacity=0.85,
                popup=folium.Popup(
                    f'<div style="width:220px;font-size:12px;">'
                    f'<b>Stop {si} — {route.route_id}</b><br>'
                    f'Mode: {_MODE_NAMES[mode_key]}<br>'
                    f'Students ({n_stu}): {student_ids_html}</div>',
                    max_width=260,
                ),
                tooltip=f"{mode_key}-{route.route_id} Stop {si} ({n_stu} students)",
            ).add_to(fg_routes)

            # ── Walk paths + student home markers ──
            for student in stop.students:
                s_node = _eng.fast_nearest_node(G, student.coords[1], student.coords[0])
                wp = walk_path_on_roads(G, s_node, stop.node_id)
                if len(wp) >= 2:
                    wcoords = _build_walk_coords(G, wp)
                    walk_dist = walk_distance_on_roads(G, s_node, stop.node_id)
                    folium.PolyLine(
                        wcoords, color=c, weight=2, opacity=0.6, dash_array="6,4",
                        tooltip=f"{student.id} walk: {walk_dist:.0f} m",
                    ).add_to(fg_walks)

                walk_m = walk_distance_on_roads(G, s_node, stop.node_id)

                # ── AM ride time: this stop → school (matrix-cache safe) ──
                stop_idx = next((i for i, s in enumerate(route.stops) if s is stop), -1)
                ride_time_am = 0.0
                if stop_idx != -1:
                    ride_time_am = calculate_route_time_from_matrix(route.stops[stop_idx:], G)
                    if ride_time_am >= 9999:
                        ride_time_am = float('inf')

                # ── PM ride time: school → this stop (reversed route) ──
                ride_time_pm = _compute_pm_ride_time(route, stop, G)

                # ── Direct times: AM = home→school, PM = school→home ──
                school_node = route.stops[-1].node_id
                direct_am = None
                direct_pm = None
                try:
                    _, direct_am = find_shortest_path_with_turns(G, s_node, school_node, weight='travel_time')
                    if not math.isfinite(direct_am):
                        direct_am = None
                except Exception:
                    pass
                try:
                    _, direct_pm = find_shortest_path_with_turns(G, school_node, s_node, weight='travel_time')
                    if not math.isfinite(direct_pm):
                        direct_pm = None
                except Exception:
                    pass

                # ── Per-direction caps ──
                k_eff = getattr(route, 'ride_time_multiplier', ride_k)
                fl    = getattr(route, 'floor_minutes',        floor_min)
                ce    = getattr(route, 'ceiling_minutes',      ceiling_min)

                def _cap(d):
                    if d is None or d <= 0:
                        return float('inf')
                    return max(fl, min(k_eff * d, d + ce))

                cap_html = (
                    _dir_cap_html('🟠 AM home→school', ride_time_am, direct_am, _cap(direct_am), k_eff) +
                    _dir_cap_html('🟦 PM school→home', ride_time_pm, direct_pm, _cap(direct_pm), k_eff)
                ) if caps_enabled else ''

                # ── Walk info ──
                stage_name = (
                    student.school_stage.name
                    if hasattr(student.school_stage, "name")
                    else str(student.school_stage)
                )
                walk_limit = student.walk_radius if hasattr(student, 'walk_radius') else 0
                walk_info  = (f"{walk_m:.0f}m / {walk_limit:.0f}m" if walk_limit > 0
                              else f"{walk_m:.0f}m (Door-to-Door)")

                popup_html = (
                    f'<div style="width:260px;font-size:12px;">'
                    f'<b>Student: {student.id}</b><br>'
                    f'Stage: {stage_name}<br>'
                    f'Home: {student.coords[0]:.5f}, {student.coords[1]:.5f}<br>'
                    f'Mode: {_MODE_NAMES[mode_key]}<br>'
                    f'<div style="margin-top:5px;border-top:1px solid #ccc;padding-top:5px;">'
                    f'<b>Route:</b> {route.route_id}<br>'
                    f'{cap_html}'
                    f'<div style="margin-top:3px;font-size:11px;">Walk to Stop: {walk_info}</div>'
                    f'</div></div>'
                )

                folium.Marker(
                    location=student.coords,
                    tooltip=f"{student.id} ({stage_name}) — AM {ride_time_am:.0f} min, walk {walk_m:.0f}m",
                    popup=folium.Popup(popup_html, max_width=300),
                    icon=folium.Icon(color=ic, icon='home', prefix='fa'),
                ).add_to(fg_walks)

        occupancies.append(student_count)

    crossings = _count_unsafe_crossings(sol, G_con, G)
    fg_routes.add_to(m)
    fg_walks.add_to(m)
    return fg_routes, fg_walks, crossings, occupancies


def _count_satisfied_per_route(sol, G, constraints):
    """Return {route_id: satisfied_count} for all active routes.

    Respects ``bidirectional_check``:
      True  → satisfied when AM **or** PM ride ≤ cap (lenient: both must fail to fail)
      False → satisfied when AM ride ≤ cap (strict)

    Students without a finite direct time are counted as satisfied.
    """
    con     = constraints or {}
    k       = float(con.get('ride_time_multiplier', 2.5))
    fl      = float(con.get('floor_minutes',        45))
    ce      = float(con.get('ceiling_minutes',      60))
    bidir   = bool(con.get('bidirectional_check',   True))
    caps_on = bool(con.get('enabled',               True))

    def _cap(d):
        if d is None or d <= 0 or not math.isfinite(d):
            return float('inf')
        return max(fl, min(k * d, d + ce))

    result = {}
    for route in sol.routes:
        if route.get_student_count() == 0:
            continue
        if not caps_on:
            result[route.route_id] = route.get_student_count()
            continue
        satisfied = 0
        school_node = route.stops[-1].node_id
        for stop in route.stops:
            if stop.stop_type == 'school':
                continue
            stop_idx = next((i for i, s in enumerate(route.stops) if s is stop), -1)
            if stop_idx == -1:
                continue
            ride_am = calculate_route_time_from_matrix(route.stops[stop_idx:], G)
            if ride_am >= 9999:
                ride_am = float('inf')
            for student in stop.students:
                s_node = _eng.fast_nearest_node(G, student.coords[1], student.coords[0])
                try:
                    _, direct_am = find_shortest_path_with_turns(
                        G, s_node, school_node, weight='travel_time')
                    if not math.isfinite(direct_am):
                        direct_am = None
                except Exception:
                    direct_am = None
                cap_am = _cap(direct_am)
                am_ok  = ride_am <= cap_am
                if not bidir:
                    if am_ok:
                        satisfied += 1
                else:
                    if am_ok:
                        satisfied += 1
                    else:
                        ride_pm = _compute_pm_ride_time(route, stop, G)
                        try:
                            _, direct_pm = find_shortest_path_with_turns(
                                G, school_node, s_node, weight='travel_time')
                            if not math.isfinite(direct_pm):
                                direct_pm = None
                        except Exception:
                            direct_pm = None
                        if ride_pm <= _cap(direct_pm):
                            satisfied += 1
        result[route.route_id] = satisfied
    return result


def _count_satisfied(sol, G, constraints):
    """Total satisfied count (sum of _count_satisfied_per_route)."""
    return sum(_count_satisfied_per_route(sol, G, constraints).values())
    
def _count_cap_violations(sol, G, constraints):
    """Count AM/PM ride-time cap violations among served students.

    Returns a dict with AM/PM counts or None values when caps are off.
    """
    if not constraints:
        return {
            "am": None, "am_checked": None, "am_pct": None,
            "pm": None, "pm_checked": None, "pm_pct": None,
        }
    enabled = bool(constraints.get("enabled", True))
    soft = bool(constraints.get("soft_ride_caps", False))
    if not enabled and not soft:
        return {
            "am": None, "am_checked": None, "am_pct": None,
            "pm": None, "pm_checked": None, "pm_pct": None,
        }

    k_mult = float(constraints.get("ride_time_multiplier", 2.5))
    floor_min = float(constraints.get("floor_minutes", 45))
    ceiling_min = float(constraints.get("ceiling_minutes", 60))

    def _edge_time(u, v):
        t = _MATRIX_CACHE.get((u, v), None)
        if t is None:
            _, t = find_shortest_path_with_turns(G, u, v)
        return t if math.isfinite(t) else None

    am_viol = am_checked = 0
    pm_viol = pm_checked = 0

    for route in sol.routes:
        if not route.stops:
            continue
        school_node = route.stops[-1].node_id

        # AM ride times: stop -> school (forward)
        am_time_by_stop = {}
        total = 0.0
        ok = True
        for i in range(len(route.stops) - 1, 0, -1):
            u = route.stops[i - 1].node_id
            v = route.stops[i].node_id
            t = _edge_time(u, v)
            if t is None:
                ok = False
                break
            total += t
            am_time_by_stop[route.stops[i - 1]] = total
        if not ok:
            am_time_by_stop = {}

        # PM ride times: school -> stop in reversed route
        pm_time_by_stop = {}
        interior = route.stops[1:-1][::-1]
        pm_stops = [route.stops[0]] + interior + [route.stops[-1]]
        total = 0.0
        ok = True
        for i in range(len(pm_stops) - 1):
            u = pm_stops[i].node_id
            v = pm_stops[i + 1].node_id
            t = _edge_time(u, v)
            if t is None:
                ok = False
                break
            total += t
            pm_time_by_stop[pm_stops[i + 1]] = total
        if not ok:
            pm_time_by_stop = {}

        for stop in route.stops:
            if stop.stop_type == "school":
                continue
            ride_am = am_time_by_stop.get(stop)
            ride_pm = pm_time_by_stop.get(stop)
            for student in stop.students:
                direct_time = compute_direct_time(student, school_node, G)
                if direct_time is None or not math.isfinite(direct_time) or direct_time <= 0:
                    continue
                cap = max(floor_min, min(k_mult * direct_time, direct_time + ceiling_min))

                if ride_am is not None:
                    am_checked += 1
                    if ride_am > cap:
                        am_viol += 1
                if ride_pm is not None:
                    pm_checked += 1
                    if ride_pm > cap:
                        pm_viol += 1

    am_pct = round(am_viol / am_checked * 100, 1) if am_checked else None
    pm_pct = round(pm_viol / pm_checked * 100, 1) if pm_checked else None
    return {
        "am": am_viol, "am_checked": am_checked, "am_pct": am_pct,
        "pm": pm_viol, "pm_checked": pm_checked, "pm_pct": pm_pct,
    }


def _add_candidate_layer(m, G, mode_key, sol, cand_cache, cand_dist):
    """Add a FeatureGroup showing all candidate bus-stop nodes considered per student.

    Each dot is a node that was evaluated as a possible stop for a specific student
    during the ALNS insertion phase.  The dot's popup shows:
      - Student ID and stage
      - Score / points (0=residential dead-end, 1=intersection OR arterial, 2=both)
      - Walk distance from home to this candidate node
      - Whether this node was the student's actual assigned stop

    Hidden by default — toggle via the layer control.
    """
    fg = FeatureGroup(name=f"{_MODE_NAMES[mode_key]} – Candidate Stops", show=False)

    # Build lookup: student_id -> the node_id of their actual assigned stop (if served)
    assigned = {}  # student_id -> stop_node_id
    for route in sol.routes:
        for stop in route.stops:
            if stop.stop_type == 'school':
                continue
            for stu in stop.students:
                assigned[stu.id] = stop.node_id

    # Build lookup: student_id -> Student object
    stu_by_id = {s.id: s for s in sol.students}

    for sid, candidates in cand_cache.items():
        student = stu_by_id.get(sid)
        if student is None:
            continue
        dist_map = cand_dist.get(sid, {})
        stage_name = (
            student.school_stage.name
            if hasattr(student.school_stage, 'name')
            else str(student.school_stage)
        )

        for node_id, coords in candidates:
            walk_m   = dist_map.get(node_id, 0.0)
            pts      = _cand_pts(G, node_id)
            is_home  = (walk_m == 0.0)
            is_chosen = (assigned.get(sid) == node_id)

            pts_label  = ['0 – residential dead-end',
                          '1 – intersection OR arterial',
                          '2 – intersection AND arterial'][pts]
            home_flag  = ' 🏠 (home snap)' if is_home  else ''
            chosen_flag = ' ✔ chosen stop' if is_chosen else ''

            popup_html = (
                f'<div style="width:230px;font-size:12px;">'
                f'<b>Candidate Stop</b>{chosen_flag}{home_flag}<br>'
                f'<b>Student: {sid}</b>&nbsp;({stage_name})<br>'
                f'Score: <b>{pts_label}</b><br>'
                f'Walk from home: <b>{walk_m:.0f} m</b><br>'
                f'Node: {node_id}'
                f'</div>'
            )

            # Colour coding: chosen=bright mode colour, 2pts=dark, 1pt=mid, 0pt=light grey
            mode_c = _ROUTE_COLORS[mode_key][0]
            if is_chosen:
                fill_c = mode_c
                r = 5
                opacity = 0.9
            elif pts == 2:
                fill_c = '#333333'
                r = 4
                opacity = 0.75
            elif pts == 1:
                fill_c = '#888888'
                r = 3
                opacity = 0.65
            else:
                fill_c = '#bbbbbb'
                r = 3
                opacity = 0.50

            folium.CircleMarker(
                location=coords,
                radius=r,
                color=fill_c,
                fill=True,
                fillColor=fill_c,
                fillOpacity=opacity,
                weight=1,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"{sid} cand: {pts}pts, {walk_m:.0f}m walk",
            ).add_to(fg)

    fg.add_to(m)
    return fg


def _add_unserved_layer(m, sol, mode_key):
    """Add a FeatureGroup with X-pin markers for every unserved student in *sol*."""
    show = mode_key in ("A", "B")
    fg = FeatureGroup(name=f"{_MODE_NAMES[mode_key]} – Unserved Students", show=show)
    for student in sol.students:
        if getattr(student, 'is_served', False):
            continue
        stage_name = (
            student.school_stage.name
            if hasattr(student.school_stage, 'name')
            else str(student.school_stage)
        )
        popup_html = (
            f'<div style="width:220px;font-size:12px;">'
            f'<b style="color:#c0392b;">&#x2716; Unserved</b><br>'
            f'<b>Student: {student.id}</b><br>'
            f'Stage: {stage_name}<br>'
            f'Home: {student.coords[0]:.5f}, {student.coords[1]:.5f}<br>'
            f'Mode: {_MODE_NAMES[mode_key]}'
            f'</div>'
        )
        folium.Marker(
            location=student.coords,
            tooltip=f"{student.id} ({stage_name}) — UNSERVED",
            popup=folium.Popup(popup_html, max_width=260),
            icon=folium.Icon(color='red', icon='times', prefix='fa'),
        ).add_to(fg)
    fg.add_to(m)
    return fg


def _add_crossing_markers(m, crossings_dict, rejected_unsafe=None):
    """Collect mode crossings (+ optional rejected synthetic points) into one FeatureGroup."""
    all_cxs = [(mk, cx) for mk, cxs in crossings_dict.items() for cx in cxs]
    rejected_unsafe = list(rejected_unsafe or [])
    if not all_cxs and not rejected_unsafe:
        return None
    fg = FeatureGroup(name="Unsafe Crossings", show=True)
    seen = set()
    for mk, cx in all_cxs:
        lk = (round(cx["lat"], 6), round(cx["lon"], 6))
        if lk in seen:
            continue
        seen.add(lk)
        folium.CircleMarker(
            location=(cx["lat"], cx["lon"]), radius=6,
            color="red", fill=True, fillColor="yellow", fillOpacity=0.9, weight=2,
            tooltip=f"Unsafe crossing \u2013 {cx['student_id']}",
        ).add_to(fg)
    for rx in rejected_unsafe:
        lk = (round(float(rx.get("lat", 0.0)), 6), round(float(rx.get("lon", 0.0)), 6))
        if lk in seen:
            continue
        seen.add(lk)
        folium.CircleMarker(
            location=(float(rx.get("lat", 0.0)), float(rx.get("lon", 0.0))), radius=5,
            color="#8e44ad", fill=True, fillColor="#ffb74d", fillOpacity=0.9, weight=2,
            tooltip="Rejected synthetic crossing (unsafe road)",
        ).add_to(fg)
    fg.add_to(m)
    return fg


def _collect_used_synthetic_edge_keys(solutions, graph_for_walk):
    """Return synthetic walk-edge keys that are actually used by student walks."""
    walk_g = getattr(_eng, "_WALK_GRAPH", None)
    if walk_g is None:
        return set()

    used = set()
    for sol in (solutions or []):
        if sol is None:
            continue
        for route in sol.routes:
            for stop in route.stops:
                if getattr(stop, "stop_type", None) == "school":
                    continue
                for student in getattr(stop, "assigned_students", []):
                    sid = getattr(student, "id", None)
                    if sid is None:
                        continue
                    start_node = sol.student_locations.get(sid)
                    if start_node is None:
                        continue
                    path = walk_path_on_roads(graph_for_walk, start_node, stop.node_id)
                    if len(path) < 2:
                        continue
                    for a, b in zip(path, path[1:]):
                        ed = walk_g.get_edge_data(a, b) or walk_g.get_edge_data(b, a)
                        if not ed:
                            continue
                        data = ed[0] if 0 in ed else list(ed.values())[0]
                        if not data.get("synthetic_crossing", False):
                            continue
                        used.add((a, b) if str(a) < str(b) else (b, a))
    return used


def _add_synthetic_crossing_markers(m, crossings_list, show_only_used=False, used_edge_keys=None):
    """Add synthetic crossings to the map with strong visual style.

    If markers list is sparse (per-drive-node lazy mode), also derive crossings
    from walk-graph edges tagged with synthetic_crossing=True.

    When show_only_used=True, only synthetic edges touched by at least one
    served student's walk path are displayed.
    """
    walk_g = getattr(_eng, "_WALK_GRAPH", None)
    derived = []
    segs = []
    used_edge_keys = used_edge_keys or set()

    if walk_g is not None:
        for u, v, k, data in walk_g.edges(keys=True, data=True):
            if not data.get("synthetic_crossing", False):
                continue
            edge_key = (u, v) if str(u) < str(v) else (v, u)
            if show_only_used and edge_key not in used_edge_keys:
                continue
            if "geometry" in data:
                coords = [(lat, lon) for lon, lat in data["geometry"].coords]
            else:
                coords = [
                    (walk_g.nodes[u]["y"], walk_g.nodes[u]["x"]),
                    (walk_g.nodes[v]["y"], walk_g.nodes[v]["x"]),
                ]
            segs.append(coords)
            mid_lat = (coords[0][0] + coords[-1][0]) / 2
            mid_lon = (coords[0][1] + coords[-1][1]) / 2
            derived.append({"lat": mid_lat, "lon": mid_lon, "length_m": float(data.get("length", 0.0))})

    all_markers = list(crossings_list or []) + derived

    def _marker_lat_lon(cx):
        if not isinstance(cx, dict):
            return None
        if "lat" in cx and "lon" in cx:
            try:
                return float(cx.get("lat", 0.0)), float(cx.get("lon", 0.0))
            except Exception:
                return None
        if all(k in cx for k in ("lat_a", "lon_a", "lat_b", "lon_b")):
            try:
                lat = (float(cx["lat_a"]) + float(cx["lat_b"])) / 2.0
                lon = (float(cx["lon_a"]) + float(cx["lon_b"])) / 2.0
                return lat, lon
            except Exception:
                return None
        return None

    seen = set()
    uniq = []
    for cx in all_markers:
        lat_lon = _marker_lat_lon(cx)
        if lat_lon is None:
            continue
        lk = (round(lat_lon[0], 6), round(lat_lon[1], 6))
        if lk in seen:
            continue
        seen.add(lk)
        uniq.append(cx)

    fg = FeatureGroup(name=f"Synthetic Crossings ({len(uniq)})", show=False)
    for seg in segs:
        # Draw white halo first, then magenta line so crossings are obvious.
        folium.PolyLine(seg, color="#ffffff", weight=8, opacity=0.85).add_to(fg)
        folium.PolyLine(seg, color="#c2185b", weight=5, opacity=0.95).add_to(fg)

    for cx in uniq:
        lat_lon = _marker_lat_lon(cx)
        if lat_lon is None:
            continue
        lat, lon = lat_lon
        length_m = float(cx.get("length_m", 0.0))
        folium.CircleMarker(
            location=(lat, lon), radius=8,
            color="#4a148c", fill=True, fillColor="#ffeb3b", fillOpacity=0.95, weight=2,
            tooltip=f"Synthetic crossing ({length_m:.1f} m)",
        ).add_to(fg)
    fg.add_to(m)
    return fg


def _add_injected_crossings_layer(m, injected_payload):
    if not injected_payload:
        return None
    edge_pairs = injected_payload.get("edge_pairs", [])
    nodes = injected_payload.get("nodes", [])
    if not edge_pairs and not nodes:
        return None

    fg = FeatureGroup(name=f"Injected Crossings ({len(edge_pairs)})", show=False)

    node_pos = {}
    for node in nodes:
        node_id = node.get("node_id")
        if node_id is None:
            continue
        lat = float(node.get("lat", 0.0))
        lon = float(node.get("lon", 0.0))
        node_pos[node_id] = (lat, lon)
        kind = node.get("node_kind", "real")
        color = "purple" if kind == "synthetic" else "blue"
        fill = "#c39bd3" if kind == "synthetic" else "#85c1e9"
        folium.CircleMarker(
            location=(lat, lon),
            radius=4,
            color=color,
            fill=True,
            fillColor=fill,
            fillOpacity=0.85,
            tooltip=f"Injected node: {node_id} ({kind})",
        ).add_to(fg)

    for edge in edge_pairs:
        node_a = edge.get("node_a")
        node_b = edge.get("node_b")
        if node_a not in node_pos or node_b not in node_pos:
            continue
        crossing_type = edge.get("crossing_type", "real_to_real")
        road_name = edge.get("road_name", "?")
        length_m = float(edge.get("length_m", 0.0))
        if crossing_type == "real_to_synthetic":
            color = "darkorange"
            line_label = "Synthetic crossing"
        else:
            color = "gold"
            line_label = "Real crossing"

        seg = [node_pos[node_a], node_pos[node_b]]
        folium.PolyLine(seg, color="#ffffff", weight=7, opacity=0.8).add_to(fg)
        folium.PolyLine(
            seg,
            color=color,
            weight=4,
            opacity=0.95,
            tooltip=f"{line_label}: {road_name} ({length_m:.1f}m)",
        ).add_to(fg)

    fg.add_to(m)
    return fg


def _add_crossing_usage_layers(m, solutions_dict, G_walk, G_drive):
    """Add crossing usage visualization layers for each mode.

    For each mode (A, B, C), creates a FeatureGroup showing which synthetic
    crossings were actually used by students' walk paths.

    Args:
        m: Folium map object
        solutions_dict: Dict with keys A, B, C mapping to ServiceSolution objects
        G_walk: Walk graph with synthetic crossings
        G_drive: Drive graph for path finding

    Returns:
        Dict mapping mode key to FeatureGroup with crossing usage markers
    """
    from detour_engine import (
        get_crossing_usage_from_solution,
        _get_crossing_geometry_and_midpoint,
    )

    fgs_usage = {}
    mode_colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}  # Blue, Orange, Green

    for mode_key, solution in solutions_dict.items():
        if solution is None:
            continue

        try:
            # Get crossing usage for this mode's solution
            crossing_usage = get_crossing_usage_from_solution(solution, G_drive, G_walk)

            if not crossing_usage:
                continue

            mode_label = {"A": "Constrained", "B": "Unconstrained", "C": "Door-to-Door"}.get(mode_key, mode_key)
            fg = FeatureGroup(name=f"Crossing Usage – Mode {mode_key} ({len(crossing_usage)})", show=False)

            for (u, v), usage_data in crossing_usage.items():
                try:
                    geom_data = _get_crossing_geometry_and_midpoint(G_walk, u, v)
                    if not geom_data.get("coords"):
                        continue

                    # Draw halo and core
                    folium.PolyLine(
                        locations=geom_data["coords"],
                        color="#ffffff", weight=8, opacity=0.85
                    ).add_to(fg)
                    folium.PolyLine(
                        locations=geom_data["coords"],
                        color=mode_colors[mode_key], weight=5, opacity=0.95
                    ).add_to(fg)

                    # Build popup
                    student_ids_str = ", ".join(usage_data.get("students", [])[:10])
                    if len(usage_data.get("students", [])) > 10:
                        student_ids_str += f", ... +{len(usage_data['students']) - 10} more"

                    home_count = len(usage_data.get("homes", []))
                    popup_html = f"""
                    <div style="width: 300px; font-size: 11px;">
                        <b>Crossing in Mode {mode_key} ({mode_label})</b><br>
                        <hr style="margin: 3px 0;">
                        <b>Length:</b> {geom_data.get('length_m', 0):.1f} m<br>
                        <b>Location:</b> {geom_data.get('lat', 0):.5f}, {geom_data.get('lon', 0):.5f}<br>
                        <hr style="margin: 3px 0;">
                        <b>Students:</b> {student_ids_str}<br>
                        <b>Count:</b> {usage_data.get('count', 0)} students from {home_count} homes<br>
                        <b>Impact:</b> These students can reach stops on opposite side of dual carriageway
                    </div>
                    """

                    folium.CircleMarker(
                        location=(geom_data.get("lat", 0), geom_data.get("lon", 0)),
                        radius=7,
                        color=mode_colors[mode_key],
                        fill=True,
                        fillColor="#ffeb3b",
                        fillOpacity=0.85,
                        weight=2,
                        popup=folium.Popup(popup_html, max_width=400),
                        tooltip=f"Mode {mode_key}: {usage_data.get('count', 0)} students use this crossing"
                    ).add_to(fg)

                except Exception as e:
                    print(f"  Warning: Could not render crossing in Mode {mode_key}: {e}")
                    continue

            fg.add_to(m)
            fgs_usage[mode_key] = fg
            print(f"  Crossing usage – Mode {mode_key}: {len(crossing_usage)} crossings with student usage")

        except Exception as e:
            print(f"  Warning: Could not extract crossing usage for Mode {mode_key}: {e}")
            continue

    return fgs_usage


def _build_custom_layer_control_js(
    map_var, fg_danger, fg_unclass, fg_walknet, fg_syn_cross,
    fgs_a, fgs_b, fgs_c,
    fg_crossings,
    fg_injected=None,
    syn_label=None,
    injected_label=None,
    fg_unserved_a=None, fg_unserved_b=None, fg_unserved_c=None,
    fg_cands_a=None,   fg_cands_b=None,   fg_cands_c=None,
    fg_usage_a=None,   fg_usage_b=None,   fg_usage_c=None,
):
    """Return JS that adds a titled, grouped layer-control widget to the map.

    Uses Folium's .get_name() to get each feature-group's actual JS variable
    name, so the control works on every freshly generated map.
    """
    va_r, va_w = fgs_a[0].get_name(), fgs_a[1].get_name()
    vb_r, vb_w = fgs_b[0].get_name(), fgs_b[1].get_name()
    vc_r, vc_w = fgs_c[0].get_name(), fgs_c[1].get_name()
    v_danger  = fg_danger.get_name()
    v_unclass = fg_unclass.get_name()
    v_walknet = fg_walknet.get_name()
    v_syn = fg_syn_cross.get_name()
    syn_label = syn_label or "Synthetic Crossings"
    injected_label = injected_label or "Injected Crossings"

    injected_row = ""
    if fg_injected is not None:
        vi = fg_injected.get_name()
        injected_row = f"""
                row('{injected_label}',
                    [{vi}],
                    map.hasLayer({vi}));"""

    crossings_row = ""
    if fg_crossings is not None:
        vc = fg_crossings.get_name()
        crossings_row = f"""
                row('Unsafe Crossings',
                    [{vc}],
                    map.hasLayer({vc}));"""

    unserved_rows = ""
    for fg_u, label in [
        (fg_unserved_a, 'Constrained – Unserved'),
        (fg_unserved_b, 'Unconstrained – Unserved'),
        (fg_unserved_c, 'Door-to-Door – Unserved'),
    ]:
        if fg_u is not None:
            vu = fg_u.get_name()
            unserved_rows += f"""
                row('{label}',
                    [{vu}],
                    map.hasLayer({vu}));"""

    candidate_rows = ""
    for fg_c2, label in [
        (fg_cands_a, 'Constrained – Candidate Stops'),
        (fg_cands_b, 'Unconstrained – Candidate Stops'),
        (fg_cands_c, 'Door-to-Door – Candidate Stops'),
    ]:
        if fg_c2 is not None:
            vca = fg_c2.get_name()
            candidate_rows += f"""
                row('{label}',
                    [{vca}],
                    map.hasLayer({vca}));"""

    usage_rows = ""
    for fg_u, label in [
        (fg_usage_a, 'Constrained – Crossing Usage'),
        (fg_usage_b, 'Unconstrained – Crossing Usage'),
        (fg_usage_c, 'Door-to-Door – Crossing Usage'),
    ]:
        if fg_u is not None:
            vu = fg_u.get_name()
            usage_rows += f"""
                row('{label}',
                    [{vu}],
                    map.hasLayer({vu}));"""

    return f"""
    window.addEventListener('load', function() {{
        var m = {map_var};
        var CustomCtrl = L.Control.extend({{
            options: {{ position: 'topright' }},
            onAdd: function(map) {{
                var c = L.DomUtil.create('div',
                    'leaflet-control-layers leaflet-control-layers-expanded');
                c.style.cssText =
                    'padding:10px 14px;min-width:210px;font-size:13px;' +
                    'font-family:Arial,sans-serif;line-height:1.4;' +
                    'max-height:calc(100vh - 320px);overflow-y:auto;';
                L.DomEvent.disableClickPropagation(c);
                L.DomEvent.disableScrollPropagation(c);
                // title
                var title = L.DomUtil.create('div', '', c);
                title.innerHTML = 'Route Views';
                title.style.cssText =
                    'font-weight:bold;font-size:14px;' +
                    'margin-bottom:8px;padding-bottom:6px;' +
                    'border-bottom:2px solid #bbb;';
                function sep() {{
                    var d = L.DomUtil.create('div', '', c);
                    d.style.cssText = 'border-top:1px solid #e0e0e0;margin:5px 0;';
                }}
                function row(label, layers, on) {{
                    var lbl = L.DomUtil.create('label', '', c);
                    lbl.style.cssText =
                        'display:flex;align-items:center;gap:7px;' +
                        'margin:5px 0;cursor:pointer;';
                    var cb = document.createElement('input');
                    cb.type = 'checkbox';
                    cb.checked = on;
                    cb.style.cssText =
                        'width:14px;height:14px;cursor:pointer;flex-shrink:0;';
                    cb.addEventListener('change', function() {{
                        layers.forEach(function(fg) {{
                            cb.checked ? map.addLayer(fg) : map.removeLayer(fg);
                        }});
                    }});
                    lbl.appendChild(cb);
                    var span = document.createElement('span');
                    span.textContent = label;
                    lbl.appendChild(span);
                }}
                row('Constrained (Safe Walking)',
                    [{va_r}, {va_w}], map.hasLayer({va_r}));
                row('Unconstrained (Any Walking)',
                    [{vb_r}, {vb_w}], map.hasLayer({vb_r}));
                row('Direct (No Walking)',
                    [{vc_r}, {vc_w}], map.hasLayer({vc_r}));
                sep();
                row('Dangerous Roads (unsafe to cross)',
                    [{v_danger}], map.hasLayer({v_danger}));
                row('Unclassified Roads (no student placement)',
                    [{v_unclass}], map.hasLayer({v_unclass}));
                row('Safe Walking Network (cyan streets)',
                    [{v_walknet}], map.hasLayer({v_walknet}));
                row('{syn_label}',
                    [{v_syn}], map.hasLayer({v_syn}));{injected_row}{crossings_row}{usage_rows}{unserved_rows}
                sep();
                var hdr2 = L.DomUtil.create('div', '', c);
                hdr2.textContent = 'Candidate Stop Inspector';
                hdr2.style.cssText =
                    'font-weight:bold;font-size:12px;margin:6px 0 2px;color:#555;';
                {candidate_rows}
                return c;
            }}
        }});
        new CustomCtrl().addTo(m);
    }});
    """


def _build_stats_html(all_stats, crossings_dict, occupancies_dict,
                      solutions_dict=None, G=None, constraints=None,
                      meta=None):
    now    = datetime.datetime.now()
    hour12 = now.hour % 12 or 12
    ampm   = "am" if now.hour < 12 else "pm"
    ts     = now.strftime("%d/%m/%y") + f" {hour12:02d}:{now.strftime('%M')} {ampm}"

    algo = (meta or {}).get("algorithm", {}) if meta else {}
    buses_cfg = (meta or {}).get("buses", {}) if meta else {}
    synth_cfg = (meta or {}).get("synthetic_crossings", {}) if meta else {}
    caps_on = (constraints or {}).get("enabled", True)
    soft_caps = (constraints or {}).get("soft_ride_caps", False)
    crossings_enabled = bool(synth_cfg.get("enabled", False))
    time_budget = algo.get("time_budget_seconds", None)
    max_cands = algo.get("max_candidates_per_student", None)
    buses_count = buses_cfg.get("count", None)
    bus_capacity = buses_cfg.get("capacity", None)
    minimize_buses = bool(algo.get("minimize_buses", False))

    blocks = ""
    _build_stats_html._mode_tables = ""   # accumulator for side-by-side mode tables
    for mk in ("A", "B", "C"):
        mc = _ROUTE_COLORS[mk][0]

        # Mode was skipped — show a dimmed placeholder
        if mk not in all_stats or all_stats[mk] is None:
            blocks += f"""
        <div style="margin-bottom:8px; padding-bottom:8px;
                    border-bottom:1px solid #e0e0e0;">
          <div style="font-weight:bold; color:#aaa; margin-bottom:3px;">
            {mk}: {_MODE_NAMES[mk]}
          </div>
          <div style="font-size:11px; color:#bbb; font-style:italic;">skipped (debug.run_mode_{mk.lower()}=false)</div>
        </div>"""
            continue
        s   = all_stats[mk]
        cx  = len(crossings_dict.get(mk, []))
        occ = occupancies_dict.get(mk, [])
        cx_color = "#c0392b" if cx > 0 else "#27ae60"
        mc = _ROUTE_COLORS[mk][0]

        # Avg occupancy as % of bus capacity
        sol = (solutions_dict or {}).get(mk)
        active_routes = [r for r in sol.routes if r.get_student_count() > 0] if sol else []
        cap = active_routes[0].bus.capacity if active_routes else None
        if occ and cap:
            avg_occ_str = f"{(sum(occ) / len(occ) / cap * 100):.0f}%"
        elif occ:
            avg_occ_str = f"{(sum(occ) / len(occ)):.1f}"
        else:
            avg_occ_str = "—"

        sat_by_route = s.get("sat_by_route", {})

        fleet_summary = s.get("fleet_search_summary")
        buses_used = s.get("buses_used")
        fleet_line = ""
        if buses_used is not None and buses_count is not None:
            cap_str = f" (cap {bus_capacity})" if bus_capacity is not None else ""
            fleet_line = f"<div style=\"font-size:11px; color:#666;\">Fleet: used {buses_used}/{buses_count} buses{cap_str}</div>"
        elif buses_count is not None:
            cap_str = f" (cap {bus_capacity})" if bus_capacity is not None else ""
            fleet_line = f"<div style=\"font-size:11px; color:#666;\">Fleet: {buses_count} buses{cap_str}</div>"
        fleet_note = f"<div style=\"font-size:11px; color:#777; margin-top:2px;\">{fleet_summary}</div>" if fleet_summary else ""
        fleet_cell = "-"
        if buses_used is not None and buses_count is not None:
            fleet_cell = f"{buses_used}/{buses_count}"
        elif buses_count is not None:
            fleet_cell = f"{buses_count}"
        
        cap_viol = s.get("cap_violations_am")
        cap_checked = s.get("cap_checked_am")
        cap_pct = s.get("cap_violation_pct_am")
        cap_pm_viol = s.get("cap_violations_pm")
        cap_pm_checked = s.get("cap_checked_pm")
        cap_pm_pct = s.get("cap_violation_pct_pm")
        cap_line = ""
        if cap_viol is not None or cap_pm_viol is not None:
            am_pct_str = f" ({cap_pct}%)" if cap_pct is not None else ""
            pm_pct_str = f" ({cap_pm_pct}%)" if cap_pm_pct is not None else ""
            am_str = f"AM {cap_viol}/{cap_checked}{am_pct_str}" if cap_viol is not None else "AM —"
            pm_str = f"PM {cap_pm_viol}/{cap_pm_checked}{pm_pct_str}" if cap_pm_viol is not None else "PM —"
            cap_line = (
                f"<div style=\"font-size:11px; color:#777; margin-top:2px;\">"
                f"Cap violations: {am_str} | {pm_str}"
                f"</div>"
            )

        blocks += f"""
        <div style="margin-bottom:8px; padding-bottom:8px;
                    border-bottom:1px solid #e0e0e0;">
          <div style="font-weight:bold; color:{mc}; margin-bottom:3px;">
            {mk}: {_MODE_NAMES[mk]}
          </div>
          <table style="width:100%; border-collapse:collapse;
                        font-size:11px; text-align:center;">
            <tr style="color:#555;">
              <td style="text-align:left; padding:1px 4px;">Routes</td>
              <td style="text-align:left; padding:1px 4px;">Fleet</td>
              <td style="text-align:left; padding:1px 4px;">Time (Bus+Walk)</td>
              <td style="text-align:left; padding:1px 4px;">Distance</td>
              <td style="text-align:left; padding:1px 4px;">Avg Occ.</td>
              <td style="text-align:left; padding:1px 4px;">Served</td>
              <td style="text-align:left; padding:1px 4px;">Satisfied</td>
              <td style="text-align:left; padding:1px 4px;">Crossings</td>
            </tr>
            <tr style="font-weight:bold;">
              <td style="padding:1px 4px;">{s['routes']}</td>
              <td style="padding:1px 4px;">{fleet_cell}</td>
              <td style="padding:1px 4px;">{s['total_time']:.0f}+{s.get('walk_stats', {}).get('avg_walk_time_min', 0) * s['served']:.0f} min</td>
              <td style="padding:1px 4px;">{s['total_dist']:.1f} km</td>
              <td style="padding:1px 4px;">{avg_occ_str}</td>
              <td style="padding:1px 4px;">{s['served']}/{s['total']}</td>
              <td style="padding:1px 4px;">{s.get('satisfied', '—')}/{s['served']}</td>
              <td style="padding:1px 4px; color:{cx_color};">{cx}</td>
            </tr>
          </table>
          {fleet_line}
          {fleet_note}
          {cap_line}
        </div>"""

        # Per-mode mini-table for the side-by-side horizontal layout
        th = "padding:1px 4px; text-align:right; border-bottom:1px solid #ccc; white-space:nowrap;"
        th_l = "padding:1px 4px; text-align:left; border-bottom:1px solid #ccc; white-space:nowrap;"
        td_r = "padding:1px 4px; text-align:right; border-bottom:1px solid #f0f0f0;"
        td_l = "padding:1px 4px; text-align:left;  border-bottom:1px solid #f0f0f0;"
        rows_html = ""
        if sol:
            for route in sorted(active_routes, key=lambda r: r.route_id):
                sc    = route.get_student_count()
                cap_r = route.bus.capacity
                sat_r = sat_by_route.get(route.route_id, "—")
                rows_html += f"""
              <tr>
                <td style="{td_l}">{route.route_id}</td>
                <td style="{td_r}">{route.total_distance:.1f}</td>
                <td style="{td_r}">{route.total_time:.0f}</td>
                <td style="{td_r}">{sc}/{cap_r}</td>
                <td style="{td_r}">{sat_r}/{sc}</td>
              </tr>"""
        mode_tables_html = getattr(_build_stats_html, '_mode_tables', "")
        mode_tables_html += f"""
          <table style="border-collapse:collapse; font-size:10px; white-space:nowrap;
                        margin-right:10px; vertical-align:top; display:inline-table;">
            <thead>
              <tr style="background:#f5f5f5; color:{mc};">
                <th colspan="5" style="padding:1px 4px; text-align:left;
                    border-bottom:1px solid #ccc; font-size:10px;">{mk}</th>
              </tr>
              <tr style="background:#f5f5f5; color:#555;">
                <th style="{th_l}">Route</th>
                <th style="{th}">Dist (km)</th>
                <th style="{th}">Time (m)</th>
                <th style="{th}">Occ</th>
                <th style="{th}">Sat</th>
              </tr>
            </thead>
            <tbody>{rows_html}
            </tbody>
          </table>"""
        _build_stats_html._mode_tables = mode_tables_html

    mode_tables_html = getattr(_build_stats_html, '_mode_tables', "")
    _build_stats_html._mode_tables = ""   # reset for next call

    # Three mode mini-tables placed side-by-side; single horizontal scrollbar at bottom
    route_table = f"""
      <div style="margin-top:6px; padding-top:6px; border-top:1px solid #ddd;">
        <div style="font-size:11px; font-weight:bold; color:#444; margin-bottom:3px;">Per-Route Details</div>
        <div style="overflow-x:auto; white-space:nowrap;">
          {mode_tables_html}
        </div>
      </div>"""

    algo_lines = []
    algo_lines.append(f"Caps: {'ON' if caps_on else 'OFF'}")
    algo_lines.append(f"SoftCaps: {'ON' if soft_caps else 'OFF'}")
    algo_lines.append(f"MinFleet: {'ON' if minimize_buses else 'OFF'}")
    algo_lines.append(f"Crossings: {'ON' if crossings_enabled else 'OFF'}")
    if time_budget is not None:
        algo_lines.append(f"Budget: {time_budget}s")
    if max_cands is not None:
        algo_lines.append(f"Candidates: {max_cands}")
    if buses_count is not None:
        cap_str = f" (cap {bus_capacity})" if bus_capacity is not None else ""
        algo_lines.append(f"Fleet: {buses_count}{cap_str}")
    algo_line = " | ".join(algo_lines)

    return f"""
    <div style="position:fixed; bottom:15px; right:15px; width:430px;
                max-height:260px; overflow-y:auto;
                background:white; border:2px solid #555; z-index:9999;
                padding:12px 14px; border-radius:6px; font-size:12px;
                font-family:Arial,sans-serif; box-shadow:2px 2px 8px rgba(0,0,0,.25);">
            <div style="font-weight:bold; font-size:13px; margin-bottom:10px;
                  padding-bottom:6px; border-bottom:2px solid #ccc;">
        Three-Mode Routing Comparison
      </div>
            <div style="font-size:11px; color:#666; margin-bottom:8px;">
                {algo_line}
            </div>
      {blocks}
      {route_table}
      <div style="font-size:10px; color:#888; margin-top:6px;">
        Toggle layers via top-right control.<br>
        <span style="color:#e74c3c;">&#x2015;&#x2015;</span> Dangerous roads
        &nbsp;&nbsp;
        <span style="color:#7f8c8d;">&#x2508;&#x2508;</span> Unclassified roads<br>
        Generated: {ts}
      </div>
    </div>
    """


# ────────────────────────────────────────────────────────────────────
# METRICS HELPERS
# ────────────────────────────────────────────────────────────────────
_WALK_SPEED_M_PER_MIN = 80.0  # comfortable pedestrian (≈ 4.8 km/h)


def _compute_walk_stats(sol, G, stage_walk):
    """Return walk-distance statistics for one solution.

    Uses ``walk_distance_on_roads`` for road-network accuracy, falling back
    to straight-line Haversine when the path isn't found.
    """
    dists = []
    utils = []
    for route in sol.routes:
        for stop in route.stops:
            if stop.stop_type == "school":
                continue
            for student in stop.students:
                s_node = _eng.fast_nearest_node(G, student.coords[1], student.coords[0])
                d = walk_distance_on_roads(G, s_node, stop.node_id)
                if d <= 0:          # fallback: straight-line
                    dlat = math.radians(stop.coords[0] - student.coords[0])
                    dlon = math.radians(stop.coords[1] - student.coords[1])
                    a = (math.sin(dlat / 2) ** 2
                         + math.cos(math.radians(student.coords[0]))
                         * math.cos(math.radians(stop.coords[0]))
                         * math.sin(dlon / 2) ** 2)
                    d = 6_371_000 * 2 * math.asin(math.sqrt(a))
                dists.append(d)
                # utilisation = fraction of walk budget actually used
                stage_name = (
                    student.school_stage.name
                    if hasattr(student.school_stage, "name")
                    else str(student.school_stage)
                )
                walk_max = stage_walk.get(stage_name, 0)
                if walk_max > 0:
                    utils.append(min(d / walk_max, 1.0))

    if not dists:
        return {"avg_walk_dist_m": 0, "median_walk_dist_m": 0,
                "max_walk_dist_m": 0, "min_walk_dist_m": 0,
                "avg_walk_time_min": 0, "avg_walk_utilisation_pct": None}
    return {
        "avg_walk_dist_m":        round(statistics.mean(dists),   1),
        "median_walk_dist_m":     round(statistics.median(dists), 1),
        "max_walk_dist_m":        round(max(dists),               1),
        "min_walk_dist_m":        round(min(dists),               1),
        "avg_walk_time_min":      round(statistics.mean(dists) / _WALK_SPEED_M_PER_MIN, 2),
        "avg_walk_utilisation_pct": round(statistics.mean(utils) * 100, 1) if utils else None,
    }


def _build_metrics(meta, stage_walk, all_stats, crossings_dict,
                   sol_a, sol_b, sol_c, G_unc, iters, total_wall=None,
                   step_times=None, mode_wall_times=None):
    """Assemble the full metrics dict that will be written to metrics.json."""
    mode_map = {
        "constrained":   ("A", sol_a),
        "unconstrained": ("B", sol_b),
        "door_to_door":  ("C", sol_c),
    }
    modes_out = {}
    for mode_key, (mk, sol) in mode_map.items():
        if sol is None or mk not in all_stats:
            modes_out[mode_key] = {"skipped": True}
            continue
        s   = all_stats[mk]
        cx  = len(crossings_dict.get(mk, []))
        n_routes = s["routes"]
        # walk stats: door-to-door has walk_radius=0, so no utilisation
        sw = stage_walk if mode_key != "door_to_door" else {k: 0 for k in stage_walk}
        walk = _compute_walk_stats(sol, G_unc, sw)
        
        # Build student list with ride time, direct potential, walk distance
        students_list = []
        for route in sol.routes:
            # Pre-compute per-stop ride-time using matrix-cache summation
            # (safe against cleared caches; falls back to lazy A* on miss)
            school_node = route.stops[-1].node_id
            for stop in route.stops:
                if stop.stop_type == "school":
                    continue

                stop_idx = next((i for i, s in enumerate(route.stops) if s is stop), -1)
                if stop_idx == -1:
                    continue

                # Ride time = sum of legs from this stop to school (matrix-safe)
                ride_time = calculate_route_time_from_matrix(route.stops[stop_idx:], G_unc)
                if ride_time >= 9999:
                    ride_time = None  # truly unreachable; treat as unknown

                for student in stop.students:
                    stage_name = (
                        student.school_stage.name
                        if hasattr(student.school_stage, "name")
                        else str(student.school_stage)
                    )

                    # Direct time home -> school (cached on student after first call)
                    direct_time = compute_direct_time(student, school_node, G_unc)
                    if not math.isfinite(direct_time):
                        direct_time = None

                    # Walk distance home -> assigned stop
                    s_node = _eng.fast_nearest_node(G_unc, student.coords[1], student.coords[0])
                    walk_dist = walk_distance_on_roads(G_unc, s_node, stop.node_id)
                    if walk_dist <= 0 or not math.isfinite(walk_dist):
                        walk_dist = 0.0

                    students_list.append({
                        "id": student.id,
                        "stage": stage_name,
                        "ride_time_min": round(ride_time, 2) if ride_time is not None else None,
                        "direct_potential_min": round(direct_time, 2) if direct_time is not None else None,
                        "walk_distance_m": round(walk_dist, 1),
                    })
        
        buses_cfg = meta.get("buses", {})
        buses_available = buses_cfg.get("count")
        bus_capacity = buses_cfg.get("capacity")
        buses_used = s.get("buses_used")
        if buses_used is None and buses_available is not None:
            buses_used = buses_available

        mode_entry = {
            "routes_created":       n_routes,
            "students_served":      s["served"],
            "students_unserved":    s["total"] - s["served"],
            "total_route_time_min": round(s["total_time"], 2),
            "total_route_dist_km":  round(s["total_dist"],  2),
            "avg_route_time_min":   round(s["total_time"] / n_routes, 2) if n_routes else 0,
            "alns_runtime_seconds": round(s["runtime"],     2),
            "mode_wall_time_seconds": s.get("mode_wall_time"),
            "unsafe_crossings":     cx,
            "walk_stats":           walk,
            "students":             students_list,
            "buses_available":       buses_available,
            "bus_capacity":          bus_capacity,
            "buses_used":            buses_used,
            "ride_cap_violations_am": s.get("cap_violations_am"),
            "ride_cap_checked_am":    s.get("cap_checked_am"),
            "ride_cap_violation_pct_am": s.get("cap_violation_pct_am"),
            "ride_cap_violations_pm": s.get("cap_violations_pm"),
            "ride_cap_checked_pm":    s.get("cap_checked_pm"),
            "ride_cap_violation_pct_pm": s.get("cap_violation_pct_pm"),
        }

        # Calculate total walking time across all students
        if walk and "avg_walk_time_min" in walk:
            total_students = s["served"]
            total_walk_time = walk["avg_walk_time_min"] * total_students
            total_time_with_walks = s["total_time"] + total_walk_time
            mode_entry["total_walk_time_min"] = round(total_walk_time, 2)
            mode_entry["total_time_with_walks_min"] = round(total_time_with_walks, 2)
        else:
            mode_entry["total_walk_time_min"] = 0
            mode_entry["total_time_with_walks_min"] = round(s["total_time"], 2)

        # Attach fleet-search diagnostics if present
        if s.get("fleet_search_log"):
            mode_entry["fleet_search"] = {
                "buses_used":    s.get("buses_used"),
                "summary":       s.get("fleet_search_summary"),
                "search_log":    s["fleet_search_log"],
            }
        modes_out[mode_key] = mode_entry

    # cross-mode comparisons (guard against skipped modes)
    def _mget(mode_key, field, default=None):
        entry = modes_out.get(mode_key, {})
        return entry.get(field, default) if not entry.get("skipped") else default

    t_con  = _mget("constrained",   "total_route_time_min", 0)
    t_unc  = _mget("unconstrained", "total_route_time_min", 0)
    t_d2d  = _mget("door_to_door",  "total_route_time_min", 0)
    cx_con = _mget("constrained",   "unsafe_crossings", 0)
    cx_unc = _mget("unconstrained", "unsafe_crossings", 0)

    # Build per-mode debug breakdown
    _dbg_modes = {}
    for _mk, _sk, _sol in [("A", "constrained", sol_a), ("B", "unconstrained", sol_b), ("C", "door_to_door", sol_c)]:
        _s = all_stats.get(_mk)
        _wt = (mode_wall_times or {}).get(_mk)
        if _s and _wt is not None:
            _alns_t = round(_s.get("runtime", 0), 2)
            _dbg_modes[_mk] = {
                "mode_wall_time_s":        _wt,
                "alns_solve_s":            _alns_t,
                "setup_and_overhead_s":    round(_wt - _alns_t, 2),
                "alns_iterations":         _s.get("iterations"),
                "n_candidates_per_student": round(
                    sum(len(v) for v in (getattr(_alns, '_student_candidate_cache', None) or {}).values())
                    / max(1, _s.get("total", 1)), 1
                ) if _sol else None,
            }
        else:
            _dbg_modes[_mk] = {"skipped": True}

    _debug_stats = {
        "step_times": step_times or {},
        "mode_breakdown": _dbg_modes,
        "synthetic_crossings": _eng.get_synthetic_diagnostics(),
    }

    return {
        "generated_at": (lambda n: n.strftime("%d/%m/%y") + f" {n.hour%12 or 12:02d}:{n.strftime('%M')} {'am' if n.hour<12 else 'pm'}")(datetime.datetime.now()),
        "total_wall_time_seconds": total_wall,
        "debug_stats": _debug_stats,
        "config": {
            "n_students":       meta.get("n_students"),
            "seed":             meta.get("seed"),
            "iterations":       iters,
            "buses_count":      meta.get("buses", {}).get("count"),
            "buses_capacity":   meta.get("buses", {}).get("capacity"),
            "minimize_buses":   meta.get("algorithm", {}).get("minimize_buses", False),
            "force_fleet_size": meta.get("algorithm", {}).get("force_fleet_size"),
            "constraints_enabled": meta.get("constraints", {}).get("enabled", True),
            "soft_ride_caps": meta.get("constraints", {}).get("soft_ride_caps", False),
            "time_budget_seconds": meta.get("algorithm", {}).get("time_budget_seconds"),
            "max_candidates_per_student": meta.get("algorithm", {}).get("max_candidates_per_student"),
            "stage_walk_limits": stage_walk,
            "stage_distribution": {
                k: v for k, v in meta.get("stage_distribution", {}).items()
                if k != "_comment"
            },
            "constraints": {
                "enabled": meta.get("constraints", {}).get("enabled", True),
                "soft_ride_caps": meta.get("constraints", {}).get("soft_ride_caps", False),
                "ride_time_multiplier": meta.get("constraints", {}).get("ride_time_multiplier"),
                "floor_minutes": meta.get("constraints", {}).get("floor_minutes"),
                "ceiling_minutes": meta.get("constraints", {}).get("ceiling_minutes"),
                "bidirectional_check": meta.get("constraints", {}).get("bidirectional_check"),
            },
            "algorithm": {
                "time_budget_seconds": meta.get("algorithm", {}).get("time_budget_seconds"),
                "max_candidates_per_student": meta.get("algorithm", {}).get("max_candidates_per_student"),
                "minimize_buses": meta.get("algorithm", {}).get("minimize_buses", False),
                "force_fleet_size": meta.get("algorithm", {}).get("force_fleet_size"),
            },
        },
        "modes": modes_out,
        "comparison": {
            "efficiency_gain_vs_d2d_pct": (
                round((t_d2d - t_con) / t_d2d * 100, 1) if t_d2d else None
            ),
            "safety_cost_vs_unconstrained_pct": (
                round((t_con - t_unc) / t_unc * 100, 1) if t_unc else None
            ),
            "crossings_eliminated_vs_unconstrained": cx_unc - cx_con,
            "constrained_total_time_min":   t_con,
            "unconstrained_total_time_min": t_unc,
            "door_to_door_total_time_min":  t_d2d,
        },
    }


def _sanitise_floats(obj):
    """Recursively replace non-finite floats with None so json.dump stays valid."""
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitise_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise_floats(v) for v in obj]
    return obj


# ────────────────────────────────────────────────────────────────────
# PUBLIC API  (callable from thin launchers)
# ────────────────────────────────────────────────────────────────────
def run(input_path=None, output_path=None, iterations=None):
    """Run the three-mode comparison and save the map.

    Parameters
    ----------
    input_path  : str | None
        Path to an input.json file.  Defaults to the bundled input.json
        sitting next to this script.
    output_path : str | None
        Absolute path for the output HTML map.  Defaults to
        ``comparison_map.html`` next to the input file.
    iterations : int | None
        Override ALNS iteration count from input.json.
    """
    import time as _wtime
    _run_start = _wtime.time()

    meta           = _load_meta(input_path)
    iters          = iterations or meta.get("algorithm", {}).get("iterations", 30)
    algo_cfg       = meta.get("algorithm", {})
    iters          = iterations or algo_cfg.get("iterations", 30)
    minimize_buses = algo_cfg.get("minimize_buses", False)
    force_k        = algo_cfg.get("force_fleet_size")

    # ── Debug / partial-run flags ──
    _dbg       = meta.get("debug", {})
    run_mode_a = bool(_dbg.get("run_mode_a", True))
    run_mode_b = bool(_dbg.get("run_mode_b", True))
    run_mode_c = bool(_dbg.get("run_mode_c", True))
    _active_modes = [m for m, en in [("A", run_mode_a), ("B", run_mode_b), ("C", run_mode_c)] if en]

    # Resolve where to write the map
    if output_path:
        output = output_path
    else:
        rel = meta.get("output", "comparison_map.html")
        base = os.path.dirname(input_path) if input_path else _SCRIPT_DIR
        output = rel if os.path.isabs(rel) else os.path.join(base, rel)

    school_cfg = meta["school"]
    raw_walk = meta.get("stage_walk_limits", DEFAULT_STAGE_WALK_LIMITS)
    # Filter out _comment and other non-stage keys
    stage_walk = {k: v for k, v in raw_walk.items()
                  if k in ("KG", "ELEMENTARY", "MIDDLE", "HIGH")}

    print("=" * 60)
    print("  THREE-MODE ROUTING COMPARISON  (input.json)")
    print("=" * 60)
    print(f"  Students : {meta['n_students']}")
    print(f"  Seed     : {meta['seed']}")
    print(f"  Stages   : {meta['stage_distribution']}")
    print(f"  Walk lim : {stage_walk}")
    print(f"  Iters    : {iters}")
    _mode_labels = {"A": "Constrained", "B": "Unconstrained", "C": "Door-to-Door"}
    _skipped = [m for m in ("A", "B", "C") if m not in _active_modes]
    _mode_wall_times = {}
    _step_times: dict = {}
    print(f"  Running  : {', '.join(_mode_labels[m] for m in _active_modes)}")
    if _skipped:
        print(f"  Skipping : {', '.join(_mode_labels[m] for m in _skipped)} (debug flags)")
    print()

    # Seed random for reproducible ALNS runs
    seed_val = meta.get('seed', 42)
    random.seed(seed_val)
    import numpy as np
    np.random.seed(seed_val)
    print(f"  Random seed: {seed_val} (ALNS deterministic)")

    # ── 1. Generate dataset ──
    print("[1/7] Generating dataset …")
    _t0 = _wtime.time()
    base_data = _generate_dataset(meta)
    base_data["meta"]["algorithm"]["iterations"] = iters
    _step_times["generate_dataset_s"] = round(_wtime.time() - _t0, 2)

    # Print stage breakdown
    stage_counts = {}
    for s in base_data["data"]["students"]:
        stage_counts[s["school_stage"]] = stage_counts.get(s["school_stage"], 0) + 1
    print(f"  Stage breakdown: {stage_counts}")

    # ── 2. Constrained graph ──
    print("\n[2/7] Building CONSTRAINED graph …")
    _t0 = _wtime.time()
    G_con = setup_graph(base_data["meta"], unconstrained=False)
    _prebuild_ball_tree(G_con)
    _step_times["build_constrained_graph_s"] = round(_wtime.time() - _t0, 2)

    # ── 3. Unconstrained graph ──
    # NOTE: setup_graph() always loads a fresh graph from pickle — G_unc and
    # G_con are fully independent objects.  No deepcopy of G_con is needed.
    print("[3/7] Building UNCONSTRAINED graph …")
    _t0 = _wtime.time()
    G_unc = setup_graph(base_data["meta"], unconstrained=True)
    _eng._BALL_TREE = None
    _eng._BALL_TREE_GRAPH_ID = None
    _eng._BALL_TREE_NODE_IDS = None
    _step_times["build_unconstrained_graph_s"] = round(_wtime.time() - _t0, 2)

    # ── 3b. Optional walking graph (cached) ──
    walk_cfg = meta.get("walk_graph", {}) if isinstance(meta, dict) else {}
    use_walk_graph = bool(walk_cfg.get("enabled", False))
    injection_cfg = meta.get("crossings_injection", {}) if isinstance(meta, dict) else {}
    injection_enabled = bool(injection_cfg.get("enabled", False))
    injection_strict = bool(injection_cfg.get("strict_validation", False))
    injection_pkl_path = _resolve_injection_pkl_path(injection_cfg, input_path)
    synth_cfg = (meta.get("synthetic_crossings") if isinstance(meta, dict) else None) or {
        "enabled": False,
        "strategy": "per_drive_node",
        "max_per_drive_node": 1,
        "min_dist_m": 6.0,
        "max_dist_m": 20.0,
        "max_per_node": 1,
        "max_total": 1000,
        "radius_km": 2.0,
        "exclude_unsafe_roads": True,
    }
    G_walk = None
    syn_list = []
    injected_payload = None
    injected_result = None
    if use_walk_graph:
        print("[3b/7] Building WALK graph...")
        _t0 = _wtime.time()
        walk_radius_km = float(walk_cfg.get("radius_km", 5.0))
        G_walk = setup_walk_graph(
            base_data["meta"],
            center=(school_cfg["latitude"], school_cfg["longitude"]),
            radius_m=walk_radius_km * 1000.0,
        )
        synth_cfg["center_lat"] = school_cfg["latitude"]
        synth_cfg["center_lon"] = school_cfg["longitude"]
        synth_cfg.setdefault("radius_km", walk_radius_km)
        # Save clean walk graph BEFORE crossings are added (set_walk_graph modifies in-place)
        import copy as _copy_mod
        _clean_walk_graph = _copy_mod.deepcopy(G_walk)

        if injection_enabled and injection_pkl_path:
            print(f"  [Crossings Injection] Loading: {injection_pkl_path}")
            try:
                injected_payload = _load_crossings_injection_payload(injection_pkl_path)
                _validate_crossings_injection_payload(injected_payload, synth_cfg)
                _eng.set_walk_graph(G_walk, synthetic_cfg={"enabled": False})
                injected_result = _inject_crossings_into_walk_graph(G_walk, injected_payload)
                _eng._SYNTHETIC_CROSSINGS = list(injected_result.get("markers", []))
                syn_list = list(injected_result.get("markers", []))
                print(
                    f"  [Crossings Injection] Injected {len(injected_result.get('edge_pairs', []))} edges, "
                    f"added {injected_result.get('added_edges', 0)} new walk edges"
                )
            except Exception as e:
                if injection_strict:
                    raise
                print(f"  [Crossings Injection] Warning: {e}. Falling back to synthetic generation.")
                injected_payload = None
                injected_result = None
                syn_list = _eng.set_walk_graph(G_walk, synthetic_cfg=synth_cfg, drive_graph=G_con)
        else:
            syn_list = _eng.set_walk_graph(G_walk, synthetic_cfg=synth_cfg, drive_graph=G_con)

        _step_times["build_walk_graph_s"] = round(_wtime.time() - _t0, 2)
    else:
        print("[3b/7] Walking graph disabled (meta.walk_graph.enabled=false)")
        _eng.set_walk_graph(None, synthetic_cfg={"enabled": False})
        _clean_walk_graph = None

    # ── 4. Mode A: Constrained ──
    # Walking BFS uses G_con (safety-restricted edges).
    # Bus driving distances ALWAYS use G_unc (full road network).
    # NOTE: For Mode A, we use the walk graph WITHOUT synthetic crossings.
    # Crossings let pedestrians cross secondary/trunk roads which defeats the
    # safety constraint.
    _saved_walk_graph = _eng._WALK_GRAPH
    if _clean_walk_graph is not None and use_walk_graph:
        _eng._WALK_GRAPH = _clean_walk_graph
        print("  [Mode A] Using walk graph WITHOUT synthetic crossings")
    _ride_caps_on = meta.get("constraints", {}).get("enabled", True)
    _t_a = _wtime.time()
    print("\n" + "-" * 50)
    print("MODE A: Constrained (safety ON, stage walk radii)")
    print("-" * 50)
    data_a = _make_constrained(base_data)
    if not _ride_caps_on:
        _relax_ride_constraints(data_a)
    _reset_caches()
    _prebuild_ball_tree(G_con)
    if force_k:
        data_a["data"]["buses"] = data_a["data"]["buses"][:int(force_k)]
        print(f"  [FleetSearch] force_fleet_size={force_k} — using fixed fleet for Mode A")
        minimize_a = False
    else:
        minimize_a = minimize_buses

    if minimize_a:
        print("  [FleetSearch] minimize_buses=True — searching minimum fleet for Mode A")
        _, sol_a, stats_a, school_a = find_minimum_fleet(
            data_a, G_con, iterations=iters, stage_walk_limits=stage_walk, G_drive=G_unc)
    else:
        sol_a, stats_a, school_a = run_algorithm(
            data_a, G_con, iterations=iters, stage_walk_limits=stage_walk, G_drive=G_unc)
    stats_a["label"] = "Mode-A"
    if "cap_violations_am" not in stats_a:
        cv = _count_cap_violations(sol_a, G_unc, meta.get("constraints", {}))
        stats_a["cap_violations_am"] = cv["am"]
        stats_a["cap_checked_am"] = cv["am_checked"]
        stats_a["cap_violation_pct_am"] = cv["am_pct"]
        stats_a["cap_violations_pm"] = cv["pm"]
        stats_a["cap_checked_pm"] = cv["pm_checked"]
        stats_a["cap_violation_pct_pm"] = cv["pm_pct"]
    if force_k:
        stats_a["buses_used"] = int(force_k)
    if "buses_used" not in stats_a:
        stats_a["buses_used"] = meta.get("buses", {}).get("count")
    # Snapshot candidate data before caches are cleared for next mode
    cands_a    = {sid: list(v) for sid, v in _alns._student_candidate_cache.items()}
    cand_dist_a = {sid: dict(v) for sid, v in _alns._student_candidate_dist.items()}
    _mode_wall_times["A"] = round(_wtime.time() - _t_a, 2)
    print(f"  [A] {stats_a['served']}/{stats_a['total']} served | "
          f"routes={stats_a['routes']} | time={stats_a['total_time']:.1f} min | "
          f"{stats_a['runtime']:.1f}s")

    # ── 5. Mode B: Unconstrained ──
    # Restore walk graph with crossings for Mode B - crossings help in unconstrained mode
    if _saved_walk_graph is not None:
        _eng._WALK_GRAPH = _saved_walk_graph
        print("  [Mode B] Restored walk graph WITH synthetic crossings")
    _t_b = _wtime.time()
    print("\n" + "-" * 50)
    print("MODE B: Unconstrained (all safe, same walk radius)")
    print("-" * 50)
    data_b = _make_unconstrained(base_data)
    if not _ride_caps_on:
        _relax_ride_constraints(data_b)
    # Keep G_unc matrix — all modes share the same driving graph.
    # Clear walk caches since walking graph changes from G_con (A) to G_unc (B).
    _reset_caches(keep_matrix=True)
    import gc as _gc; _gc.collect()   # reclaim freed walk-graph + candidate memory
    _prebuild_ball_tree(G_unc)
    if force_k:
        data_b["data"]["buses"] = data_b["data"]["buses"][:int(force_k)]
        print(f"  [FleetSearch] force_fleet_size={force_k} — using fixed fleet for Mode B")
        minimize_b = False
    else:
        minimize_b = minimize_buses

    if minimize_b:
        print("  [FleetSearch] minimize_buses=True — searching minimum fleet for Mode B")
        _, sol_b, stats_b, school_b = find_minimum_fleet(
            data_b, G_unc, iterations=iters, G_drive=G_unc)
    else:
        sol_b, stats_b, school_b = run_algorithm(data_b, G_unc, iterations=iters, G_drive=G_unc)
    stats_b["label"] = "Mode-B"
    if "cap_violations_am" not in stats_b:
        cv = _count_cap_violations(sol_b, G_unc, meta.get("constraints", {}))
        stats_b["cap_violations_am"] = cv["am"]
        stats_b["cap_checked_am"] = cv["am_checked"]
        stats_b["cap_violation_pct_am"] = cv["am_pct"]
        stats_b["cap_violations_pm"] = cv["pm"]
        stats_b["cap_checked_pm"] = cv["pm_checked"]
        stats_b["cap_violation_pct_pm"] = cv["pm_pct"]
    if force_k:
        stats_b["buses_used"] = int(force_k)
    if "buses_used" not in stats_b:
        stats_b["buses_used"] = meta.get("buses", {}).get("count")
    cands_b    = {sid: list(v) for sid, v in _alns._student_candidate_cache.items()}
    cand_dist_b = {sid: dict(v) for sid, v in _alns._student_candidate_dist.items()}
    _mode_wall_times["B"] = round(_wtime.time() - _t_b, 2)
    print(f"  [B] {stats_b['served']}/{stats_b['total']} served | "
          f"routes={stats_b['routes']} | time={stats_b['total_time']:.1f} min | "
          f"{stats_b['runtime']:.1f}s")

    # ── 6. Mode C: Door-to-Door ──
    _t_c = _wtime.time()
    print("\n" + "-" * 50)
    print("MODE C: Door-to-Door (walk=0, bus visits every home)")
    print("-" * 50)
    data_c = _make_door_to_door(base_data)
    if not _ride_caps_on:
        _relax_ride_constraints(data_c)
    # Keep G_unc matrix AND walk caches — Mode C uses the same G_unc as Mode B.
    # Only ALNS candidate caches are cleared (different walk_radius=0 config).
    _reset_caches(keep_matrix=True, keep_walk=True)
    _prebuild_ball_tree(G_unc)
    if force_k:
        data_c["data"]["buses"] = data_c["data"]["buses"][:int(force_k)]
        print(f"  [FleetSearch] force_fleet_size={force_k} — using fixed fleet for Mode C")
        minimize_c = False
    else:
        minimize_c = minimize_buses

    if minimize_c:
        print("  [FleetSearch] minimize_buses=True — searching minimum fleet for Mode C")
        _, sol_c, stats_c, school_c = find_minimum_fleet(
            data_c, G_unc, iterations=iters, G_drive=G_unc)
    else:
        sol_c, stats_c, school_c = run_algorithm(data_c, G_unc, iterations=iters, G_drive=G_unc)
    stats_c["label"] = "Mode-C"
    if "cap_violations_am" not in stats_c:
        cv = _count_cap_violations(sol_c, G_unc, meta.get("constraints", {}))
        stats_c["cap_violations_am"] = cv["am"]
        stats_c["cap_checked_am"] = cv["am_checked"]
        stats_c["cap_violation_pct_am"] = cv["am_pct"]
        stats_c["cap_violations_pm"] = cv["pm"]
        stats_c["cap_checked_pm"] = cv["pm_checked"]
        stats_c["cap_violation_pct_pm"] = cv["pm_pct"]
    if force_k:
        stats_c["buses_used"] = int(force_k)
    if "buses_used" not in stats_c:
        stats_c["buses_used"] = meta.get("buses", {}).get("count")
    cands_c    = {sid: list(v) for sid, v in _alns._student_candidate_cache.items()}
    cand_dist_c = {sid: dict(v) for sid, v in _alns._student_candidate_dist.items()}
    _mode_wall_times["C"] = round(_wtime.time() - _t_c, 2)
    print(f"  [C] {stats_c['served']}/{stats_c['total']} served | "
          f"routes={stats_c['routes']} | time={stats_c['total_time']:.1f} min | "
          f"{stats_c['runtime']:.1f}s")

    # ── 7. Build map ──
    _t0 = _wtime.time()
    print("\n" + "-" * 50)
    print("BUILDING COMPARISON MAP")
    print("-" * 50)

    center = (school_cfg["latitude"], school_cfg["longitude"])
    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")

    # School marker
    folium.Marker(
        location=center, popup="<b>SCHOOL</b>", tooltip="School",
        icon=folium.Icon(color="darkgreen", icon="graduation-cap", prefix="fa"),
    ).add_to(m)

    # Dangerous roads layer
    fg_danger = FeatureGroup(name="Dangerous Roads (unsafe to cross)", show=True)
    danger_segs = _extract_segments(G_con, center[0], center[1], "dangerous")
    for seg in danger_segs:
        folium.PolyLine(seg, color="#e74c3c", weight=3, opacity=0.45,
                        dash_array="6,4").add_to(fg_danger)
    fg_danger.add_to(m)
    print(f"  Dangerous-road segments: {len(danger_segs)}")

    # Unclassified roads layer
    fg_unclass = FeatureGroup(name="Unclassified Roads (no student placement)", show=False)
    unclass_segs = _extract_segments(G_con, center[0], center[1], "unclassified")
    for seg in unclass_segs:
        folium.PolyLine(seg, color="#7f8c8d", weight=2, opacity=0.5,
                        dash_array="3,5", tooltip="Unclassified road").add_to(fg_unclass)
    fg_unclass.add_to(m)
    print(f"  Unclassified-road segments: {len(unclass_segs)}")

    # Walk network layer
    fg_walknet = FeatureGroup(name="Safe Walking Network", show=False)
    walk_segs = _extract_walk_segments(G_con, center[0], center[1], radius_km=4.0, safe_only=True)
    for seg in walk_segs:
        folium.PolyLine(seg, color="#ffffff", weight=4, opacity=0.45).add_to(fg_walknet)
        folium.PolyLine(seg, color="#00acc1", weight=2.5, opacity=0.9).add_to(fg_walknet)
    fg_walknet.add_to(m)
    print(f"  Safe walk-network segments: {len(walk_segs)}")

    crossings_dict, occupancies_dict, all_stats = {}, {}, {}

    # Clear path cache so rendering computes fresh turn-aware paths on G_unc
    _eng._path_cache.clear()
    _eng._MATRIX_CACHE.clear()
    _eng._MATRIX_CACHE_LENGTH.clear()

    # Bus routes are always rendered with G_unc (the bus drives on all roads)
    fgs = {}          # mk -> (fg_routes, fg_walks)
    fgs_unserved = {}  # mk -> fg_unserved
    for mk, sol, stats in [(mk, sol, st) for mk, sol, st in [
        ("A", sol_a, stats_a),
        ("B", sol_b, stats_b),
        ("C", sol_c, stats_c),
    ] if sol is not None]:
        print(f"  Drawing Mode {mk} …")
        fg_r, fg_w, cx, occ = _add_route_layer(m, G_unc, sol, mk, G_con,
                                                 constraints=meta.get("constraints"))
        fgs[mk] = (fg_r, fg_w)
        crossings_dict[mk] = cx
        occupancies_dict[mk] = occ
        all_stats[mk] = stats
        _sat_by_route = _count_satisfied_per_route(sol, G_unc, meta.get("constraints", {}))
        all_stats[mk]["satisfied"]     = sum(_sat_by_route.values())
        all_stats[mk]["sat_by_route"]  = _sat_by_route
        if "cap_violations_am" not in all_stats[mk]:
            cv = _count_cap_violations(sol, G_unc, meta.get("constraints", {}))
            all_stats[mk]["cap_violations_am"] = cv["am"]
            all_stats[mk]["cap_checked_am"] = cv["am_checked"]
            all_stats[mk]["cap_violation_pct_am"] = cv["am_pct"]
            all_stats[mk]["cap_violations_pm"] = cv["pm"]
            all_stats[mk]["cap_checked_pm"] = cv["pm_checked"]
            all_stats[mk]["cap_violation_pct_pm"] = cv["pm_pct"]
        fgs_unserved[mk] = _add_unserved_layer(m, sol, mk)

    # Candidate stop inspector layers (one per mode, hidden by default)
    cand_data = {mk: cd for mk, cd in {
        "A": (sol_a, cands_a,  cand_dist_a, G_con) if sol_a else None,
        "B": (sol_b, cands_b,  cand_dist_b, G_unc) if sol_b else None,
        "C": (sol_c, cands_c,  cand_dist_c, G_unc) if sol_c else None,
    }.items() if cd is not None}
    fgs_cands = {}
    for mk, (sol, cds, cdst, G_mk) in cand_data.items():
        fgs_cands[mk] = _add_candidate_layer(m, G_mk, mk, sol, cds, cdst)

    _rej_syn_unsafe = _eng.get_synthetic_rejected_unsafe()
    fg_crossings = _add_crossing_markers(m, crossings_dict, rejected_unsafe=_rej_syn_unsafe)
    _syn_markers = _eng.get_synthetic_crossings()
    _show_only_used_syn = bool(synth_cfg.get("show_only_used", True))
    _used_syn_edges = _collect_used_synthetic_edge_keys([sol_a, sol_b, sol_c], G_unc) if _show_only_used_syn else set()
    fg_syn = _add_synthetic_crossing_markers(
        m,
        _syn_markers,
        show_only_used=_show_only_used_syn,
        used_edge_keys=_used_syn_edges,
    )
    fg_injected = _add_injected_crossings_layer(m, injected_payload)
    _syn_label = getattr(fg_syn, "layer_name", "Synthetic Crossings")
    _injected_label = getattr(fg_injected, "layer_name", "Injected Crossings") if fg_injected else "Injected Crossings"
    print(f"  Synthetic crossings (markers): {len(_syn_markers)}")
    if _show_only_used_syn:
        print(f"  Synthetic crossings (used edges): {len(_used_syn_edges)}")
    if injected_result is not None:
        print(f"  Injected crossings (edges): {len(injected_result.get('edge_pairs', []))}")
    print(f"  Rejected synthetic (unsafe-road): {len(_rej_syn_unsafe)}")
    if len(_syn_markers) == 0:
        print("  WARNING: zero synthetic crossings were generated with current thresholds.")

    # Add crossing usage visualization for each mode (which crossings were actually used)
    try:
        G_walk = _eng._WALK_GRAPH or _eng._get_walk_graph(G_unc)
        fgs_crossing_usage = _add_crossing_usage_layers(
            m,
            {"A": sol_a, "B": sol_b, "C": sol_c},
            G_walk,
            G_unc
        )
    except Exception as e:
        print(f"  Warning: Could not add crossing usage visualization: {e}")
        fgs_crossing_usage = {}

    # Print crossing BFS statistics
    crossing_stats = get_crossing_bfs_stats()
    if crossing_stats["students_checked"] > 0:
        print(f"  Crossing BFS stats:")
        print(f"    Students checked: {crossing_stats['students_checked']}")
        print(f"    Candidates enabled by crossings: {crossing_stats['candidates_via_crossing']}")
        print(f"    Students benefiting from crossings: {crossing_stats['students_with_crossing_benefit']}")

    # Fill in empty FeatureGroups for any skipped modes so the layer control doesn't crash
    for _mk in ("A", "B", "C"):
        if _mk not in fgs:
            _emp = FeatureGroup(name=f"Mode {_mk} (skipped)", show=False)
            fgs[_mk] = (_emp, _emp)

    # Custom grouped layer control (title + 3 mode checkboxes, no radio buttons)
    map_var = f"map_{m._id}"
    ctrl_js = _build_custom_layer_control_js(
        map_var, fg_danger, fg_unclass, fg_walknet, fg_syn,
        fgs["A"], fgs["B"], fgs["C"],
        fg_crossings,
        fg_injected=fg_injected,
        syn_label=_syn_label,
        injected_label=_injected_label,
        fg_unserved_a=fgs_unserved.get("A"),
        fg_unserved_b=fgs_unserved.get("B"),
        fg_unserved_c=fgs_unserved.get("C"),
        fg_cands_a=fgs_cands.get("A"),
        fg_cands_b=fgs_cands.get("B"),
        fg_cands_c=fgs_cands.get("C"),
        fg_usage_a=fgs_crossing_usage.get("A"),
        fg_usage_b=fgs_crossing_usage.get("B"),
        fg_usage_c=fgs_crossing_usage.get("C"),
    )
    m.get_root().script.add_child(folium.Element(ctrl_js))

    m.get_root().html.add_child(folium.Element(
        _build_stats_html(all_stats, crossings_dict, occupancies_dict,
                          solutions_dict={"A": sol_a, "B": sol_b, "C": sol_c},
                          G=G_unc,
                          constraints=meta.get("constraints", {}),
                          meta=meta)))

    m.save(output)
    fsize_kb = os.path.getsize(output) / 1024
    _step_times["build_map_s"] = round(_wtime.time() - _t0, 2)
    print(f"\n  Map saved: {output}  ({fsize_kb:.0f} KB)")

    # ── Metrics JSON ──
    _total_wall = round(_wtime.time() - _run_start, 2)
    metrics = _build_metrics(
        meta, stage_walk, all_stats, crossings_dict,
        sol_a, sol_b, sol_c, G_unc, iters,
        total_wall=_total_wall,
        step_times=_step_times,
        mode_wall_times=_mode_wall_times,
    )
    metrics_path = os.path.join(os.path.dirname(output), "output.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(_sanitise_floats(metrics), f, indent=2, ensure_ascii=False)
    print(f"  Metrics  : {metrics_path}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    hdr = f"{'Mode':<30} {'Routes':>6} {'Time':>8} {'Dist':>8} {'Served':>8} {'Crossings':>10} {'Wall(s)':>8}"
    print(hdr)
    print("-" * len(hdr))
    for mk in _active_modes:
        s  = all_stats[mk]
        cx = len(crossings_dict.get(mk, []))
        wt = _mode_wall_times.get(mk, 0)
        print(f"{_MODE_NAMES[mk]:<30} {s['routes']:>6} {s['total_time']:>8.1f} "
              f"{s['total_dist']:>8.1f} {s['served']}/{s['total']:>5} {cx:>10} {wt:>8.1f}")
    if _skipped:
        for mk in _skipped:
            print(f"{_MODE_NAMES[mk]:<30}{'— SKIPPED —':>55}")
    print(f"\nWall-clock per mode:  {', '.join(f'{m}={_mode_wall_times[m]:.1f}s' for m in _active_modes)}")
    print(f"Total wall-clock:     {_total_wall:.1f}s")
    print(f"\nStage distribution used: { {k:v for k,v in meta['stage_distribution'].items() if k != '_comment'} }")
    print(f"Walk limits used: {stage_walk}")
    print(f"\nOpen '{output}' in a browser to explore.")


# ────────────────────────────────────────────────────────────────────
# CLI entry-point
# ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Three-Mode Comparison runner (reads input.json)",
    )
    parser.add_argument(
        "--input", default=None,
        help="Path to input.json (default: input.json next to this script)",
    )
    parser.add_argument(
        "--iterations", type=int, default=None,
        help="Override ALNS iterations from meta.json",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output HTML path (default: from meta.json)",
    )
    args = parser.parse_args()
    run(
        input_path=args.input,
        output_path=args.output,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
