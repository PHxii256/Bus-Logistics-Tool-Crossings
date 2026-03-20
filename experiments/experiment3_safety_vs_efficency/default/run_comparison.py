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

import os, sys, json, time, copy, math, argparse

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
    setup_graph, precompute_matrix, run_algorithm,
    DEFAULT_STAGE_WALK_LIMITS,
)
from data_loader   import load_mode1_input
from solution_state import ServiceSolution
from detour_engine  import (
    calculate_route_path_and_stats,
    walk_path_on_roads,
    walk_distance_on_roads,
)

# Patch: fast snap for large graphs
_eng._FAST_SNAP_MODE = True

# ────────────────────────────────────────────────────────────────────
# Load meta.json
# ────────────────────────────────────────────────────────────────────
_META_PATH = os.path.join(_SCRIPT_DIR, "meta.json")

def _load_meta():
    with open(_META_PATH) as f:
        return json.load(f)


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
def _reset_caches():
    _alns._student_candidate_cache.clear()
    _eng._MATRIX_CACHE.clear()
    _eng._MATRIX_CACHE_LENGTH.clear()
    _eng._path_cache.clear()
    _eng._WALK_DIST_CACHE.clear()
    _eng._STUDENT_NODE_CACHE.clear()
    _eng._WALK_GRAPH = None
    _eng._safe_nodes_cache.clear()


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
    """Mode A: keep safety flags, relax ride-time caps for fairness."""
    d = copy.deepcopy(data)
    return _relax_ride_constraints(d)


def _make_unconstrained(data):
    """Mode B: all-safe walking, relax ride-time caps."""
    d = copy.deepcopy(data)
    for s in d["data"]["students"]:
        s["walk_radius_override"] = 400
    return _relax_ride_constraints(d)


def _make_door_to_door(data):
    """Mode C: no walking, relax ride-time caps."""
    d = copy.deepcopy(data)
    for s in d["data"]["students"]:
        s["walk_radius_override"] = 0
    return _relax_ride_constraints(d)


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


# ────────────────────────────────────────────────────────────────────
# MAP BUILDING  (with PolyLineTextPath arrows, like visualization.py)
# ────────────────────────────────────────────────────────────────────
_ROUTE_COLORS = {
    "A": ["#2196F3", "#1565C0", "#0D47A1", "#82B1FF"],
    "B": ["#4CAF50", "#2E7D32", "#1B5E20", "#A5D6A7"],
    "C": ["#FF9800", "#E65100", "#BF360C", "#FFCC80"],
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
    wcoords = []
    for wi in range(len(wp) - 1):
        u2, v2 = wp[wi], wp[wi + 1]
        ed2 = G.get_edge_data(u2, v2) or G.get_edge_data(v2, u2)
        if ed2:
            dd = ed2[0] if 0 in ed2 else list(ed2.values())[0]
            if "geometry" in dd:
                for lon, lat in dd["geometry"].coords:
                    wcoords.append((lat, lon))
            else:
                wcoords.append((G.nodes[u2]["y"], G.nodes[u2]["x"]))
        else:
            wcoords.append((G.nodes[u2]["y"], G.nodes[u2]["x"]))
    wcoords.append((G.nodes[wp[-1]]["y"], G.nodes[wp[-1]]["x"]))
    return wcoords


def _add_route_layer(m, G, sol, mode_key, G_con):
    """Add route + walk FeatureGroups for one mode.  Returns (fg_routes, fg_walks, crossings, occupancies)."""
    show = mode_key in ("A", "B")   # show constrained + unconstrained by default
    fg_routes = FeatureGroup(name=f"{_MODE_NAMES[mode_key]} – Routes",        show=show)
    fg_walks  = FeatureGroup(name=f"{_MODE_NAMES[mode_key]} – Walking Paths", show=show)
    colors = _ROUTE_COLORS[mode_key]
    active  = [r for r in sol.routes if r.get_student_count() > 0]
    occupancies = []

    for ri, route in enumerate(active):
        c = colors[ri % len(colors)]

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
            folium.CircleMarker(
                location=stop.coords, radius=7,
                color=c, fill=True, fillColor=c, fillOpacity=0.85,
                popup=f"Mode {mode_key} {route.route_id} Stop-{si} ({n_stu} students)",
                tooltip=f"{mode_key}-{route.route_id} Stop {si} ({n_stu} students)",
            ).add_to(fg_routes)

            # ── Walk paths + student homes ──
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
                folium.CircleMarker(
                    location=student.coords, radius=4,
                    color=c, fill=True, fillColor="white", fillOpacity=0.9, weight=2,
                    tooltip=f"{student.id} ({student.school_stage.name}) — walk {walk_m:.0f} m",
                ).add_to(fg_walks)

        occupancies.append(student_count)

    crossings = _count_unsafe_crossings(sol, G_con, G)
    fg_routes.add_to(m)
    fg_walks.add_to(m)
    return fg_routes, fg_walks, crossings, occupancies


def _add_crossing_markers(m, crossings_dict):
    for mk, cxs in crossings_dict.items():
        if not cxs:
            continue
        fg = FeatureGroup(name=f"Unsafe Crossings ({mk})", show=(mk != "A"))
        seen = set()
        for cx in cxs:
            lk = (round(cx["lat"], 6), round(cx["lon"], 6))
            if lk in seen:
                continue
            seen.add(lk)
            folium.CircleMarker(
                location=(cx["lat"], cx["lon"]), radius=6,
                color="red", fill=True, fillColor="yellow", fillOpacity=0.9, weight=2,
                tooltip=f"Unsafe crossing ({mk}) – {cx['student_id']}",
            ).add_to(fg)
        fg.add_to(m)


def _build_stats_html(all_stats, crossings_dict, occupancies_dict):
    blocks = ""
    for mk in ("A", "B", "C"):
        s = all_stats[mk]
        cx = len(crossings_dict.get(mk, []))
        occ = occupancies_dict.get(mk, [])
        avg_occ = (sum(occ) / len(occ)) if occ else 0
        cx_color = "#c0392b" if cx > 0 else "#27ae60"
        mc = _ROUTE_COLORS[mk][0]
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
              <td style="text-align:left; padding:1px 4px;">Total Time</td>
              <td style="text-align:left; padding:1px 4px;">Distance</td>
              <td style="text-align:left; padding:1px 4px;">Avg Occ.</td>
              <td style="text-align:left; padding:1px 4px;">Served</td>
              <td style="text-align:left; padding:1px 4px;">Crossings</td>
            </tr>
            <tr style="font-weight:bold;">
              <td style="padding:1px 4px;">{s['routes']}</td>
              <td style="padding:1px 4px;">{s['total_time']:.0f} min</td>
              <td style="padding:1px 4px;">{s['total_dist']:.1f} km</td>
              <td style="padding:1px 4px;">{avg_occ:.1f}</td>
              <td style="padding:1px 4px;">{s['served']}/{s['total']}</td>
              <td style="padding:1px 4px; color:{cx_color};">{cx}</td>
            </tr>
          </table>
        </div>"""

    return f"""
    <div style="position:fixed; bottom:15px; right:15px; width:380px;
                max-height:calc(100vh - 80px); overflow-y:auto;
                background:white; border:2px solid #555; z-index:9999;
                padding:12px 14px; border-radius:6px; font-size:12px;
                font-family:Arial,sans-serif; box-shadow:2px 2px 8px rgba(0,0,0,.25);">
      <div style="font-weight:bold; font-size:13px; margin-bottom:10px;
                  padding-bottom:6px; border-bottom:2px solid #ccc;">
        Three-Mode Routing Comparison
      </div>
      {blocks}
      <div style="font-size:10px; color:#888; margin-top:4px;">
        Toggle layers via top-right control.<br>
        <span style="color:#e74c3c;">&#x2015;&#x2015;</span> Dangerous roads
        &nbsp;&nbsp;
        <span style="color:#7f8c8d;">&#x2508;&#x2508;</span> Unclassified roads
      </div>
    </div>
    """


# ────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Three-Mode Comparison (from meta.json)")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Override ALNS iterations (default: from meta.json)")
    parser.add_argument("--output", default=None,
                        help="Output HTML path (default: from meta.json)")
    args = parser.parse_args()

    meta = _load_meta()
    iters  = args.iterations or meta.get("algorithm", {}).get("iterations", 30)
    output = args.output     or meta.get("output", "comparison_map.html")

    # Resolve output relative to this script's directory
    if not os.path.isabs(output):
        output = os.path.join(_SCRIPT_DIR, output)

    school_cfg = meta["school"]
    raw_walk = meta.get("stage_walk_limits", DEFAULT_STAGE_WALK_LIMITS)
    # Filter out _comment and other non-stage keys
    stage_walk = {k: v for k, v in raw_walk.items()
                  if k in ("KG", "ELEMENTARY", "MIDDLE", "HIGH")}

    print("=" * 60)
    print("  THREE-MODE ROUTING COMPARISON  (meta.json)")
    print("=" * 60)
    print(f"  Students : {meta['n_students']}")
    print(f"  Seed     : {meta['seed']}")
    print(f"  Stages   : {meta['stage_distribution']}")
    print(f"  Walk lim : {stage_walk}")
    print(f"  Iters    : {iters}")
    print()

    # ── 1. Generate dataset ──
    print("[1/7] Generating dataset …")
    base_data = _generate_dataset(meta)
    base_data["meta"]["algorithm"]["iterations"] = iters

    # Print stage breakdown
    stage_counts = {}
    for s in base_data["data"]["students"]:
        stage_counts[s["school_stage"]] = stage_counts.get(s["school_stage"], 0) + 1
    print(f"  Stage breakdown: {stage_counts}")

    # ── 2. Constrained graph ──
    print("\n[2/7] Building CONSTRAINED graph …")
    G_con = setup_graph(base_data["meta"], unconstrained=False)
    _prebuild_ball_tree(G_con)

    import copy as _copy
    G_con_saved = _copy.deepcopy(G_con)

    # ── 3. Unconstrained graph ──
    print("[3/7] Building UNCONSTRAINED graph …")
    G_unc = setup_graph(base_data["meta"], unconstrained=True)
    G_con = G_con_saved
    _eng._BALL_TREE = None
    _eng._BALL_TREE_GRAPH_ID = None
    _eng._BALL_TREE_NODE_IDS = None

    # ── 4. Mode A: Constrained ──
    # Walking BFS uses G_con (safety-restricted edges).
    # Bus driving distances ALWAYS use G_unc (full road network).
    # Ride-time caps are relaxed so only the walking safety variable differs.
    print("\n" + "-" * 50)
    print("MODE A: Constrained (safety ON, stage walk radii)")
    print("-" * 50)
    data_a = _make_constrained(base_data)
    _reset_caches()
    _prebuild_ball_tree(G_con)
    sol_a, stats_a, school_a = run_algorithm(
        data_a, G_con, iterations=iters, stage_walk_limits=stage_walk,
        G_drive=G_unc)
    stats_a["label"] = "Mode-A"
    print(f"  [A] {stats_a['served']}/{stats_a['total']} served | "
          f"routes={stats_a['routes']} | time={stats_a['total_time']:.1f} min | "
          f"{stats_a['runtime']:.1f}s")

    # ── 5. Mode B: Unconstrained ──
    print("\n" + "-" * 50)
    print("MODE B: Unconstrained (all safe, same walk radius)")
    print("-" * 50)
    data_b = _make_unconstrained(base_data)
    _reset_caches()
    _prebuild_ball_tree(G_unc)
    sol_b, stats_b, school_b = run_algorithm(data_b, G_unc, iterations=iters,
                                              G_drive=G_unc)
    stats_b["label"] = "Mode-B"
    print(f"  [B] {stats_b['served']}/{stats_b['total']} served | "
          f"routes={stats_b['routes']} | time={stats_b['total_time']:.1f} min | "
          f"{stats_b['runtime']:.1f}s")

    # ── 6. Mode C: Door-to-Door ──
    print("\n" + "-" * 50)
    print("MODE C: Door-to-Door (walk=0, bus visits every home)")
    print("-" * 50)
    data_c = _make_door_to_door(base_data)
    _reset_caches()
    _prebuild_ball_tree(G_unc)
    sol_c, stats_c, school_c = run_algorithm(data_c, G_unc, iterations=iters,
                                              G_drive=G_unc)
    stats_c["label"] = "Mode-C"
    print(f"  [C] {stats_c['served']}/{stats_c['total']} served | "
          f"routes={stats_c['routes']} | time={stats_c['total_time']:.1f} min | "
          f"{stats_c['runtime']:.1f}s")

    # ── 7. Build map ──
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
    fg_danger = FeatureGroup(name="\u26A0 Dangerous Roads (unsafe to cross)", show=True)
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

    crossings_dict, occupancies_dict, all_stats = {}, {}, {}

    # Clear path cache so rendering computes fresh turn-aware paths on G_unc
    _eng._path_cache.clear()
    _eng._MATRIX_CACHE.clear()
    _eng._MATRIX_CACHE_LENGTH.clear()

    # Bus routes are always rendered with G_unc (the bus drives on all roads)
    for mk, sol, stats in [
        ("A", sol_a, stats_a),
        ("B", sol_b, stats_b),
        ("C", sol_c, stats_c),
    ]:
        print(f"  Drawing Mode {mk} …")
        _, _, cx, occ = _add_route_layer(m, G_unc, sol, mk, G_con)
        crossings_dict[mk] = cx
        occupancies_dict[mk] = occ
        all_stats[mk] = stats

    _add_crossing_markers(m, crossings_dict)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(
        _build_stats_html(all_stats, crossings_dict, occupancies_dict)))

    m.save(output)
    fsize_kb = os.path.getsize(output) / 1024
    print(f"\n  Map saved: {output}  ({fsize_kb:.0f} KB)")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    hdr = f"{'Mode':<30} {'Routes':>6} {'Time':>8} {'Dist':>8} {'Served':>8} {'Crossings':>10}"
    print(hdr)
    print("-" * len(hdr))
    for mk in ("A", "B", "C"):
        s = all_stats[mk]
        cx = len(crossings_dict.get(mk, []))
        print(f"{_MODE_NAMES[mk]:<30} {s['routes']:>6} {s['total_time']:>8.1f} "
              f"{s['total_dist']:>8.1f} {s['served']}/{s['total']:>5} {cx:>10}")
    print(f"\nStage distribution used: { {k:v for k,v in meta['stage_distribution'].items() if k != '_comment'} }")
    print(f"Walk limits used: {stage_walk}")
    print(f"\nOpen '{output}' in a browser to explore.")


if __name__ == "__main__":
    main()
