"""
visualize_comparison.py — Three-Mode Routing Comparison Visualization

Generates a Folium HTML map with toggleable layers showing:
  Mode A (Constrained):       Safety flags ON, stage-based walk radii
  Mode B (Unconstrained):     All edges safe, walk_radius=400 for all, ride caps disabled
  Mode C (Door-to-Door):      All edges safe, walk_radius=0 for all (bus visits every home)

Overlays:
  - Dangerous roads layer (roads marked unsafe in constrained graph)
  - Unsafe-crossing markers (walks that cross unsafe edges, detected for all modes)
  - Walking paths (home → stop) per mode
  - Stats panel comparing routes, time, occupancy, crossings

Usage:
    python visualize_comparison.py [--seed 42] [--students 40] [--iterations 20] [--output comparison_map.html]
"""

import os, sys, json, time, copy, argparse, math
import folium
from folium import plugins, FeatureGroup

# ── path fix ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import detour_engine as _eng
import alns_engine   as _alns

from run_algorithm  import setup_graph, precompute_matrix
from data_loader    import load_json, load_mode1_input, serialize_routes
from solution_state import ServiceSolution
from alns_engine    import ALNSEngine
from detour_engine  import (
    calculate_route_time_from_matrix,
    calculate_route_distance_from_matrix,
    calculate_route_path_and_stats,
    walk_path_on_roads,
    walk_distance_on_roads,
    calculate_walk_penalty,
    haversine_walk_distance,
    find_shortest_path_with_turns,
)
from entities import Student

# Enable fast snap for large graphs
_eng._FAST_SNAP_MODE = True


# ============================================================================
# DATASET GENERATION  (inline, mirrors experiments/generate_dataset.py)
# ============================================================================

import random
import numpy as np

DEFAULT_SCHOOL = {
    "name": "Victory College School",
    "latitude": 29.964406,
    "longitude": 31.270319,
}

def _gaussian_annulus(center_lat, center_lon, peak_km=2.0, sigma_km=1.0,
                      min_km=0.4, max_km=5.0):
    while True:
        dist = random.gauss(peak_km, sigma_km)
        if min_km <= dist <= max_km:
            break
    angle = random.uniform(0, 2 * math.pi)
    d_lat = (dist * math.sin(angle)) / 111.0
    d_lon = (dist * math.cos(angle)) / (111.0 * math.cos(math.radians(center_lat)))
    return center_lat + d_lat, center_lon + d_lon


def generate_input_json(n_students: int, seed: int) -> dict:
    """Build a synthetic input dict (same format as experiment data files)."""
    random.seed(seed)
    np.random.seed(seed)
    students = []
    for i in range(n_students):
        lat, lon = _gaussian_annulus(DEFAULT_SCHOOL["latitude"],
                                     DEFAULT_SCHOOL["longitude"])
        age = random.randint(5, 17)
        if   age <= 6:  stage = "KG"
        elif age <= 11: stage = "ELEMENTARY"
        elif age <= 14: stage = "MIDDLE"
        else:           stage = "HIGH"
        students.append({
            "id": f"S{i+1:03d}", "latitude": lat, "longitude": lon,
            "age": age, "school_stage": stage, "fee": 100.0,
        })
    return {
        "meta": {
            "mode": "generate_routes", "city": "Cairo",
            "description": f"Comparison dataset - {n_students} students, seed {seed}",
            "constraints": {
                "ride_time_multiplier": 2.5, "floor_minutes": 45,
                "ceiling_minutes": 60, "daily_detour_budget_minutes": 5,
            },
            "algorithm": {"method": "alns", "iterations": 200},
        },
        "data": {
            "school": DEFAULT_SCHOOL,
            "buses": [
                {"id": f"BUS_{i+1}", "type": "Standard",
                 "capacity": 60, "fixed_cost": 50, "var_cost_km": 1.0}
                for i in range(4)
            ],
            "students": students,
        },
    }


# ============================================================================
# RUN HELPERS
# ============================================================================

def _reset_caches(full=True):
    """Clear module-level caches between runs."""
    _alns._student_candidate_cache.clear()
    if full:
        _eng._MATRIX_CACHE.clear()
        _eng._MATRIX_CACHE_LENGTH.clear()
        _eng._path_cache.clear()
        _eng._WALK_DIST_CACHE.clear()
        _eng._STUDENT_NODE_CACHE.clear()
        _eng._WALK_GRAPH = None
        _eng._safe_nodes_cache.clear()


def _prebuild_ball_tree(G):
    print("  Pre-building BallTree spatial index ...", end="", flush=True)
    t0 = time.time()
    _eng._get_or_build_ball_tree(G)
    print(f" done in {time.time()-t0:.1f}s")


def _run_mode(data: dict, G, label: str, iterations: int):
    """Run ALNS on *data* using graph *G*.  Returns (best_sol, stats_dict)."""
    _reset_caches(full=True)
    students, buses, routes, school_coords, constraints, algo_cfg = load_mode1_input(data, G)
    iters = iterations or algo_cfg.get("iterations", 60)
    precompute_matrix(students, routes, G)

    initial = ServiceSolution(students, routes, G)
    engine  = ALNSEngine(initial, iterations=iters)
    t0      = time.time()
    best    = engine.run()
    elapsed = time.time() - t0

    for r in best.routes:
        t = calculate_route_time_from_matrix(r.stops, G)
        r.total_time = t if t is not None else 0.0
        d = calculate_route_distance_from_matrix(r.stops, G)
        r.total_distance = d if d is not None else 0.0

    served = sum(1 for s in best.students if s.is_served)
    total  = len(best.students)
    active = [r for r in best.routes if r.get_student_count() > 0]
    total_time = sum(r.total_time for r in active)
    total_dist = sum(r.total_distance for r in active)

    print(f"  [{label}] {served}/{total} served | routes={len(active)} | "
          f"time={total_time:.1f}min | dist={total_dist:.1f}km | {elapsed:.1f}s")

    stats = {
        "label": label, "served": served, "total": total,
        "routes": len(active), "total_time": round(total_time, 2),
        "total_dist": round(total_dist, 2),
        "objective": round(best.calculate_objective(), 2),
        "runtime": round(elapsed, 2),
    }
    return best, stats, school_coords


# ============================================================================
# INPUT MUTATION
# ============================================================================

def _make_unconstrained(data: dict) -> dict:
    """Mode B: walk_radius=400 for everyone, ride caps disabled."""
    d = copy.deepcopy(data)
    for s in d["data"]["students"]:
        s["walk_radius_override"] = 400
    d["meta"].setdefault("constraints", {}).update({
        "ride_time_multiplier": 999,
        "floor_minutes": 999,
        "ceiling_minutes": 999,
    })
    return d


def _make_door_to_door(data: dict) -> dict:
    """Mode C: walk_radius=0 for everyone (bus visits every home), ride caps disabled."""
    d = copy.deepcopy(data)
    for s in d["data"]["students"]:
        s["walk_radius_override"] = 0
    d["meta"].setdefault("constraints", {}).update({
        "ride_time_multiplier": 999,
        "floor_minutes": 999,
        "ceiling_minutes": 999,
    })
    return d


# ============================================================================
# CROSSING DETECTION
# ============================================================================

# Maximum total unsafe-edge run length (metres) that qualifies as a TRUE crossing.
# A real road crossing spans ~10–80 m of carriageway.
# Anything >= this threshold is the student walking ALONGSIDE the road, not crossing it.
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


def _classify_walk_path(walk_path, G_constrained):
    """Detect true dangerous road crossings along a walk path.

    An edge is 'dangerous' only if its highway type is in _DANGEROUS_HW_TYPES
    (primary, trunk, secondary, motorway).  Tertiary and residential are safe.

    The run-length heuristic:
      • A short run of dangerous edges (< 150 m) bounded by non-dangerous
        edges on both sides = TRUE crossing (perpendicular).
      • A longer run or one that starts/ends the path = walking alongside.
    """
    if len(walk_path) < 2:
        return []

    # Build per-edge metadata: (is_dangerous, length_m, u, v, highway)
    edges = []
    for i in range(len(walk_path) - 1):
        u, v = walk_path[i], walk_path[i + 1]
        ed = G_constrained.get_edge_data(u, v) or G_constrained.get_edge_data(v, u)
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

        # ── Start of a dangerous run ──
        run_start   = i
        run_total_m = 0.0
        run_hws     = set()
        while i < len(edges) and edges[i][0]:
            run_total_m += edges[i][1]
            run_hws.add(edges[i][4])
            i += 1
        run_end = i   # exclusive

        # Find the middle edge of this run for the marker position
        accumulated = 0.0
        mid_u, mid_v = edges[run_start][2], edges[run_start][3]
        for j in range(run_start, run_end):
            accumulated += edges[j][1]
            if accumulated >= run_total_m / 2:
                mid_u, mid_v = edges[j][2], edges[j][3]
                break

        came_from_safe = (run_start == 0) or not edges[run_start - 1][0]
        goes_to_safe   = (run_end >= len(edges)) or not edges[run_end][0]

        # Only a TRUE crossing if short AND bounded by safe territory on both sides
        if run_total_m < _CROSSING_RUN_MAX_M and came_from_safe and goes_to_safe:
            mid_lat = (G_constrained.nodes[mid_u]["y"] + G_constrained.nodes[mid_v]["y"]) / 2
            mid_lon = (G_constrained.nodes[mid_u]["x"] + G_constrained.nodes[mid_v]["x"]) / 2
            crossings.append({
                "u": mid_u, "v": mid_v,
                "lat": mid_lat, "lon": mid_lon,
                "run_length_m": round(run_total_m, 1),
                "road_types": ', '.join(sorted(run_hws)),
            })

    return crossings


def _count_unsafe_crossings(sol, G_constrained, G_solving):
    """Detect dangerous crossings for every served student.

    Walks on G_solving (the graph used for that mode's routing), but checks
    edge danger using G_constrained (the constrained ground-truth graph).
    """
    crossings = []
    for route in sol.routes:
        for stop in route.stops:
            if stop.stop_type == "school":
                continue
            for student in stop.students:
                s_node = _eng.fast_nearest_node(G_solving, student.coords[1], student.coords[0])
                walk_path = walk_path_on_roads(G_solving, s_node, stop.node_id)
                for cx in _classify_walk_path(walk_path, G_constrained):
                    cx["student_id"] = student.id
                    crossings.append(cx)
    return crossings


# ============================================================================
# EXTRACT DANGEROUS ROAD SEGMENTS
# ============================================================================

def _extract_dangerous_roads(G_constrained, center_lat=None, center_lon=None, radius_km=4.0):
    """Return list of (coords_list) for edges where is_safe_to_cross=False.
    
    Only includes edges within *radius_km* of center (school) and longer than 30m
    to keep the HTML file manageable.
    """
    segments = []
    seen = set()
    has_center = center_lat is not None and center_lon is not None
    for u, v, k, data in G_constrained.edges(keys=True, data=True):
        if data.get("is_safe_to_cross", True):
            continue
        if data.get("length", 0) < 30:
            continue
        edge_key = (min(u, v), max(u, v))
        if edge_key in seen:
            continue
        seen.add(edge_key)
        mid_lat = (G_constrained.nodes[u]["y"] + G_constrained.nodes[v]["y"]) / 2
        mid_lon = (G_constrained.nodes[u]["x"] + G_constrained.nodes[v]["x"]) / 2
        if has_center:
            # Quick haversine approximation in km
            dlat = abs(mid_lat - center_lat) * 111.0
            dlon = abs(mid_lon - center_lon) * 111.0 * math.cos(math.radians(center_lat))
            if math.sqrt(dlat**2 + dlon**2) > radius_km:
                continue
        if "geometry" in data:
            coords = [(lat, lon) for lon, lat in data["geometry"].coords]
        else:
            coords = [
                (G_constrained.nodes[u]["y"], G_constrained.nodes[u]["x"]),
                (G_constrained.nodes[v]["y"], G_constrained.nodes[v]["x"]),
            ]
        segments.append(coords)
    return segments


# ============================================================================
# MAP BUILDING
# ============================================================================

_ROUTE_COLORS = {
    "A": ["#2196F3", "#1565C0", "#0D47A1", "#82B1FF"],    # blues
    "B": ["#4CAF50", "#2E7D32", "#1B5E20", "#A5D6A7"],    # greens
    "C": ["#FF9800", "#E65100", "#BF360C", "#FFCC80"],     # oranges
}

_MODE_NAMES = {
    "A": "Constrained (Safe Walking)",
    "B": "Unconstrained (Any Walking)",
    "C": "Door-to-Door (No Walking)",
}


def _build_walk_coords(G, wp):
    """Convert a walk-path node list to a list of (lat, lon) tuples following road geometry."""
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


def _add_route_layer(m, G, sol, mode_key, school_coords, G_constrained):
    """Add two FeatureGroup layers for one mode: bus routes and walking paths (separate).

    Returns (fg_routes, fg_walks, crossings_list, occupancy_list).
    """
    show = (mode_key == "A")
    fg_routes = FeatureGroup(name=f"{_MODE_NAMES[mode_key]} – Routes",        show=show)
    fg_walks  = FeatureGroup(name=f"{_MODE_NAMES[mode_key]} – Walking Paths", show=show)
    colors = _ROUTE_COLORS[mode_key]
    active_routes = [r for r in sol.routes if r.get_student_count() > 0]
    occupancies = []

    for ri, route in enumerate(active_routes):
        c = colors[ri % len(colors)]

        # 1. Route polyline following road geometry
        if len(route.stops) > 1:
            full_path, _ = calculate_route_path_and_stats(G, route.stops, weight="travel_time")
            if full_path:
                offset = 0.00003 * (ri - 0.5)
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

                pl = folium.PolyLine(coords, color=c, weight=5, opacity=0.8,
                                     popup=f"Mode {mode_key} Route {route.route_id} "
                                           f"({route.get_student_count()} students, "
                                           f"{route.total_time:.0f} min, {route.total_distance:.1f} km)")
                pl.add_to(fg_routes)
                plugins.PolyLineTextPath(
                    pl, "          \u27A4          ", repeat=True, offset=6,
                    attributes={"fill": c, "font-weight": "bold", "font-size": "24"},
                ).add_to(fg_routes)

        # 2. Stop markers (bus routes layer)
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

            # 3. Walking paths + student homes (walking paths layer)
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

                # Student home marker — goes on the walking paths layer
                walk_m = walk_distance_on_roads(G, s_node, stop.node_id)
                folium.CircleMarker(
                    location=student.coords, radius=4,
                    color=c, fill=True, fillColor="white", fillOpacity=0.9, weight=2,
                    tooltip=f"{student.id} ({student.school_stage.name}) — walk {walk_m:.0f} m",
                ).add_to(fg_walks)

        occupancies.append(student_count)

    # Crossings (detected against constrained graph ground truth)
    crossings = _count_unsafe_crossings(sol, G_constrained, G)

    fg_routes.add_to(m)
    fg_walks.add_to(m)
    return fg_routes, fg_walks, crossings, occupancies


def _add_dangerous_roads_layer(m, G_constrained):
    """Overlay red dashed lines for all unsafe-to-cross road segments near the school."""
    fg = FeatureGroup(name="⚠ Dangerous Roads (unsafe to cross)", show=True)
    segments = _extract_dangerous_roads(
        G_constrained,
        center_lat=DEFAULT_SCHOOL["latitude"],
        center_lon=DEFAULT_SCHOOL["longitude"],
        radius_km=5.0,
    )
    for seg in segments:
        folium.PolyLine(seg, color="#e74c3c", weight=3, opacity=0.45,
                        dash_array="6,4").add_to(fg)
    fg.add_to(m)
    print(f"  Dangerous-road segments: {len(segments)}")
    return fg


def _add_unclassified_roads_layer(m, G_constrained):
    """Overlay grey dashed lines for unclassified roads near the school.
    These are OSM roads of ambiguous/unknown type — students are not placed on them.
    """
    fg = FeatureGroup(name="Unclassified Roads (no student placement)", show=False)
    seen = set()
    center_lat = DEFAULT_SCHOOL["latitude"]
    center_lon = DEFAULT_SCHOOL["longitude"]
    count = 0
    for u, v, k, data in G_constrained.edges(keys=True, data=True):
        hw = data.get("highway", "")
        if isinstance(hw, list):
            hw = hw[0]
        if hw != "unclassified":
            continue
        if data.get("length", 0) < 20:
            continue
        edge_key = (min(u, v), max(u, v))
        if edge_key in seen:
            continue
        seen.add(edge_key)
        mid_lat = (G_constrained.nodes[u]["y"] + G_constrained.nodes[v]["y"]) / 2
        mid_lon = (G_constrained.nodes[u]["x"] + G_constrained.nodes[v]["x"]) / 2
        dlat = abs(mid_lat - center_lat) * 111.0
        dlon = abs(mid_lon - center_lon) * 111.0 * math.cos(math.radians(center_lat))
        if math.sqrt(dlat**2 + dlon**2) > 5.0:
            continue
        if "geometry" in data:
            coords = [(lat, lon) for lon, lat in data["geometry"].coords]
        else:
            coords = [
                (G_constrained.nodes[u]["y"], G_constrained.nodes[u]["x"]),
                (G_constrained.nodes[v]["y"], G_constrained.nodes[v]["x"]),
            ]
        folium.PolyLine(coords, color="#7f8c8d", weight=2, opacity=0.5,
                        dash_array="3,5",
                        tooltip="Unclassified road").add_to(fg)
        count += 1
    fg.add_to(m)
    print(f"  Unclassified-road segments: {count}")
    return fg


def _add_crossing_markers(m, crossings_dict):
    """Add FeatureGroup with X markers for each mode's unsafe crossings."""
    for mode_key, crossings in crossings_dict.items():
        if not crossings:
            continue
        fg = FeatureGroup(
            name=f"Unsafe Crossings ({mode_key})",
            show=(mode_key != "A"),  # hide constrained by default (should be 0)
        )
        seen = set()
        for cx in crossings:
            loc_key = (round(cx["lat"], 6), round(cx["lon"], 6))
            if loc_key in seen:
                continue
            seen.add(loc_key)
            folium.CircleMarker(
                location=(cx["lat"], cx["lon"]),
                radius=6, color="red", fill=True, fillColor="yellow",
                fillOpacity=0.9, weight=2,
                tooltip=f"Unsafe crossing ({mode_key}) – {cx['student_id']}",
            ).add_to(fg)
        fg.add_to(m)


def _build_stats_html(all_stats, crossings_dict, occupancies_dict):
    """Build a compact stats panel fixed to the bottom-right of the map."""
    blocks = ""
    for mode_key in ("A", "B", "C"):
        s   = all_stats[mode_key]
        cx  = len(crossings_dict.get(mode_key, []))
        occ = occupancies_dict.get(mode_key, [])
        avg_occ   = (sum(occ) / len(occ)) if occ else 0
        cx_color  = "#c0392b" if cx > 0 else "#27ae60"
        mode_color = _ROUTE_COLORS[mode_key][0]
        blocks += f"""
        <div style="margin-bottom:8px; padding-bottom:8px;
                    border-bottom:1px solid #e0e0e0;">
          <div style="font-weight:bold; color:{mode_color}; margin-bottom:3px;">
            {mode_key}: {_MODE_NAMES[mode_key]}
          </div>
          <table style="width:100%; border-collapse:collapse;
                        font-size:11px; text-align:center;">
            <tr style="color:#555;">
              <td style="text-align:left; padding:1px 4px;">Routes</td>
              <td style="text-align:left; padding:1px 4px;">Total Time</td>
              <td style="text-align:left; padding:1px 4px;">Distance</td>
              <td style="text-align:left; padding:1px 4px;">Avg Occupancy</td>
              <td style="text-align:left; padding:1px 4px;">Served</td>
              <td style="text-align:left; padding:1px 4px;">Crossings</td>
            </tr>
            <tr style="font-weight:bold;">
              <td style="padding:1px 4px;">{s['routes']}</td>
              <td style="padding:1px 4px;">{s['total_time']:.0f} min</td>
              <td style="padding:1px 4px;">{s['total_dist']:.1f} km</td>
              <td style="padding:1px 4px;">{avg_occ:.1f} students</td>
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
        Toggle layers via the control (top-right).<br>
        <span style="color:#e74c3c;">&#x2015;&#x2015;</span> Dangerous roads (unsafe to cross)&nbsp;&nbsp;
        <span style="color:#7f8c8d;">&#x2508;&#x2508;</span> Unclassified roads (no student placement)
      </div>
    </div>
    """


def build_comparison_map(
    sol_a, stats_a, school_a,
    sol_b, stats_b, school_b,
    sol_c, stats_c, school_c,
    G_constrained, G_unconstrained,
    output_file="comparison_map.html",
):
    """Assemble the final Folium map with all three modes overlaid."""
    # Center on school
    center = (DEFAULT_SCHOOL["latitude"], DEFAULT_SCHOOL["longitude"])
    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")

    # School marker
    folium.Marker(
        location=center,
        popup="<b>SCHOOL</b>",
        tooltip="School",
        icon=folium.Icon(color="darkgreen", icon="graduation-cap", prefix="fa"),
    ).add_to(m)

    # Road overlays
    _add_dangerous_roads_layer(m, G_constrained)
    _add_unclassified_roads_layer(m, G_constrained)

    crossings_dict = {}
    occupancies_dict = {}
    all_stats = {}

    # Mode A: constrained routes on constrained graph
    print("  Drawing Mode A (Constrained)...")
    _, _, cx_a, occ_a = _add_route_layer(m, G_constrained, sol_a, "A", school_a, G_constrained)
    crossings_dict["A"] = cx_a
    occupancies_dict["A"] = occ_a
    all_stats["A"] = stats_a

    # Mode B: unconstrained routes on unconstrained graph
    print("  Drawing Mode B (Unconstrained)...")
    _, _, cx_b, occ_b = _add_route_layer(m, G_unconstrained, sol_b, "B", school_b, G_constrained)
    crossings_dict["B"] = cx_b
    occupancies_dict["B"] = occ_b
    all_stats["B"] = stats_b

    # Mode C: door-to-door routes on unconstrained graph
    print("  Drawing Mode C (Door-to-Door)...")
    _, _, cx_c, occ_c = _add_route_layer(m, G_unconstrained, sol_c, "C", school_c, G_constrained)
    crossings_dict["C"] = cx_c
    occupancies_dict["C"] = occ_c
    all_stats["C"] = stats_c

    # Crossing markers
    _add_crossing_markers(m, crossings_dict)

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Stats panel
    stats_html = _build_stats_html(all_stats, crossings_dict, occupancies_dict)
    m.get_root().html.add_child(folium.Element(stats_html))

    m.save(output_file)
    print(f"\n  Map saved: {output_file}")
    return crossings_dict, occupancies_dict


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Three-Mode Routing Comparison Visualization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--students", type=int, default=40, help="Number of students")
    parser.add_argument("--iterations", type=int, default=20,
                        help="ALNS iterations per mode (lower = faster)")
    parser.add_argument("--output", default="comparison_map.html",
                        help="Output HTML file")
    parser.add_argument("--input", default=None,
                        help="Use an existing input JSON instead of generating one")
    args = parser.parse_args()

    print("=" * 60)
    print("  THREE-MODE ROUTING COMPARISON")
    print("=" * 60)

    # ── 1. Dataset ─────────────────────────────────────────────
    if args.input and os.path.exists(args.input):
        print(f"\nLoading input from: {args.input}")
        base_data = load_json(args.input)
    else:
        print(f"\nGenerating dataset: {args.students} students, seed={args.seed}")
        base_data = generate_input_json(args.students, args.seed)

    # Override iterations
    base_data["meta"]["algorithm"]["iterations"] = args.iterations

    # ── 2. Build Constrained Graph ─────────────────────────────
    print("\n[Graph] Building CONSTRAINED graph...")
    G_con = setup_graph(base_data["meta"], unconstrained=False)
    _prebuild_ball_tree(G_con)

    # ── 3. Build Unconstrained Graph ───────────────────────────
    # setup_graph mutates the same cached graph object, so we need a copy first
    import copy as _copy
    G_con_saved = _copy.deepcopy(G_con)

    print("[Graph] Building UNCONSTRAINED graph...")
    G_unc = setup_graph(base_data["meta"], unconstrained=True)
    # Restore the constrained copy (setup_graph overwrote the cached graph)
    G_con = G_con_saved

    # Rebuild BallTree for the unconstrained graph (it changed)
    _eng._BALL_TREE = None
    _eng._BALL_TREE_GRAPH_ID = None
    _eng._BALL_TREE_NODE_IDS = None

    # ── 4. Run Mode A: Constrained ─────────────────────────────
    print("\n" + "-" * 50)
    print("MODE A: Constrained (safety ON, stage walk radii)")
    print("-" * 50)
    _prebuild_ball_tree(G_con)
    sol_a, stats_a, school_a = _run_mode(base_data, G_con, "Mode-A", args.iterations)

    # ── 5. Run Mode B: Unconstrained ──────────────────────────
    print("\n" + "-" * 50)
    print("MODE B: Unconstrained (all safe, walk=400m)")
    print("-" * 50)
    data_b = _make_unconstrained(base_data)
    _prebuild_ball_tree(G_unc)
    sol_b, stats_b, school_b = _run_mode(data_b, G_unc, "Mode-B", args.iterations)

    # ── 6. Run Mode C: Door-to-Door ──────────────────────────
    print("\n" + "-" * 50)
    print("MODE C: Door-to-Door (walk=0, bus visits every home)")
    print("-" * 50)
    data_c = _make_door_to_door(base_data)
    sol_c, stats_c, school_c = _run_mode(data_c, G_unc, "Mode-C", args.iterations)

    # ── 7. Build Comparison Map ────────────────────────────────
    print("\n" + "-" * 50)
    print("BUILDING COMPARISON MAP")
    print("-" * 50)
    crossings_dict, occupancies_dict = build_comparison_map(
        sol_a, stats_a, school_a,
        sol_b, stats_b, school_b,
        sol_c, stats_c, school_c,
        G_con, G_unc,
        output_file=args.output,
    )

    # ── 8. Summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    header = f"{'Mode':<30} {'Routes':>6} {'Time':>8} {'Dist':>8} {'Served':>8} {'Crossings':>10}"
    print(header)
    print("-" * len(header))
    for key in ("A", "B", "C"):
        s = {"A": stats_a, "B": stats_b, "C": stats_c}[key]
        cx = len(crossings_dict.get(key, []))
        print(f"{_MODE_NAMES[key]:<30} {s['routes']:>6} {s['total_time']:>8.1f} "
              f"{s['total_dist']:>8.1f} {s['served']}/{s['total']:>5} {cx:>10}")
    print(f"\nOpen '{args.output}' in a browser to explore the interactive map.")


if __name__ == "__main__":
    main()
