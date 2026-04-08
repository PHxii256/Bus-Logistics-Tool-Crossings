"""
Safety-Aware Dynamic Detour Engine for School Bus Optimization

This module provides the core algorithms for:
1. Geocoding student addresses to the road network (snapping to edges)
2. Validating safe pedestrian paths (no arterial crossings)
3. Calculating Cheapest Insertion costs for student assignments
4. Managing temporary and permanent student detour requests
5. Enforcing safety and time constraints
"""
import requests
import networkx as nx
import osmnx as ox
import math
import heapq
import time
from shapely.geometry import Point, LineString
from shapely.ops import substring


# ============================================================================
# WALKING DISTANCE AND PENALTY CALCULATIONS
# ============================================================================

# Cached undirected graph for pedestrian walking (ignores one-way, U-turn rules)
_WALK_GRAPH = None
# Synthetic crossing markers created for the walk graph
_SYNTHETIC_CROSSINGS = []
# Synthetic crossing config/state for on-demand generation per drive node
_SYNTHETIC_CFG = {}
_SYNTHETIC_TOTAL_ADDED = 0
_SYNTHETIC_DRIVE_DONE = set()
_SYNTHETIC_DIAGNOSTICS = {}
_SYNTHETIC_REJECTED_UNSAFE = []
# Debug counters for synthetic crossing usage in BFS
_CROSSING_BFS_STATS = {
    "students_checked": 0,
    "candidates_via_crossing": 0,  # drive nodes only reachable via synthetic crossing
    "students_with_crossing_benefit": 0,  # students who got extra candidates via crossings
    "students_explored_crossing": 0,  # students whose BFS traversed >=1 synthetic crossing edge
    "allowed_students_checked": 0,  # policy-eligible students checked by BFS
    "allowed_students_explored_crossing": 0,  # eligible students whose BFS traversed crossings
}
# Cache: (student_node, stop_node) -> walk_distance_meters
_WALK_DIST_CACHE = {}
# Cache: (walk_graph_id, drive_node_id) -> mapped_walk_node_id
_WALK_NODE_MAP_CACHE = {}
# Walk-graph spatial hash for lightweight neighborhood lookup
_WALK_SPATIAL_INDEX = None
_WALK_SPATIAL_INDEX_META = None
# Cache: (drive_graph_id, walk_node_id) -> nearest drive node id
_WALK_TO_DRIVE_NODE_CACHE = {}
# Cache: (drive_graph_id, drive_node_id) -> lane signature dict or None
_DRIVE_NODE_SIGNATURE_CACHE = {}
# Cache: (drive_graph_id, drive_node_id) -> bool safe-to-cross around node
_DRIVE_NODE_SAFE_CACHE = {}
# Cache for class/name-constrained signatures used by named-opposite strategy
_DRIVE_NODE_FILTERED_SIG_CACHE = {}
# Cache: (lat, lon) -> nearest_graph_node for walking
_STUDENT_NODE_CACHE = {}

_MAJOR_HIGHWAYS = {"motorway", "trunk", "primary", "secondary"}

_DEFAULT_STAGE_CROSSING_POLICY = {
    "HIGH": {"secondary", "tertiary"},
    "MIDDLE": {"tertiary"},
    "ELEMENTARY": set(),
    "KG": set(),
    "DISABLED": set(),
    "UNKNOWN": set(),
}
_SYNTHETIC_STAGE_CROSSING_POLICY = {
    key: set(values) for key, values in _DEFAULT_STAGE_CROSSING_POLICY.items()
}


def _is_valid_coordinate(value):
    """Check if coordinate is valid (not None and not NaN)."""
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return True


def _are_valid_coordinates(*coords):
    """Check if all coordinates are valid (not None and not NaN)."""
    return all(_is_valid_coordinate(c) for c in coords)


def _normalize_highway(hw):
    """Normalize OSM highway value to a base type."""
    if isinstance(hw, list):
        hw = hw[0] if hw else ""
    hw = str(hw or "").lower()
    return hw[:-5] if hw.endswith("_link") else hw


def _normalize_crossing_road_class(value):
    """Normalize a crossing road class token (e.g. secondary_link -> secondary)."""
    return _normalize_highway(value)


def _normalize_stage_crossing_policy(raw_policy):
    """Return canonical stage->allowed crossing classes mapping."""
    policy = {key: set(values) for key, values in _DEFAULT_STAGE_CROSSING_POLICY.items()}
    if not isinstance(raw_policy, dict):
        return policy

    for raw_key, raw_classes in raw_policy.items():
        key = str(raw_key or "").strip().upper()
        if "DISABLED" in key:
            key = "DISABLED"
        if key not in policy:
            continue

        if raw_classes is None:
            policy[key] = set()
            continue
        if not isinstance(raw_classes, (list, tuple, set)):
            continue

        normalized = set()
        for cls in raw_classes:
            token = _normalize_crossing_road_class(cls)
            if token:
                normalized.add(token)
        policy[key] = normalized
    return policy


def _resolve_stage_crossing_policy_key(student_stage, student_disabled=False):
    """Map student attributes to crossing policy key."""
    if bool(student_disabled):
        return "DISABLED"
    if hasattr(student_stage, "name"):
        stage_key = str(student_stage.name).strip().upper()
    else:
        stage_key = str(student_stage or "").strip().upper()
    if stage_key in _SYNTHETIC_STAGE_CROSSING_POLICY:
        return stage_key
    return "UNKNOWN"


def _allowed_synthetic_crossing_classes(student_stage, student_disabled=False):
    """Return allowed crossing road classes for the given student."""
    policy_key = _resolve_stage_crossing_policy_key(student_stage, student_disabled)
    return _SYNTHETIC_STAGE_CROSSING_POLICY.get(policy_key, set())


def _edge_allows_student_crossing(edge_data, student_stage, student_disabled=False):
    """Return whether a synthetic crossing edge is allowed for this student."""
    if not bool(edge_data.get("synthetic_crossing", False)):
        return True

    allowed_classes = _allowed_synthetic_crossing_classes(student_stage, student_disabled)
    if not allowed_classes:
        return False

    road_class = _normalize_crossing_road_class(edge_data.get("road_class"))
    if not road_class:
        # Unknown synthetic crossing class: default deny for safety.
        return False
    return road_class in allowed_classes


def _normalize_road_name(name):
    """Normalize OSM road name into a stable lowercase token string."""
    if isinstance(name, list):
        name = name[0] if name else ""
    txt = str(name or "").strip().lower()
    if not txt:
        return ""
    # Normalize frequent separator/punctuation variance.
    txt = txt.replace("-", " ").replace("_", " ").replace("/", " ")
    while "  " in txt:
        txt = txt.replace("  ", " ")
    return txt


def _angle_diff_deg(a, b):
    """Smallest absolute angular difference in degrees (0..180)."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _edge_bearing_fallback(graph, u, v):
    """Compute bearing if edge metadata doesn't include one."""
    try:
        lat1, lon1 = graph.nodes[u]["y"], graph.nodes[u]["x"]
        lat2, lon2 = graph.nodes[v]["y"], graph.nodes[v]["x"]
    except Exception:
        return None
    dy = lat2 - lat1
    dx = lon2 - lon1
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return None
    ang = math.degrees(math.atan2(dx, dy))
    return (ang + 360.0) % 360.0


def _bearing_between_walk_nodes(walk_graph, u, v):
    """Compute bearing from walk node u to walk node v."""
    try:
        lat1, lon1 = walk_graph.nodes[u]["y"], walk_graph.nodes[u]["x"]
        lat2, lon2 = walk_graph.nodes[v]["y"], walk_graph.nodes[v]["x"]
    except Exception:
        return None
    dy = lat2 - lat1
    dx = lon2 - lon1
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return None
    ang = math.degrees(math.atan2(dx, dy))
    return (ang + 360.0) % 360.0


def _walk_node_local_bearing(walk_graph, node_id):
    """Return a representative local bearing at a walk node."""
    try:
        edges = list(walk_graph.edges(node_id, keys=True, data=True))
    except Exception:
        return None
    best = None
    for u, v, k, data in edges:
        nbr = v if u == node_id else u
        b = _bearing_between_walk_nodes(walk_graph, node_id, nbr)
        if b is None:
            continue
        length = float(data.get("length", 1e9))
        if best is None or length < best[0]:
            best = (length, b)
    return best[1] if best else None


def _parallel_alignment_delta_deg(a, b):
    """Return delta-to-parallel in degrees, where 0 means parallel.

    Treat same-direction and opposite-direction as equally parallel.
    """
    d = _angle_diff_deg(a, b)
    return min(d, abs(180.0 - d))


def _is_parallel_carriageway_pair(
    drive_graph,
    walk_graph,
    walk_node_a,
    walk_node_b,
    max_parallel_delta_deg=25.0,
    require_same_highway=True,
):
    """True when both sides map to major carriageway lanes that are parallel.

    Residential/local roads are excluded because signatures are extracted only
    from major one-way highways in _drive_node_signature().
    """
    if drive_graph is None:
        return False
    d1 = _nearest_drive_node_for_walk_node(walk_node_a, walk_graph, drive_graph)
    d2 = _nearest_drive_node_for_walk_node(walk_node_b, walk_graph, drive_graph)
    if d1 is None or d2 is None or d1 == d2:
        return False
    s1 = _drive_node_signature(drive_graph, d1)
    s2 = _drive_node_signature(drive_graph, d2)
    if s1 is None or s2 is None:
        return False
    if require_same_highway and s1["highway"] != s2["highway"]:
        return False
    return _parallel_alignment_delta_deg(s1["bearing"], s2["bearing"]) <= float(max_parallel_delta_deg)


def _drive_node_signature(drive_graph, drive_node_id):
    """Extract representative lane signature near a drive node.

    Signature is used to detect opposite-direction major-road carriageways.
    """
    ckey = (id(drive_graph), drive_node_id)
    if ckey in _DRIVE_NODE_SIGNATURE_CACHE:
        return _DRIVE_NODE_SIGNATURE_CACHE[ckey]

    candidates = []

    try:
        out_edges = list(drive_graph.out_edges(drive_node_id, keys=True, data=True))
        in_edges = list(drive_graph.in_edges(drive_node_id, keys=True, data=True))
    except Exception:
        out_edges = []
        in_edges = []

    for u, v, k, data in out_edges + in_edges:
        hw = _normalize_highway(data.get("highway", ""))
        if hw not in _MAJOR_HIGHWAYS:
            continue
        oneway = data.get("oneway", False)
        if isinstance(oneway, str):
            oneway = oneway.lower() in ("yes", "true", "1")
        if not bool(oneway):
            continue
        bearing = data.get("bearing")
        if bearing is None:
            bearing = _edge_bearing_fallback(drive_graph, u, v)
        if bearing is None:
            continue
        length = float(data.get("length", 1e9))
        candidates.append((length, {
            "highway": hw,
            "bearing": float(bearing),
            "oneway": True,
        }))

    sig = min(candidates, key=lambda x: x[0])[1] if candidates else None
    _DRIVE_NODE_SIGNATURE_CACHE[ckey] = sig
    return sig


def _drive_node_filtered_signature(
    drive_graph,
    drive_node_id,
    allowed_highways=None,
    require_name=False,
):
    """Return a representative edge signature with highway/name filters."""
    hw_key = tuple(sorted(allowed_highways or []))
    ckey = (id(drive_graph), drive_node_id, hw_key, bool(require_name))
    if ckey in _DRIVE_NODE_FILTERED_SIG_CACHE:
        return _DRIVE_NODE_FILTERED_SIG_CACHE[ckey]

    candidates = []
    try:
        out_edges = list(drive_graph.out_edges(drive_node_id, keys=True, data=True))
        in_edges = list(drive_graph.in_edges(drive_node_id, keys=True, data=True))
    except Exception:
        out_edges = []
        in_edges = []

    for u, v, k, data in out_edges + in_edges:
        hw = _normalize_highway(data.get("highway", ""))
        if allowed_highways and hw not in allowed_highways:
            continue
        oneway = data.get("oneway", False)
        if isinstance(oneway, str):
            oneway = oneway.lower() in ("yes", "true", "1")
        if not bool(oneway):
            continue
        road_name = _normalize_road_name(data.get("name", ""))
        if require_name and not road_name:
            continue
        bearing = data.get("bearing")
        if bearing is None:
            bearing = _edge_bearing_fallback(drive_graph, u, v)
        if bearing is None:
            continue
        edge_len = float(data.get("length", 1e9))
        candidates.append((edge_len, {
            "highway": hw,
            "bearing": float(bearing),
            "oneway": True,
            "name": road_name,
        }))

    sig = min(candidates, key=lambda x: x[0])[1] if candidates else None
    _DRIVE_NODE_FILTERED_SIG_CACHE[ckey] = sig
    return sig


def _drive_node_is_safe_to_cross(drive_graph, drive_node_id):
    """Return True only when local drive edges around the node are safe to cross."""
    ckey = (id(drive_graph), drive_node_id)
    if ckey in _DRIVE_NODE_SAFE_CACHE:
        return _DRIVE_NODE_SAFE_CACHE[ckey]

    edge_iter = []
    try:
        edge_iter.extend(list(drive_graph.out_edges(drive_node_id, keys=True, data=True)))
        edge_iter.extend(list(drive_graph.in_edges(drive_node_id, keys=True, data=True)))
    except Exception:
        _DRIVE_NODE_SAFE_CACHE[ckey] = True
        return True

    if not edge_iter:
        _DRIVE_NODE_SAFE_CACHE[ckey] = True
        return True

    best_len = float("inf")
    best_safe = True
    for _u, _v, _k, data in edge_iter:
        edge_len = float(data.get("length", 1e9))
        edge_safe = bool(data.get("is_safe_to_cross", True))
        if edge_len < best_len:
            best_len = edge_len
            best_safe = edge_safe

    _DRIVE_NODE_SAFE_CACHE[ckey] = best_safe
    return best_safe


def _nearest_drive_node_for_walk_node(walk_node_id, walk_graph, drive_graph):
    """Map a walk node to the nearest drive node (cached)."""
    ckey = (id(drive_graph), walk_node_id)
    cached = _WALK_TO_DRIVE_NODE_CACHE.get(ckey)
    if cached is not None:
        return cached
    try:
        lat = walk_graph.nodes[walk_node_id]["y"]
        lon = walk_graph.nodes[walk_node_id]["x"]
    except Exception:
        return None
    try:
        # Build once, then query in O(log n) for many walk nodes.
        _get_or_build_ball_tree(drive_graph)
        dn = fast_nearest_node(drive_graph, lon, lat)
    except Exception:
        dn = None
    _WALK_TO_DRIVE_NODE_CACHE[ckey] = dn
    return dn


def _is_dual_carriageway_pair(drive_graph, walk_graph, base_drive_node, other_walk_node, min_opposite_deg=150.0):
    """Return True when nodes look like opposite lanes of a dual carriageway."""
    base_sig = _drive_node_signature(drive_graph, base_drive_node)
    if not base_sig:
        return False

    other_drive = _nearest_drive_node_for_walk_node(other_walk_node, walk_graph, drive_graph)
    if other_drive is None or other_drive == base_drive_node:
        return False

    other_sig = _drive_node_signature(drive_graph, other_drive)
    if not other_sig:
        return False

    # Keep same major road class and opposite movement direction.
    if base_sig["highway"] != other_sig["highway"]:
        return False
    return _angle_diff_deg(base_sig["bearing"], other_sig["bearing"]) >= float(min_opposite_deg)

def _get_walk_graph(graph):
    """Get or create a graph for walking.

    If a dedicated walk graph was set, use it. Otherwise, fall back
    to an undirected version of the provided graph.
    """
    global _WALK_GRAPH
    if _WALK_GRAPH is not None:
        return _WALK_GRAPH
    _WALK_GRAPH = graph.to_undirected()
    return _WALK_GRAPH


def _map_to_walk_node(node_id, drive_graph, walk_graph, generate_synthetic=False):
    """Map a node from drive_graph to the nearest node in walk_graph."""
    if node_id in walk_graph:
        if generate_synthetic:
            _ensure_synthetic_near_drive_node(node_id, drive_graph, walk_graph, node_id)
        return node_id
    cache_key = (id(walk_graph), node_id)
    cached = _WALK_NODE_MAP_CACHE.get(cache_key)
    if cached is not None:
        if generate_synthetic:
            _ensure_synthetic_near_drive_node(node_id, drive_graph, walk_graph, cached)
        return cached
    try:
        lat = drive_graph.nodes[node_id]['y']
        lon = drive_graph.nodes[node_id]['x']
    except Exception:
        return None

    mapped = _nearest_node_any_id(walk_graph, lon, lat)
    
    _WALK_NODE_MAP_CACHE[cache_key] = mapped
    if generate_synthetic:
        _ensure_synthetic_near_drive_node(node_id, drive_graph, walk_graph, mapped)
    return mapped


def _nearest_node_any_id(graph, lon, lat):
    """Nearest node lookup that works with int and string node IDs.

    osmnx.nearest_nodes may fail on some graphs when node IDs are non-numeric.
    Fallback to BallTree/manual search to keep behavior robust.
    """
    try:
        return ox.nearest_nodes(graph, lon, lat)
    except Exception:
        pass

    try:
        tree, node_ids = _get_or_build_ball_tree(graph)
        import numpy as np
        pt_rad = np.deg2rad([[lat, lon]])
        _, pos = tree.query(pt_rad, k=1)
        return node_ids[pos[0][0]]
    except Exception:
        pass

    # Last resort: linear scan in degree space (small local fallback path only).
    best = None
    best_d2 = float('inf')
    lat_rad = math.radians(lat)
    cos_lat = max(1e-6, math.cos(lat_rad))
    for n, d in graph.nodes(data=True):
        nlat = d.get('y')
        nlon = d.get('x')
        if nlat is None or nlon is None:
            continue
        dy = float(nlat) - float(lat)
        dx = (float(nlon) - float(lon)) * cos_lat
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best = n
    return best


def _build_walk_spatial_index(walk_graph, cell_m):
    """Build a lightweight spatial hash for walk nodes."""
    global _WALK_SPATIAL_INDEX, _WALK_SPATIAL_INDEX_META
    meta = (id(walk_graph), float(cell_m))
    if _WALK_SPATIAL_INDEX is not None and _WALK_SPATIAL_INDEX_META == meta:
        return _WALK_SPATIAL_INDEX

    idx = {}
    for n, d in walk_graph.nodes(data=True):
        lat = d.get('y')
        lon = d.get('x')
        if not _are_valid_coordinates(lat, lon):
            continue
        y_m = lat * 111000.0
        x_m = lon * 111000.0
        gx = int(x_m // cell_m)
        gy = int(y_m // cell_m)
        idx.setdefault((gx, gy), []).append(n)

    _WALK_SPATIAL_INDEX = idx
    _WALK_SPATIAL_INDEX_META = meta
    return idx


def _ensure_synthetic_near_drive_node(drive_node_id, drive_graph, walk_graph, mapped_walk_node):
    """Create a few synthetic crossings once per drive node (on-demand)."""
    global _SYNTHETIC_TOTAL_ADDED
    cfg = _SYNTHETIC_CFG or {}
    strategy = str(cfg.get('strategy', 'per_drive_node')).lower()
    if strategy == 'batch':
        return
    diag = _SYNTHETIC_DIAGNOSTICS
    if diag:
        diag["checked_drive_nodes"] = int(diag.get("checked_drive_nodes", 0)) + 1
    if not cfg.get('enabled', False):
        return
    if drive_node_id in _SYNTHETIC_DRIVE_DONE:
        return

    max_total = int(cfg.get('max_total', 1000))
    if _SYNTHETIC_TOTAL_ADDED >= max_total:
        _SYNTHETIC_DRIVE_DONE.add(drive_node_id)
        return

    if mapped_walk_node not in walk_graph:
        _SYNTHETIC_DRIVE_DONE.add(drive_node_id)
        return

    min_dist_m = float(cfg.get('min_dist_m', 6.0))
    max_dist_m = float(cfg.get('max_dist_m', 20.0))
    per_drive = int(cfg.get('max_per_drive_node', cfg.get('max_per_node', 1)))
    require_dual = bool(cfg.get('require_dual_carriageway', True))
    min_opposite_deg = float(cfg.get('min_opposite_bearing_deg', 150.0))
    require_perp = bool(cfg.get('require_perpendicular_crossing', True))
    min_perp_deg = float(cfg.get('min_perpendicular_deg', 60.0))
    max_perp_deg = float(cfg.get('max_perpendicular_deg', 120.0))
    max_existing_path_m = float(cfg.get('max_existing_path_m', 80.0))
    exclude_unsafe_roads = bool(cfg.get('exclude_unsafe_roads', True))
    base_sig = _drive_node_signature(drive_graph, drive_node_id)
    if exclude_unsafe_roads and not _drive_node_is_safe_to_cross(drive_graph, drive_node_id):
        if diag:
            diag["rejected_unsafe_road"] = int(diag.get("rejected_unsafe_road", 0)) + 1
        _SYNTHETIC_REJECTED_UNSAFE.append({
            "lat": float(walk_graph.nodes[mapped_walk_node].get("y", 0.0)),
            "lon": float(walk_graph.nodes[mapped_walk_node].get("x", 0.0)),
            "reason": "base_drive_node_unsafe",
        })
        _SYNTHETIC_DRIVE_DONE.add(drive_node_id)
        return
    if require_dual and base_sig is None:
        _SYNTHETIC_DRIVE_DONE.add(drive_node_id)
        return
    if per_drive <= 0:
        _SYNTHETIC_DRIVE_DONE.add(drive_node_id)
        return

    node_lat = walk_graph.nodes[mapped_walk_node].get('y')
    node_lon = walk_graph.nodes[mapped_walk_node].get('x')
    if not _are_valid_coordinates(node_lat, node_lon):
        _SYNTHETIC_DRIVE_DONE.add(drive_node_id)
        return

    cell_m = max(max_dist_m, 1.0)
    idx = _build_walk_spatial_index(walk_graph, cell_m)
    y_m = node_lat * 111000.0
    x_m = node_lon * 111000.0
    gx = int(x_m // cell_m)
    gy = int(y_m // cell_m)

    candidate_ids = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            candidate_ids.extend(idx.get((gx + dx, gy + dy), []))

    best = []
    for n2 in candidate_ids:
        if diag:
            diag["candidate_pairs_checked"] = int(diag.get("candidate_pairs_checked", 0)) + 1
        if n2 == mapped_walk_node:
            continue
        if walk_graph.has_edge(mapped_walk_node, n2) or walk_graph.has_edge(n2, mapped_walk_node):
            if diag:
                diag["rejected_existing_edge"] = int(diag.get("rejected_existing_edge", 0)) + 1
            continue
        lat2 = walk_graph.nodes[n2].get('y')
        lon2 = walk_graph.nodes[n2].get('x')
        if lat2 is None or lon2 is None:
            continue
        dist_m = math.hypot((node_lat - lat2) * 111000.0, (node_lon - lon2) * 111000.0)
        if dist_m < min_dist_m or dist_m > max_dist_m:
            if diag:
                diag["rejected_distance"] = int(diag.get("rejected_distance", 0)) + 1
            continue
        if exclude_unsafe_roads:
            other_drive = _nearest_drive_node_for_walk_node(n2, walk_graph, drive_graph)
            if other_drive is None or not _drive_node_is_safe_to_cross(drive_graph, other_drive):
                if diag:
                    diag["rejected_unsafe_road"] = int(diag.get("rejected_unsafe_road", 0)) + 1
                _SYNTHETIC_REJECTED_UNSAFE.append({
                    "lat": (node_lat + lat2) / 2,
                    "lon": (node_lon + lon2) / 2,
                    "reason": "candidate_pair_unsafe",
                })
                continue
        if require_dual and not _is_dual_carriageway_pair(
            drive_graph, walk_graph, drive_node_id, n2, min_opposite_deg=min_opposite_deg
        ):
            if diag:
                diag["rejected_lane_pair"] = int(diag.get("rejected_lane_pair", 0)) + 1
            continue
        if require_perp and base_sig is not None:
            cross_bearing = _bearing_between_walk_nodes(walk_graph, mapped_walk_node, n2)
            if cross_bearing is None:
                if diag:
                    diag["rejected_missing_cross_bearing"] = int(diag.get("rejected_missing_cross_bearing", 0)) + 1
                continue
            cross_diff = _angle_diff_deg(base_sig["bearing"], cross_bearing)
            if cross_diff < min_perp_deg or cross_diff > max_perp_deg:
                if diag:
                    diag["rejected_not_perpendicular"] = int(diag.get("rejected_not_perpendicular", 0)) + 1
                continue
        if max_existing_path_m > 0:
            try:
                existing_len = nx.shortest_path_length(walk_graph, mapped_walk_node, n2, weight='length')
                if existing_len <= max_existing_path_m:
                    if diag:
                        diag["rejected_existing_walk_short"] = int(diag.get("rejected_existing_walk_short", 0)) + 1
                    continue
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
        best.append((dist_m, n2, lat2, lon2))

    best.sort(key=lambda x: x[0])
    added_here = 0
    for dist_m, n2, lat2, lon2 in best:
        if added_here >= per_drive or _SYNTHETIC_TOTAL_ADDED >= max_total:
            break
        if walk_graph.has_edge(mapped_walk_node, n2) or walk_graph.has_edge(n2, mapped_walk_node):
            continue
        walk_graph.add_edge(mapped_walk_node, n2, length=dist_m, travel_time=dist_m / 80.0,
                            synthetic_crossing=True, is_safe_to_cross=True)
        _SYNTHETIC_CROSSINGS.append({
            'lat': (node_lat + lat2) / 2,
            'lon': (node_lon + lon2) / 2,
            'length_m': round(dist_m, 1),
            'drive_node_id': drive_node_id,
        })
        _SYNTHETIC_TOTAL_ADDED += 1
        if diag:
            diag["added"] = int(diag.get("added", 0)) + 1
        added_here += 1

    _SYNTHETIC_DRIVE_DONE.add(drive_node_id)


def _store_debug_edges(edges, walk_nodes=None, synthetic_nodes=None):
    """Store drive edges, walk nodes, and synthetic nodes for debug visualization."""
    global _CROSSING_DEBUG_CANDIDATES
    _CROSSING_DEBUG_CANDIDATES = {
        "edges": edges,
        "walk_nodes": walk_nodes or [],
        "synthetic_nodes": synthetic_nodes or [],
        "candidates": [],
        "direction_stats": {},
    }


_CROSSING_DEBUG_CANDIDATES = {}


def get_crossing_debug_candidates():
    """Return debug info about crossing candidates."""
    return dict(_CROSSING_DEBUG_CANDIDATES)


# ============================================================================
# DRIVE NODE CROSSING FUNCTIONS
# ============================================================================

def _ensure_node_in_walk_graph(walk_graph, node_id, drive_graph):
    """Ensure a drive node exists in walk_graph with proper coordinates.

    Returns True if node exists or was successfully added with valid coordinates.
    Returns False if node has invalid/missing coordinates and should be skipped.
    """
    if node_id in walk_graph:
        return True  # Already exists
    if node_id in drive_graph:
        lat = drive_graph.nodes[node_id].get('y')
        lon = drive_graph.nodes[node_id].get('x')
        if _are_valid_coordinates(lat, lon):
            walk_graph.add_node(node_id, x=lon, y=lat)
            return True
    return False  # Invalid coordinates or node not found


def build_drive_node_crossings(
    walk_graph,
    drive_graph,
    min_dist_m=6.0,
    max_dist_m=60.0,
    min_opposite_bearing_deg=150.0,
    max_crossing_angle_delta_deg=30.0,
    min_perpendicular_deg=None,
    max_perpendicular_deg=None,
    min_spacing_m=100.0,
    min_spacing_per_road_m=100.0,
    center_lat=None,
    center_lon=None,
    radius_km=None,
    enable_guaranteed_connection=True,
    guaranteed_connection_distance_m=6.0,
    fallback_mode="adaptive",
):
    """Build crossings between drive nodes on opposite sides of secondary/tertiary roads.

    Algorithm:
    1. Extract drive nodes from oneway secondary/tertiary edges
    2. Group by road name, split into opposite directions A/B
    3. Sort both sides by position along road axis
    4. For each B node: find closest unpaired A, validate, pair or skip
    5. Create synthetic nodes for unpaired drive nodes

    NOTE: Connects ALL possible drive nodes - no max_per_road or max_total limits.

    Returns:
        (crossings, diagnostics, debug_info)
    """
    crossings = []
    diag = {
        "strategy": "drive_node_crossings",
        "drive_edges_found": 0,
        "drive_nodes_extracted": 0,
        "roads_found": 0,
        "roads_with_both_directions": 0,
        "real_to_real_crossings": 0,
        "real_to_synthetic_crossings": 0,
        "synthetic_nodes_created": 0,
        "rejected_no_opposite": 0,

        # NEW: Granular pairing rejection tracking
        "pairing_attempts": 0,
        "pairing_rejected_distance_too_short": 0,
        "pairing_rejected_distance_too_long": 0,
        "pairing_rejected_angle": 0,
        "pairing_rejected_bearing_calc_failed": 0,

        # Existing counters
        "rejected_distance": 0,  # Legacy counter, will be replaced by granular ones
        "rejected_angle": 0,     # Legacy counter, will be replaced by granular ones
        "rejected_spacing": 0,
        "skipped_global_connection_lock": 0,
        "rejected_fallback_geometry": 0,
        "fallback_reused_existing_nodes": 0,

        # NEW: Enhanced 3-stage fallback tracking
        "fallback_projected_with_spacing": 0,
        "fallback_projected_no_spacing": 0,
        "fallback_guaranteed": 0,
        "fallback_failed_completely": 0,
    }

    debug_info = {
        "edges": [],
        "drive_nodes": [],
        "synthetic_nodes": [],
        "crossings": [],
    }

    if walk_graph is None or drive_graph is None:
        global _SYNTHETIC_DIAGNOSTICS
        _SYNTHETIC_DIAGNOSTICS = diag
        _store_debug_edges([], [], [])
        return crossings, diag, debug_info

    cos_lat_ref = math.cos(math.radians(center_lat)) if center_lat is not None else 1.0
    allowed_hw = {"secondary", "tertiary"}

    # Helper closures for coordinate conversion
    def _to_meters(lat, lon):
        return lon * 111000.0 * cos_lat_ref, lat * 111000.0

    def _from_meters(x_m, y_m):
        return y_m / 111000.0, x_m / (111000.0 * cos_lat_ref)

    def _bearing_from_xy(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return None
        return math.degrees(math.atan2(dx, dy)) % 360.0

    def _perp_delta_deg(crossing_bearing, road_axis_bearing):
        p1 = (road_axis_bearing + 90.0) % 360.0
        p2 = (road_axis_bearing + 270.0) % 360.0
        return min(_angle_diff_deg(crossing_bearing, p1), _angle_diff_deg(crossing_bearing, p2))

    def _is_crossing_angle_valid(crossing_bearing, road_axis_bearing):
        """Validate crossing angle against configured perpendicular constraints.

        Returns tuple: (is_valid, debug_delta_to_perpendicular).
        """
        cross_axis = _angle_diff_deg(crossing_bearing, road_axis_bearing)
        delta_to_perp = abs(cross_axis - 90.0)

        if min_perpendicular_deg is not None or max_perpendicular_deg is not None:
            min_deg = float(min_perpendicular_deg) if min_perpendicular_deg is not None else (90.0 - float(max_crossing_angle_delta_deg))
            max_deg = float(max_perpendicular_deg) if max_perpendicular_deg is not None else (90.0 + float(max_crossing_angle_delta_deg))
            return min_deg <= cross_axis <= max_deg, delta_to_perp

        return delta_to_perp <= float(max_crossing_angle_delta_deg), delta_to_perp

    def _in_radius(lat, lon):
        if center_lat is None or center_lon is None or radius_km is None:
            return True
        dlat = abs(lat - center_lat) * 111.0
        dlon = abs(lon - center_lon) * 111.0 * cos_lat_ref
        return math.sqrt(dlat * dlat + dlon * dlon) <= radius_km

    # Step 1: Extract drive nodes from oneway secondary/tertiary edges
    drive_edges_by_name = {}  # (road_name, highway) -> list of edge dicts

    for u, v, key, data in drive_graph.edges(keys=True, data=True):
        hw = data.get("highway", "")
        if isinstance(hw, list):
            hw = hw[0] if hw else ""
        hw = str(hw).lower().strip()

        if hw not in allowed_hw:
            continue

        oneway = data.get("oneway", False)
        if isinstance(oneway, str):
            oneway = oneway.lower() in ("yes", "true", "1")
        if not oneway:
            continue

        name = data.get("name", "")
        if isinstance(name, list):
            name = name[0] if name else ""
        name = str(name).strip()
        if not name:
            continue

        u_data = drive_graph.nodes[u]
        v_data = drive_graph.nodes[v]
        u_lat, u_lon = u_data.get("y"), u_data.get("x")
        v_lat, v_lon = v_data.get("y"), v_data.get("x")

        if not _are_valid_coordinates(u_lat, u_lon, v_lat, v_lon):
            continue

        mid_lat = (u_lat + v_lat) / 2.0
        mid_lon = (u_lon + v_lon) / 2.0
        if not _in_radius(mid_lat, mid_lon):
            continue

        bearing = data.get("bearing")
        if bearing is None:
            dlat = (v_lat - u_lat) * 111000.0
            dlon = (v_lon - u_lon) * 111000.0 * cos_lat_ref
            bearing = math.degrees(math.atan2(dlon, dlat)) % 360.0

        road_key = (name, hw)
        if road_key not in drive_edges_by_name:
            drive_edges_by_name[road_key] = []

        u_x, u_y = _to_meters(u_lat, u_lon)
        v_x, v_y = _to_meters(v_lat, v_lon)

        drive_edges_by_name[road_key].append({
            "u": u, "v": v, "key": key,
            "u_lat": u_lat, "u_lon": u_lon,
            "v_lat": v_lat, "v_lon": v_lon,
            "u_x": u_x, "u_y": u_y,
            "v_x": v_x, "v_y": v_y,
            "name": name, "highway": hw,
            "bearing": float(bearing),
        })

    diag["drive_edges_found"] = sum(len(edges) for edges in drive_edges_by_name.values())
    diag["roads_found"] = len(drive_edges_by_name)

    if not drive_edges_by_name:
        _SYNTHETIC_DIAGNOSTICS = diag
        _store_debug_edges([], [], [])
        return crossings, diag, debug_info

    # Extract unique drive nodes from edges
    drive_nodes_by_road = {}  # road_key -> {node_id -> node_record}

    for road_key, edges in drive_edges_by_name.items():
        nodes = {}
        for edge in edges:
            # Add both endpoints
            for endpoint in ['u', 'v']:
                node_id = edge[endpoint]
                if node_id not in nodes:
                    lat = edge[f"{endpoint}_lat"]
                    lon = edge[f"{endpoint}_lon"]
                    x = edge[f"{endpoint}_x"]
                    y = edge[f"{endpoint}_y"]
                    nodes[node_id] = {
                        "node_id": node_id,
                        "lat": lat,
                        "lon": lon,
                        "x": x,
                        "y": y,
                        "edge_bearing": edge["bearing"],
                        "road_name": edge["name"],
                        "highway": edge["highway"],
                    }
        drive_nodes_by_road[road_key] = nodes

    diag["drive_nodes_extracted"] = sum(len(nodes) for nodes in drive_nodes_by_road.values())

    # Synthetic node counter
    synthetic_counter = [0]

    def _create_synthetic_node(lat, lon, direction):
        """Create a new synthetic walk node."""
        node_id = f"synth_drive_crossing_{synthetic_counter[0]}"
        synthetic_counter[0] += 1
        walk_graph.add_node(node_id, y=lat, x=lon, synthetic=True)
        x_m, y_m = _to_meters(lat, lon)
        return {
            "node_id": node_id,
            "lat": lat,
            "lon": lon,
            "x": x_m,
            "y": y_m,
            "direction": direction,
        }

    def _create_fallback_crossing(
        source_node,
        target_x,
        target_y,
        crossing_type,
        globally_connected_nodes,
        existing_crossings,
        all_crossing_midpoints,
        walk_graph,
        diag,
        debug_info,
        road_key,
        synthetic_counter,
        _from_meters,
        _create_synthetic_node,
        _find_existing_node_near_xy,
    ):
        """Helper to create fallback crossing and update all tracking.

        Returns True if crossing was created successfully, False otherwise.
        """
        dist = math.hypot(source_node["x"] - target_x, source_node["y"] - target_y)
        mid_x, mid_y = (source_node["x"] + target_x) / 2, (source_node["y"] + target_y) / 2

        # Find or create target node
        target_node = _find_existing_node_near_xy(target_x, target_y, tolerance_m=1.5)
        created_synthetic = False

        if target_node is None:
            target_lat, target_lon = _from_meters(target_x, target_y)
            # Synthetic node is on opposite side from source
            source_dir = source_node.get("direction", "A")
            target_dir = "B" if source_dir == "A" else "A"
            target_node = _create_synthetic_node(target_lat, target_lon, target_dir)
            created_synthetic = True
            diag["synthetic_nodes_created"] += 1

            # Add to debug_info for visualization
            debug_info["synthetic_nodes"].append({
                "node_id": target_node["node_id"],
                "lat": target_lat,
                "lon": target_lon,
                "direction": target_dir,
                "road_name": road_key[0],
            })
        else:
            diag["fallback_reused_existing_nodes"] += 1

        # Check if target already connected
        if target_node["node_id"] in globally_connected_nodes:
            diag["skipped_global_connection_lock"] += 1
            return False

        # Create edge
        n1, n2 = source_node["node_id"], target_node["node_id"]
        edge_key = tuple(sorted([n1, n2], key=str))

        if walk_graph.has_edge(n1, n2) or edge_key in existing_crossings:
            return False

        # Ensure both nodes exist in walk_graph with proper coordinates
        if not _ensure_node_in_walk_graph(walk_graph, n1, drive_graph):
            return False
        if not _ensure_node_in_walk_graph(walk_graph, n2, drive_graph):
            return False

        walk_graph.add_edge(
            n1, n2,
            length=dist,
            travel_time=dist / 80.0,
            synthetic_crossing=True,
            is_safe_to_cross=True,
            crossing_rule="drive_node_crossings",
            crossing_subtype=crossing_type,
            road_name=road_key[0],
            road_class=road_key[1],
        )

        existing_crossings.add(edge_key)
        all_crossing_midpoints.append((mid_x, mid_y))
        globally_connected_nodes.add(n1)
        globally_connected_nodes.add(n2)

        # Add to crossings list
        final_crossing_type = "real_to_synthetic" if created_synthetic else "real_to_real"
        mid_lat = (float(source_node["lat"]) + float(target_node["lat"])) / 2.0
        mid_lon = (float(source_node["lon"]) + float(target_node["lon"])) / 2.0
        crossings.append({
            "lat": mid_lat,
            "lon": mid_lon,
            "node_a": n1,
            "node_b": n2,
            "lat_a": source_node["lat"],
            "lon_a": source_node["lon"],
            "lat_b": target_node["lat"],
            "lon_b": target_node["lon"],
            "length_m": dist,
            "crossing_type": final_crossing_type,
            "road_name": road_key[0],
        })

        # Add to debug_info
        debug_info["crossings"].append({
            "node_a": n1,
            "node_b": n2,
            "crossing_type": crossing_type,
            "road_name": road_key[0],
            "distance_m": round(dist, 2),
        })

        # Increment correct diagnostic counter
        if created_synthetic:
            diag["real_to_synthetic_crossings"] += 1
        else:
            diag["real_to_real_crossings"] += 1

        return True

    def _nearest_point_on_segment(px, py, ax, ay, bx, by):
        """Return nearest point on segment AB to point P."""
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        ab_len_sq = abx * abx + aby * aby
        if ab_len_sq < 1e-9:
            return ax, ay
        t = max(0, min(1, (apx * abx + apy * aby) / ab_len_sq))
        return ax + t * abx, ay + t * aby

    def _nearest_projection_to_side(source_x, source_y, segments):
        best_proj = None
        best_proj_dist = float("inf")
        for ax, ay, bx, by in segments:
            px, py = _nearest_point_on_segment(source_x, source_y, ax, ay, bx, by)
            dist = math.hypot(source_x - px, source_y - py)
            if dist < best_proj_dist:
                best_proj_dist = dist
                best_proj = (px, py)
        return best_proj

    def _fallback_target_on_opposite_side(source_node, target_segments):
        """Project fallback synthetic node directly onto opposite-side geometry."""
        if not target_segments:
            return None

        sx, sy = source_node["x"], source_node["y"]
        proj = _nearest_projection_to_side(sx, sy, target_segments)
        if proj is None:
            return None
        return proj

    def _find_existing_node_near_xy(x_m, y_m, tolerance_m=1.5):
        """Return an existing walk node near a projected fallback target.

        This prevents creating synthetic nodes exactly over existing real/synthetic nodes.
        """
        tol = float(tolerance_m)
        best = None
        best_d = float("inf")
        for node_id, ndata in walk_graph.nodes(data=True):
            lat = ndata.get("y")
            lon = ndata.get("x")
            if not _are_valid_coordinates(lat, lon):
                continue
            nx_m, ny_m = _to_meters(lat, lon)
            d = math.hypot(nx_m - x_m, ny_m - y_m)
            if d <= tol and d < best_d:
                best = {
                    "node_id": node_id,
                    "lat": lat,
                    "lon": lon,
                    "x": nx_m,
                    "y": ny_m,
                    "synthetic": bool(ndata.get("synthetic", False)),
                }
                best_d = d
        return best

    # Track all crossings for spacing and global node usage.
    all_crossing_midpoints = []
    existing_crossings = set()
    globally_connected_nodes = set()

    for u, v, k, data in walk_graph.edges(keys=True, data=True):
        if data.get("synthetic_crossing"):
            existing_crossings.add(tuple(sorted([u, v], key=str)))

    # Step 2: Process each road
    for road_key, node_dict in drive_nodes_by_road.items():
        nodes = list(node_dict.values())

        if len(nodes) < 2:
            continue

        # Split into opposite directions A/B based on bearing
        ref_bearing = nodes[0]["edge_bearing"]
        nodes_a, nodes_b = [], []

        for node in nodes:
            angle_diff = _angle_diff_deg(node["edge_bearing"], ref_bearing)
            if angle_diff < 90.0:
                node["direction"] = "A"
                nodes_a.append(node)
            else:
                node["direction"] = "B"
                nodes_b.append(node)

        if not nodes_a or not nodes_b:
            diag["rejected_no_opposite"] += 1
            continue

        # Check if opposite bearings differ enough
        avg_bearing_a = sum(n["edge_bearing"] for n in nodes_a) / len(nodes_a)
        avg_bearing_b = sum(n["edge_bearing"] for n in nodes_b) / len(nodes_b)
        if _angle_diff_deg(avg_bearing_a, avg_bearing_b) < min_opposite_bearing_deg:
            diag["rejected_no_opposite"] += 1
            continue

        diag["roads_with_both_directions"] += 1
        axis_bearing = avg_bearing_a

        # Store edges for debug visualization
        for edge in drive_edges_by_name[road_key]:
            # Determine direction based on bearing
            angle_diff = _angle_diff_deg(edge["bearing"], ref_bearing)
            direction = "A" if angle_diff < 90.0 else "B"
            debug_info["edges"].append({
                "u_lat": edge["u_lat"], "u_lon": edge["u_lon"],
                "v_lat": edge["v_lat"], "v_lon": edge["v_lon"],
                "name": edge["name"], "highway": edge["highway"],
                "bearing": edge["bearing"], "direction": direction,
            })

        # Sort nodes by position along road axis
        # Calculate reference point (centroid)
        ref_x = sum(n["x"] for n in nodes_a + nodes_b) / len(nodes_a + nodes_b)
        ref_y = sum(n["y"] for n in nodes_a + nodes_b) / len(nodes_a + nodes_b)

        # Create axis unit vector
        axis_rad = math.radians(axis_bearing)
        axis_dx = math.sin(axis_rad)
        axis_dy = math.cos(axis_rad)

        # Project nodes onto axis
        for node in nodes_a + nodes_b:
            dx = node["x"] - ref_x
            dy = node["y"] - ref_y
            node["axis_position"] = dx * axis_dx + dy * axis_dy

        nodes_a.sort(key=lambda n: n["axis_position"])
        nodes_b.sort(key=lambda n: n["axis_position"])

        # Store drive nodes for debug
        for node in nodes_a:
            debug_info["drive_nodes"].append({
                "node_id": node["node_id"],
                "lat": node["lat"],
                "lon": node["lon"],
                "road_name": node["road_name"],
                "highway": node["highway"],
                "direction": "A",
            })
        for node in nodes_b:
            debug_info["drive_nodes"].append({
                "node_id": node["node_id"],
                "lat": node["lat"],
                "lon": node["lon"],
                "road_name": node["road_name"],
                "highway": node["highway"],
                "direction": "B",
            })

        # Collect edge segments for each side (for synthetic projection)
        segs_a = []
        for edge in drive_edges_by_name[road_key]:
            angle_diff = _angle_diff_deg(edge["bearing"], ref_bearing)
            if angle_diff < 90.0:
                segs_a.append((edge["u_x"], edge["u_y"], edge["v_x"], edge["v_y"]))

        segs_b = []
        for edge in drive_edges_by_name[road_key]:
            angle_diff = _angle_diff_deg(edge["bearing"], ref_bearing)
            if angle_diff >= 90.0:
                segs_b.append((edge["u_x"], edge["u_y"], edge["v_x"], edge["v_y"]))

        # Step 3: Pair B -> A (greedy sorted matching)
        paired_a = set()
        paired_b = set()
        road_midpoints = []

        for node_b in nodes_b:
            if node_b["node_id"] in globally_connected_nodes:
                diag["skipped_global_connection_lock"] += 1
                continue

            # Find closest unpaired A node
            best_a = None
            best_dist = float("inf")

            for node_a in nodes_a:
                if node_a["node_id"] in paired_a:
                    continue
                if node_a["node_id"] in globally_connected_nodes:
                    continue

                diag["pairing_attempts"] += 1
                dist = math.hypot(node_a["x"] - node_b["x"], node_a["y"] - node_b["y"])

                # Check distance constraints WITH COUNTERS
                if dist < min_dist_m:
                    diag["pairing_rejected_distance_too_short"] += 1
                    continue
                if dist > max_dist_m:
                    diag["pairing_rejected_distance_too_long"] += 1
                    continue

                # Check perpendicularity WITH COUNTERS
                cross_bearing = _bearing_from_xy(node_a["x"], node_a["y"], node_b["x"], node_b["y"])
                if cross_bearing is None:
                    diag["pairing_rejected_bearing_calc_failed"] += 1
                    continue

                angle_ok, _angle_delta = _is_crossing_angle_valid(cross_bearing, axis_bearing)
                if not angle_ok:
                    diag["pairing_rejected_angle"] += 1
                    continue

                if dist < best_dist:
                    best_dist = dist
                    best_a = node_a

            if best_a:
                # Valid real-to-real crossing
                node_a = best_a
                paired_a.add(node_a["node_id"])
                paired_b.add(node_b["node_id"])

                # Check spacing
                mid_x = (node_a["x"] + node_b["x"]) / 2
                mid_y = (node_a["y"] + node_b["y"]) / 2

                too_close_global = any(
                    math.hypot(mid_x - ox, mid_y - oy) < min_spacing_m
                    for ox, oy in all_crossing_midpoints
                )

                if too_close_global:
                    diag["rejected_spacing"] += 1
                    paired_a.discard(node_a["node_id"])
                    paired_b.discard(node_b["node_id"])
                    continue

                # Create crossing edge
                n1, n2 = node_a["node_id"], node_b["node_id"]
                edge_key = tuple(sorted([n1, n2], key=str))

                if walk_graph.has_edge(n1, n2) or edge_key in existing_crossings:
                    continue

                # Ensure both nodes exist in walk_graph with proper coordinates
                if not _ensure_node_in_walk_graph(walk_graph, n1, drive_graph):
                    continue
                if not _ensure_node_in_walk_graph(walk_graph, n2, drive_graph):
                    continue

                walk_graph.add_edge(
                    n1, n2,
                    length=best_dist,
                    travel_time=best_dist / 80.0,
                    synthetic_crossing=True,
                    is_safe_to_cross=True,
                    crossing_rule="drive_node_crossings",
                    road_name=road_key[0],
                    road_class=road_key[1],
                )
                existing_crossings.add(edge_key)
                all_crossing_midpoints.append((mid_x, mid_y))
                road_midpoints.append((mid_x, mid_y))
                globally_connected_nodes.add(node_a["node_id"])
                globally_connected_nodes.add(node_b["node_id"])

                mid_lat, mid_lon = _from_meters(mid_x, mid_y)
                crossings.append({
                    "lat": mid_lat,
                    "lon": mid_lon,
                    "lat_a": node_a["lat"],
                    "lon_a": node_a["lon"],
                    "lat_b": node_b["lat"],
                    "lon_b": node_b["lon"],
                    "length_m": round(best_dist, 1),
                    "road_name": road_key[0],
                    "road_class": road_key[1],
                    "crossing_type": "real_to_real",
                })

                debug_info["crossings"].append({
                    "lat_a": node_a["lat"],
                    "lon_a": node_a["lon"],
                    "lat_b": node_b["lat"],
                    "lon_b": node_b["lon"],
                    "type": "real_to_real",
                    "road_name": road_key[0],
                })

                diag["real_to_real_crossings"] += 1

        # Step 4: Create synthetic nodes for unpaired A nodes (ENHANCED with 3-stage fallback).
        for node_a in nodes_a:
            if node_a["node_id"] in paired_a:
                continue
            if node_a["node_id"] in globally_connected_nodes:
                diag["skipped_global_connection_lock"] += 1
                continue

            # Stage 1: Try projection with normal spacing
            target_xy = _fallback_target_on_opposite_side(node_a, segs_b)
            if target_xy is not None:
                synth_x, synth_y = target_xy
                dist = math.hypot(node_a["x"] - synth_x, node_a["y"] - synth_y)
                mid_x, mid_y = (node_a["x"] + synth_x) / 2, (node_a["y"] + synth_y) / 2

                too_close = any(
                    math.hypot(mid_x - ox, mid_y - oy) < min_spacing_m
                    for ox, oy in all_crossing_midpoints
                )

                if not too_close:
                    # SUCCESS: Create crossing with projection
                    success = _create_fallback_crossing(
                        node_a, synth_x, synth_y, "real_to_synthetic_projected",
                        globally_connected_nodes, existing_crossings,
                        all_crossing_midpoints, walk_graph, diag, debug_info,
                        road_key, synthetic_counter, _from_meters, _create_synthetic_node,
                        _find_existing_node_near_xy
                    )
                    if success:
                        diag["fallback_projected_with_spacing"] += 1
                        road_midpoints.append((mid_x, mid_y))
                        continue

            # Stage 2: If enabled, retry projection WITHOUT spacing constraint
            if enable_guaranteed_connection and target_xy is not None:
                synth_x, synth_y = target_xy
                mid_x, mid_y = (node_a["x"] + synth_x) / 2, (node_a["y"] + synth_y) / 2
                success = _create_fallback_crossing(
                    node_a, synth_x, synth_y, "real_to_synthetic_projected_no_spacing",
                    globally_connected_nodes, existing_crossings,
                    all_crossing_midpoints, walk_graph, diag, debug_info,
                    road_key, synthetic_counter, _from_meters, _create_synthetic_node,
                    _find_existing_node_near_xy
                )
                if success:
                    diag["fallback_projected_no_spacing"] += 1
                    road_midpoints.append((mid_x, mid_y))
                    continue

            # Stage 3: GUARANTEED fallback - place at exactly 6m perpendicular
            if enable_guaranteed_connection:
                # Calculate perpendicular direction (90° from road axis)
                perp_angle = (axis_bearing + 90.0) % 360.0
                perp_rad = math.radians(perp_angle)
                guaranteed_dist = float(guaranteed_connection_distance_m)  # 6.0m

                synth_x = node_a["x"] + guaranteed_dist * math.sin(perp_rad)
                synth_y = node_a["y"] + guaranteed_dist * math.cos(perp_rad)
                mid_x, mid_y = (node_a["x"] + synth_x) / 2, (node_a["y"] + synth_y) / 2

                success = _create_fallback_crossing(
                    node_a, synth_x, synth_y, "real_to_synthetic_guaranteed",
                    globally_connected_nodes, existing_crossings,
                    all_crossing_midpoints, walk_graph, diag, debug_info,
                    road_key, synthetic_counter, _from_meters, _create_synthetic_node,
                    _find_existing_node_near_xy
                )
                if success:
                    diag["fallback_guaranteed"] += 1
                    road_midpoints.append((mid_x, mid_y))
                    continue

            # No connection created (only if guaranteed connection disabled)
            diag["fallback_failed_completely"] += 1

        # Step 5: Create synthetic nodes for unpaired B nodes (ENHANCED with 3-stage fallback).
        for node_b in nodes_b:
            if node_b["node_id"] in paired_b:
                continue
            if node_b["node_id"] in globally_connected_nodes:
                diag["skipped_global_connection_lock"] += 1
                continue

            # Stage 1: Try projection with normal spacing
            target_xy = _fallback_target_on_opposite_side(node_b, segs_a)
            if target_xy is not None:
                synth_x, synth_y = target_xy
                dist = math.hypot(node_b["x"] - synth_x, node_b["y"] - synth_y)
                mid_x, mid_y = (synth_x + node_b["x"]) / 2, (synth_y + node_b["y"]) / 2

                too_close = any(
                    math.hypot(mid_x - ox, mid_y - oy) < min_spacing_m
                    for ox, oy in all_crossing_midpoints
                )

                if not too_close:
                    # SUCCESS: Create crossing with projection
                    success = _create_fallback_crossing(
                        node_b, synth_x, synth_y, "real_to_synthetic_projected",
                        globally_connected_nodes, existing_crossings,
                        all_crossing_midpoints, walk_graph, diag, debug_info,
                        road_key, synthetic_counter, _from_meters, _create_synthetic_node,
                        _find_existing_node_near_xy
                    )
                    if success:
                        diag["fallback_projected_with_spacing"] += 1
                        road_midpoints.append((mid_x, mid_y))
                        continue

            # Stage 2: If enabled, retry projection WITHOUT spacing constraint
            if enable_guaranteed_connection and target_xy is not None:
                synth_x, synth_y = target_xy
                mid_x, mid_y = (synth_x + node_b["x"]) / 2, (synth_y + node_b["y"]) / 2
                success = _create_fallback_crossing(
                    node_b, synth_x, synth_y, "real_to_synthetic_projected_no_spacing",
                    globally_connected_nodes, existing_crossings,
                    all_crossing_midpoints, walk_graph, diag, debug_info,
                    road_key, synthetic_counter, _from_meters, _create_synthetic_node,
                    _find_existing_node_near_xy
                )
                if success:
                    diag["fallback_projected_no_spacing"] += 1
                    road_midpoints.append((mid_x, mid_y))
                    continue

            # Stage 3: GUARANTEED fallback - place at exactly 6m perpendicular (opposite direction)
            if enable_guaranteed_connection:
                # Calculate perpendicular direction (270° from road axis for B nodes)
                perp_angle = (axis_bearing - 90.0) % 360.0
                perp_rad = math.radians(perp_angle)
                guaranteed_dist = float(guaranteed_connection_distance_m)  # 6.0m

                synth_x = node_b["x"] + guaranteed_dist * math.sin(perp_rad)
                synth_y = node_b["y"] + guaranteed_dist * math.cos(perp_rad)
                mid_x, mid_y = (synth_x + node_b["x"]) / 2, (synth_y + node_b["y"]) / 2

                success = _create_fallback_crossing(
                    node_b, synth_x, synth_y, "real_to_synthetic_guaranteed",
                    globally_connected_nodes, existing_crossings,
                    all_crossing_midpoints, walk_graph, diag, debug_info,
                    road_key, synthetic_counter, _from_meters, _create_synthetic_node,
                    _find_existing_node_near_xy
                )
                if success:
                    diag["fallback_guaranteed"] += 1
                    road_midpoints.append((mid_x, mid_y))
                    continue

            # No connection created (only if guaranteed connection disabled)
            diag["fallback_failed_completely"] += 1

    _SYNTHETIC_DIAGNOSTICS = diag
    _store_debug_edges(debug_info["edges"], debug_info["drive_nodes"], debug_info["synthetic_nodes"])

    return crossings, diag, debug_info


def set_walk_graph(walk_graph, synthetic_cfg=None, drive_graph=None):
    """Set a dedicated walk graph and optionally add synthetic crossings."""
    global _WALK_GRAPH, _SYNTHETIC_CROSSINGS
    global _SYNTHETIC_CFG, _SYNTHETIC_TOTAL_ADDED, _SYNTHETIC_DRIVE_DONE
    global _WALK_SPATIAL_INDEX, _WALK_SPATIAL_INDEX_META
    global _WALK_TO_DRIVE_NODE_CACHE, _DRIVE_NODE_SIGNATURE_CACHE
    global _DRIVE_NODE_SAFE_CACHE
    global _DRIVE_NODE_FILTERED_SIG_CACHE
    global _SYNTHETIC_DIAGNOSTICS
    global _SYNTHETIC_REJECTED_UNSAFE
    global _SYNTHETIC_STAGE_CROSSING_POLICY
    _WALK_GRAPH = walk_graph
    _WALK_DIST_CACHE.clear()
    _WALK_NODE_MAP_CACHE.clear()
    _safe_nodes_cache.clear()  # Clear BFS cache when walk graph changes
    # Reset crossing BFS stats for fresh tracking
    global _CROSSING_BFS_STATS
    _CROSSING_BFS_STATS = {
        "students_checked": 0,
        "candidates_via_crossing": 0,
        "students_with_crossing_benefit": 0,
        "students_explored_crossing": 0,
        "allowed_students_checked": 0,
        "allowed_students_explored_crossing": 0,
    }
    _WALK_SPATIAL_INDEX = None
    _WALK_SPATIAL_INDEX_META = None
    _WALK_TO_DRIVE_NODE_CACHE = {}
    _DRIVE_NODE_SIGNATURE_CACHE = {}
    _DRIVE_NODE_SAFE_CACHE = {}
    _DRIVE_NODE_FILTERED_SIG_CACHE = {}
    _SYNTHETIC_CROSSINGS = []
    _SYNTHETIC_REJECTED_UNSAFE = []
    _SYNTHETIC_TOTAL_ADDED = 0
    _SYNTHETIC_DRIVE_DONE = set()
    _SYNTHETIC_DIAGNOSTICS = {
        "strategy": "none",
        "checked_drive_nodes": 0,
        "candidate_pairs_checked": 0,
        "rejected_existing_edge": 0,
        "rejected_distance": 0,
        "rejected_lane_pair": 0,
        "rejected_unsafe_road": 0,
        "rejected_missing_cross_bearing": 0,
        "rejected_not_perpendicular": 0,
        "rejected_existing_walk_short": 0,
        "added": 0,
    }
    cfg = synthetic_cfg or {}
    _SYNTHETIC_CFG = dict(cfg)
    stage_policy_cfg = (
        cfg.get("allowed_crossing_road_classes_by_stage")
        or cfg.get("stage_based_crossing_policy")
    )
    _SYNTHETIC_STAGE_CROSSING_POLICY = _normalize_stage_crossing_policy(stage_policy_cfg)
    strategy = str(cfg.get('strategy', 'per_drive_node')).lower()

    if _WALK_GRAPH is not None and cfg.get("enabled") and strategy == 'drive_node_crossings':
        crossings, diag, debug_info = build_drive_node_crossings(
            _WALK_GRAPH,
            drive_graph=drive_graph,
            min_dist_m=float(cfg.get("min_dist_m", 6.0)),
            max_dist_m=float(cfg.get("max_dist_m", 60.0)),
            min_opposite_bearing_deg=float(cfg.get("min_opposite_bearing_deg", 150.0)),
            max_crossing_angle_delta_deg=float(cfg.get("max_crossing_angle_delta_deg", 30.0)),
            min_perpendicular_deg=(
                float(cfg["min_perpendicular_deg"])
                if cfg.get("min_perpendicular_deg") is not None
                else None
            ),
            max_perpendicular_deg=(
                float(cfg["max_perpendicular_deg"])
                if cfg.get("max_perpendicular_deg") is not None
                else None
            ),
            min_spacing_m=float(cfg.get("min_spacing_m", 100.0)),
            min_spacing_per_road_m=float(cfg.get("min_spacing_per_road_m", 100.0)),
            center_lat=cfg.get("center_lat"),
            center_lon=cfg.get("center_lon"),
            radius_km=cfg.get("radius_km"),
            enable_guaranteed_connection=bool(cfg.get("enable_guaranteed_connection", True)),
            guaranteed_connection_distance_m=float(cfg.get("guaranteed_connection_distance_m", 6.0)),
            fallback_mode=str(cfg.get("fallback_mode", "adaptive")),
        )
        _SYNTHETIC_CROSSINGS = crossings
        _SYNTHETIC_TOTAL_ADDED = len(crossings)
        # Update diagnostics from returned dict
        _SYNTHETIC_DIAGNOSTICS.update(diag)
    elif _WALK_GRAPH is not None and cfg.get("enabled"):
        _SYNTHETIC_DIAGNOSTICS["strategy"] = "none"
    return _SYNTHETIC_CROSSINGS


def get_synthetic_crossings():
    """Return the last synthetic crossings list."""
    return list(_SYNTHETIC_CROSSINGS)


def get_synthetic_diagnostics():
    """Return counters that explain synthetic-crossing generation decisions."""
    return dict(_SYNTHETIC_DIAGNOSTICS)


def get_synthetic_rejected_unsafe():
    """Return rejected synthetic candidates blocked due to unsafe-road filtering."""
    return list(_SYNTHETIC_REJECTED_UNSAFE)


def haversine_walk_distance(lat1, lon1, lat2, lon2):
    """Calculate straight-line distance in meters (for quick estimates only)."""
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def walk_distance_on_roads(graph, node_a, node_b):
    """Calculate walking distance along roads (undirected) between two nodes.
    Ignores one-way restrictions and U-turn rules since pedestrians use sidewalks.
    Results are cached for fast repeated lookups during ALNS.
    
    Returns:
        float: distance in meters, or float('inf') if no path exists
    """
    if node_a == node_b:
        return 0.0
    walk_g = _get_walk_graph(graph)
    mapped_a = _map_to_walk_node(node_a, graph, walk_g, generate_synthetic=False)
    # Generate synthetic links only around stop/candidate-stop side (node_b).
    mapped_b = _map_to_walk_node(node_b, graph, walk_g, generate_synthetic=True)
    if mapped_a is None or mapped_b is None:
        return float('inf')
    cache_key = (mapped_a, mapped_b, id(walk_g))
    if cache_key in _WALK_DIST_CACHE:
        return _WALK_DIST_CACHE[cache_key]
    try:
        dist = nx.shortest_path_length(walk_g, mapped_a, mapped_b, weight='length')
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        dist = float('inf')
    _WALK_DIST_CACHE[cache_key] = dist
    _WALK_DIST_CACHE[(mapped_b, mapped_a, id(walk_g))] = dist  # Symmetric
    return dist


def walk_path_on_roads(graph, node_a, node_b):
    """Find the walking path along roads (undirected) between two nodes.
    Returns the list of node IDs along the path.
    
    Returns:
        list: path node IDs, or empty list if no path exists
    """
    if node_a == node_b:
        return [node_a]
    walk_g = _get_walk_graph(graph)
    mapped_a = _map_to_walk_node(node_a, graph, walk_g, generate_synthetic=False)
    mapped_b = _map_to_walk_node(node_b, graph, walk_g, generate_synthetic=True)
    if mapped_a is None or mapped_b is None:
        return []
    try:
        return nx.shortest_path(walk_g, mapped_a, mapped_b, weight='length')
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def get_walk_absolute_max(walk_radius):
    """Get the absolute maximum walk distance based on student stage.
    - ELEMENTARY/KG (0m recommended): 150m emergency max
    - MIDDLE (100m recommended): 300m absolute max
    - HIGH (200m recommended): 500m absolute max
    """
    if walk_radius == 0:
        return 500  # Door-to-door: allow finding nodes within 500m for isolated homes
    return min(walk_radius * 3, 500)


def calculate_walk_penalty(student, stop_node, graph):
    """Calculate a soft penalty for walking beyond the recommended radius.
    Uses straight-line (haversine) distance for fast O(1) evaluation during ALNS.
    Road-network walking paths are only computed for visualization (see walk_path_on_roads).
    
    Returns:
        (penalty_minutes, actual_walk_m, is_over_limit)
        - penalty_minutes: float, time penalty to add to objective
        - actual_walk_m: float, straight-line walking distance in meters
        - is_over_limit: bool, True if walk exceeds recommended radius
    """
    student_lat, student_lon = student.coords
    
    if graph is None:
        # In graph-free benchmark mode, we assume the student is already at the stop
        # or we don't have node coordinates to calculate soft penalties.
        return 0.0, 0.0, False

    try:
        stop_lat = graph.nodes[stop_node]['y']
        stop_lon = graph.nodes[stop_node]['x']
    except (KeyError, TypeError, AttributeError):
        return 0.0, 0.0, False
    
    # Haversine: O(1) math, called thousands of times during ALNS
    actual_walk_m = haversine_walk_distance(student_lat, student_lon, stop_lat, stop_lon)
    
    recommended_radius = student.walk_radius
    absolute_max = get_walk_absolute_max(recommended_radius)
    
    # Within recommended radius - no penalty
    if actual_walk_m <= recommended_radius:
        return 0.0, actual_walk_m, False
    
    # Beyond absolute maximum - reject
    if actual_walk_m > absolute_max:
        return float('inf'), actual_walk_m, True
    
    # Between recommended and absolute max - escalating penalty
    excess_m = actual_walk_m - recommended_radius
    if recommended_radius > 0:
        ratio = excess_m / recommended_radius  # 0.0 to 2.0
    else:
        ratio = excess_m / 50.0  # Normalize against 50m baseline for elementary
    
    # 2 minutes penalty per ratio unit
    penalty_minutes = ratio * 2.0
    return penalty_minutes, actual_walk_m, True


def get_turn_penalty(bearing1, bearing2):
    """Calculate a time penalty based on the difference between two edge bearings.
    
    Args:
        bearing1: Bearing of the incoming edge
        bearing2: Bearing of the outgoing edge
        
    Returns:
        float: Penalty in minutes (Massive for U-turns to make them illegal)
    """
    if bearing1 is None or bearing2 is None:
        return 0
        
    angle_diff = abs(bearing1 - bearing2)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
        
    # U-turn penalty (very sharp angles) - SET TO MASSIVE TO MAKE ILLEGAL
    if angle_diff > 140:
        return 9999.0
    
    # Normal turn penalty (90 degrees approx)
    if angle_diff > 45:
        return 0.2
        
    return 0.0


def calculate_weighted_path_time(graph, path_nodes):
    """Calculate total travel time including turn penalties for a given path.
    """
    if not path_nodes or len(path_nodes) < 2:
        return 0.0
        
    total_time = 0.0
    prev_bearing = None
    
    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i+1]
        
        # Get edge data (using first available key)
        edge_data = graph.get_edge_data(u, v)
        if not edge_data:
            continue
            
        # Handle Multigraph
        data = edge_data[0] if 0 in edge_data else list(edge_data.values())[0]
        
        # Add travel time
        total_time += data.get('travel_time', 0)
        
        # Add turn penalty if we have a previous edge
        current_bearing = data.get('bearing')
        if prev_bearing is not None:
            total_time += get_turn_penalty(prev_bearing, current_bearing)
            
        prev_bearing = current_bearing
        
    return total_time


# Global cache for shortest paths to speed up iterations
_path_cache = {}
_MATRIX_CACHE = {}       # (source, target) -> travel_time in minutes
_MATRIX_CACHE_LENGTH = {} # (source, target) -> length in meters
_DIJKSTRA_DONE = set()   # source nodes we've already run single-source Dijkstra on

# Graph-free mode: pre-registered (lat, lon) -> (node_id, (lat, lon)) mappings.
# When set, snap_address_to_edge returns immediately without touching OSMnx.
_SNAP_OVERRIDE = {}  # (lat, lon) -> (node_id, (lat, lon))

# Persistent coordinate snap cache: (lat, lon) -> (vnode_id, (lat, lon))
# Survives across ALNS runs with same input so ox.nearest_edges is called only once per location.
_COORD_SNAP_CACHE = {}

# When True, snap_address_to_edge uses ox.nearest_nodes (fast, ~1ms) instead of
# ox.nearest_edges + edge splitting (~4s on large graphs).  Set this before
# running experiments on large city graphs.  Results are slightly less precise
# (snaps to road intersection rather than road edge) but comparisons remain valid.
_FAST_SNAP_MODE = False

# Pre-built BallTree for fast nearest-node lookup.  Built once per graph to
# avoid OSMnx rebuilding it (+ converting 609K nodes to GeoDataFrame) on every call.
_BALL_TREE = None
_BALL_TREE_GRAPH_ID = None   # id(G) so we detect graph replacement
_BALL_TREE_NODE_IDS = None


def _get_or_build_ball_tree(G):
    """Return a pre-built (BallTree, node_id_list) for G, building once if needed."""
    global _BALL_TREE, _BALL_TREE_GRAPH_ID, _BALL_TREE_NODE_IDS
    g_id = id(G)
    if _BALL_TREE is not None and _BALL_TREE_GRAPH_ID == g_id:
        return _BALL_TREE, _BALL_TREE_NODE_IDS
    from sklearn.neighbors import BallTree
    import numpy as np
    node_ids = list(G.nodes())
    coords_rad = np.deg2rad([[G.nodes[n]['y'], G.nodes[n]['x']] for n in node_ids])
    _BALL_TREE = BallTree(coords_rad, metric='haversine')
    _BALL_TREE_GRAPH_ID = g_id
    _BALL_TREE_NODE_IDS = node_ids
    print(f'  [snap] BallTree built for {len(node_ids):,} nodes')
    return _BALL_TREE, _BALL_TREE_NODE_IDS


def fast_nearest_node(graph, lon, lat):
    """Find nearest graph node.  Uses cached BallTree if available (< 0.001s)
    else falls back to ox.nearest_nodes (~1.4s on 600K-node graphs)."""
    if _BALL_TREE is not None and _BALL_TREE_GRAPH_ID == id(graph):
        import numpy as np
        pt_rad = np.deg2rad([[lat, lon]])
        _, pos = _BALL_TREE.query(pt_rad, k=1)
        return _BALL_TREE_NODE_IDS[pos[0][0]]
    return ox.nearest_nodes(graph, lon, lat)


def get_heuristic_time(u, v, graph, max_speed_kmh=80):
    """Admissible heuristic for time-based A*: straight line distance / max speed."""
    node_u = graph.nodes[u]
    node_v = graph.nodes[v]
    # Haversine distance in meters
    lat1, lon1 = node_u['y'], node_u['x']
    lat2, lon2 = node_v['y'], node_v['x']
    
    # Simple Haversine approx
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lon2 - lon1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    dist_m = 2 * 6371000 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Max speed in meters per minute (80km/h conservative max)
    max_meters_per_min = (max_speed_kmh * 1000) / 60
    return dist_m / max_meters_per_min


def _reconstruct_path(came_from, source, target_state):
    """Reconstruct path from predecessor map."""
    path = []
    state = target_state
    while state is not None:
        node = state[0]
        path.append(node)
        state = came_from.get(state)
    path.reverse()
    return path


def find_shortest_path_with_turns(graph, source, target, weight='travel_time', initial_bearing=None):
    """Find the shortest path while making U-turns effectively illegal.
    
    Uses a custom A* search where the state is (node, incoming_bearing).
    This prevents 180-degree turns and applies minor penalties for 90-degree turns.
    Uses a predecessor map instead of storing full paths on the heap for speed.
    """
    if source == target:
        return [source], 0.0

    # Cache key: (source, target, rounded_bearing)
    cache_key = (source, target, round(initial_bearing, 1) if initial_bearing is not None else None)
    
    # Check _path_cache FIRST (has both path AND time — needed by path-dependent functions)
    if cache_key in _path_cache:
        return _path_cache[cache_key]

    # Fallback: _MATRIX_CACHE has time only (no path). Used for time-only lookups.
    if initial_bearing is None and (source, target) in _MATRIX_CACHE:
        return None, _MATRIX_CACHE[(source, target)]

    distances = {}
    came_from = {}  # state -> parent_state (predecessor map)
    
    initial_brg_key = round(initial_bearing, 2) if initial_bearing is not None else None
    start_state = (source, initial_brg_key)
    
    # Counter for stable heap ordering (avoids comparing tuples with None)
    counter = 0
    
    # (estimated_total_time, counter, actual_time, current_node, prev_bearing_key)
    h = get_heuristic_time(source, target, graph)
    pq = [(h, counter, 0.0, source, initial_bearing)]
    came_from[start_state] = None
    distances[start_state] = 0.0
    
    while pq:
        (est_total, _, current_time, current_node, prev_bearing) = heapq.heappop(pq)
        
        state = (current_node, round(prev_bearing, 2) if prev_bearing is not None else None)
        
        if current_node == target:
            # Reconstruct path from predecessor map
            path = _reconstruct_path(came_from, source, state)
            # Store in both caches
            _path_cache[cache_key] = (path, current_time)
            if initial_bearing is None:
                _MATRIX_CACHE[(source, target)] = current_time
            return path, current_time
            
        if state in distances and distances[state] < current_time:
            continue
        
        if current_node not in graph:
            continue
            
        for neighbor in graph.successors(current_node):
            edge_data_dict = graph.get_edge_data(current_node, neighbor)
            for key, data in edge_data_dict.items():
                cost = data.get(weight, 0)
                current_bearing = data.get('bearing')
                
                penalty = get_turn_penalty(prev_bearing, current_bearing)
                if penalty > 1000:
                    continue
                    
                new_time = current_time + cost + penalty
                bearing_key = round(current_bearing, 2) if current_bearing is not None else None
                new_state = (neighbor, bearing_key)
                
                if new_state not in distances or distances[new_state] > new_time:
                    distances[new_state] = new_time
                    came_from[new_state] = state
                    h_neighbor = get_heuristic_time(neighbor, target, graph)
                    counter += 1
                    heapq.heappush(pq, (new_time + h_neighbor, counter, new_time, neighbor, current_bearing))
    
    # CACHE NEGATIVE RESULTS — prevents re-running expensive exhaustive searches
    _path_cache[cache_key] = (None, float('inf'))
    if initial_bearing is None:
        _MATRIX_CACHE[(source, target)] = float('inf')
    return None, float('inf')


def precalculate_distance_matrix(graph, critical_node_ids, fast_mode=False):
    """Pre-calculate path times AND distances between all critical nodes.

    fast_mode=False (default): uses A* with 180-turn illegal logic (accurate,
      but slow on large graphs like Cairo with 600K+ nodes).
    fast_mode=True: uses NetworkX single-source Dijkstra (no turn penalties,
      ~50-100x faster on large graphs – suitable for experiments where
      relative comparison matters more than absolute accuracy).

    Fills _MATRIX_CACHE (travel_time in minutes) and _MATRIX_CACHE_LENGTH (meters).
    """
    node_list = list(critical_node_ids)
    total = len(node_list) * (len(node_list) - 1)
    print(f"Pre-calculating distance matrix for {len(node_list)} nodes ({total} pairs)... [{'fast' if fast_mode else 'accurate'}]")
    
    start_time = time.time()

    if fast_mode:
        # Batch single-source Dijkstra: one run per source covers all targets.
        # cutoff=60min limits exploration – Cairo routes are < 50 min in practice.
        #
        # Optimisation: track which sources have already had Dijkstra run via
        # _DIJKSTRA_DONE.  When the node list grows between calls (e.g. Mode A
        # → Mode B adds ~55 new candidate nodes), we avoid re-running Dijkstra
        # from all 700+ old sources.  Instead:
        #   • Forward Dijkstra from NEW sources  → fills (new → all)
        #   • Reverse Dijkstra from NEW targets  → fills (old → new)
        # This turns 791 Dijkstra calls into ~110, saving ~300 s.
        CUTOFF_MINUTES = 60.0
        node_set  = set(node_list)
        new_nodes = [n for n in node_list if n not in _DIJKSTRA_DONE]
        old_nodes = [n for n in node_list if n in _DIJKSTRA_DONE]

        # --- forward Dijkstra from new sources only ---
        for src in new_nodes:
            try:
                dist_time = nx.single_source_dijkstra_path_length(
                    graph, src, weight='travel_time', cutoff=CUTOFF_MINUTES)
            except Exception:
                dist_time = {}
            for tgt in node_list:
                if tgt != src and (src, tgt) not in _MATRIX_CACHE:
                    t = dist_time.get(tgt, float('inf'))
                    _MATRIX_CACHE[(src, tgt)] = t
                    _MATRIX_CACHE_LENGTH[(src, tgt)] = t * 500.0 if t < float('inf') else float('inf')
            _DIJKSTRA_DONE.add(src)

        # --- reverse Dijkstra: fill (old_src → new_tgt) pairs ---
        # Running Dijkstra on the reversed graph from a new target T gives
        # dist_rev(T, S) = dist(S → T) in the original graph.
        if old_nodes and new_nodes:
            # Identify which new nodes are actually needed as targets for old sources
            new_tgts = [t for t in new_nodes
                        if any((s, t) not in _MATRIX_CACHE for s in old_nodes)]
            if new_tgts:
                G_rev = graph.reverse(copy=False)   # view – O(1) memory
                for tgt in new_tgts:
                    try:
                        dist_rev = nx.single_source_dijkstra_path_length(
                            G_rev, tgt, weight='travel_time', cutoff=CUTOFF_MINUTES)
                    except Exception:
                        dist_rev = {}
                    for src in old_nodes:
                        if (src, tgt) not in _MATRIX_CACHE:
                            t = dist_rev.get(src, float('inf'))
                            _MATRIX_CACHE[(src, tgt)] = t
                            _MATRIX_CACHE_LENGTH[(src, tgt)] = t * 500.0 if t < float('inf') else float('inf')

        elapsed = time.time() - start_time
        fwd_cnt = len(new_nodes)
        rev_cnt = len(new_nodes) if old_nodes and new_nodes else 0
        skip_note = (f", {len(old_nodes)}/{len(node_list)} sources skipped "
                     f"(already cached), {fwd_cnt} fwd + {rev_cnt} rev Dijkstra") if old_nodes else ""
        print(f"Pre-calculation complete (fast). Matrix entries: {len(_MATRIX_CACHE)}, "
              f"Length entries: {len(_MATRIX_CACHE_LENGTH)}, Time: {elapsed:.1f}s{skip_note}")
        return

    # ---- accurate A* mode (original behaviour) ----
    count = 0
    for start_node in node_list:
        for end_node in node_list:
            if start_node == end_node:
                continue
            
            # This fills _path_cache and _MATRIX_CACHE with time data
            path, t = find_shortest_path_with_turns(graph, start_node, end_node)
            
            # Also compute length from the path to fill _MATRIX_CACHE_LENGTH
            if path and t < float('inf'):
                dist_m = 0.0
                for pi in range(len(path) - 1):
                    ed = graph.get_edge_data(path[pi], path[pi+1])
                    if ed:
                        d = ed[0] if 0 in ed else list(ed.values())[0]
                        dist_m += d.get('length', 0)
                _MATRIX_CACHE_LENGTH[(start_node, end_node)] = dist_m
            else:
                _MATRIX_CACHE_LENGTH[(start_node, end_node)] = float('inf')
            
            count += 1
            if count % 200 == 0:
                elapsed = time.time() - start_time
                print(f"  Processed {count}/{total} pairs... ({elapsed:.1f}s)")
    
    print(f"Pre-calculation complete. Matrix entries: {len(_MATRIX_CACHE)}, Length entries: {len(_MATRIX_CACHE_LENGTH)}")


def shortest_path_length_with_turns(graph, source, target, weight='travel_time', initial_bearing=None):
    """Fast lookup from Matrix Cache if available, else run A*."""
    if initial_bearing is None:
        if weight == 'length' and (source, target) in _MATRIX_CACHE_LENGTH:
            return _MATRIX_CACHE_LENGTH[(source, target)]
        if weight == 'travel_time' and (source, target) in _MATRIX_CACHE:
            return _MATRIX_CACHE[(source, target)]
        
    _, t = find_shortest_path_with_turns(graph, source, target, weight=weight, initial_bearing=initial_bearing)
    
    # OPTIMIZATION: Write-through cache on miss to avoid re-calculating same paths repeatedly
    if initial_bearing is None and t < float('inf'):
        if weight == 'travel_time':
            _MATRIX_CACHE[(source, target)] = t
        elif weight == 'length':
            _MATRIX_CACHE_LENGTH[(source, target)] = t
            
    return t


def calculate_route_time_from_matrix(stops, graph=None):
    """Ultra-fast route time using O(1) matrix lookups.
    On cache miss: lazily computes the pair via A* and caches it.
    No need for upfront all-pairs precomputation.
    """
    if len(stops) < 2:
        return 0.0
    total = 0.0
    for i in range(len(stops) - 1):
        pair = (stops[i].node_id, stops[i+1].node_id)
        if pair in _MATRIX_CACHE:
            t = _MATRIX_CACHE[pair]
            if t == float('inf'):
                print(f"[DEBUG] Pair {pair} in cache but value is inf")
                return 9999.0
            total += t
        elif graph is not None:
            # Lazy compute: run A* once, result is cached for future lookups
            path, t = find_shortest_path_with_turns(graph, pair[0], pair[1])
            if t == float('inf'):
                print(f"[DEBUG] A* for pair {pair} returned inf (no path found)")
                return 9999.0
            # Also compute length while we have the path
            if path:
                dist_m = 0.0
                for pi in range(len(path) - 1):
                    ed = graph.get_edge_data(path[pi], path[pi+1])
                    if ed:
                        d = ed[0] if 0 in ed else list(ed.values())[0]
                        dist_m += d.get('length', 0)
                _MATRIX_CACHE_LENGTH[pair] = dist_m
            total += t
        else:
            print(f"[DEBUG] calculate_route_time_from_matrix: No graph provided for pair {pair}, returning None")
            return None  # No graph provided, can't compute
    return total


def calculate_route_distance_from_matrix(stops, graph=None):
    """Ultra-fast route distance using O(1) matrix lookups.
    On cache miss: lazily computes the pair via A* and caches it.
    Returns distance in kilometers.
    """
    if len(stops) < 2:
        return 0.0
    total_m = 0.0
    for i in range(len(stops) - 1):
        pair = (stops[i].node_id, stops[i+1].node_id)
        if pair in _MATRIX_CACHE_LENGTH:
            d = _MATRIX_CACHE_LENGTH[pair]
            if d == float('inf'):
                return 0.0
            total_m += d
        elif graph is not None:
            # Lazy compute: run A* to get path, compute length from it
            path, t = find_shortest_path_with_turns(graph, pair[0], pair[1])
            if path and t < float('inf'):
                dist_m = 0.0
                for pi in range(len(path) - 1):
                    ed = graph.get_edge_data(path[pi], path[pi+1])
                    if ed:
                        dd = ed[0] if 0 in ed else list(ed.values())[0]
                        dist_m += dd.get('length', 0)
                _MATRIX_CACHE_LENGTH[pair] = dist_m
                total_m += dist_m
            else:
                _MATRIX_CACHE_LENGTH[pair] = float('inf')
                return 0.0
        else:
            return None  # No graph provided
    return total_m / 1000.0


def get_bearing_of_path(graph, path):
    """Get the bearing of the last edge in a path."""
    if not path or len(path) < 2:
        return None
    u, v = path[-2], path[-1]
    edge_data = graph.get_edge_data(u, v)
    if not edge_data:
        return None
    data = edge_data[0] if 0 in edge_data else list(edge_data.values())[0]
    return data.get('bearing')


from entities import Stop, Route, Student


# ============================================================================
# GEOCODING & NODE MAPPING: Convert addresses to graph nodes
# ============================================================================

def snap_address_to_edge(coords, graph):
    """Snap student coordinates to the exact point on the nearest road edge by splitting it.
    
    This creates a virtual node in the graph at the projected point, effectively
    forcing the routing algorithm to pass right in front of the house.
    
    Graph-free shortcut: if coords is present in _SNAP_OVERRIDE the pre-registered
    (node_id, (lat, lon)) pair is returned immediately without any OSMnx call.
    This enables benchmark / Euclidean-matrix mode where all nodes are stop indices.
    """
    if coords in _SNAP_OVERRIDE:
        return _SNAP_OVERRIDE[coords]

    # Persistent coord snap cache — avoids repeating ox.nearest_edges across ALNS runs
    if coords in _COORD_SNAP_CACHE:
        return _COORD_SNAP_CACHE[coords]

    lat, lon = coords
    # Use a unique but consistent INT ID for the virtual node for OSMnx compatibility
    lat_key = int(abs(lat) * 1000000)
    lon_key = int(abs(lon) * 1000000)
    vnode_id = int(f"999{lat_key}{lon_key}")
    
    # If the virtual node already exists, return it (added in a previous run)
    if vnode_id in graph:
        result = (vnode_id, (graph.nodes[vnode_id]['y'], graph.nodes[vnode_id]['x']))
        _COORD_SNAP_CACHE[coords] = result
        return result

    # Fast-snap mode: use pre-built BallTree (no GeoDataFrame conversion per call)
    if _FAST_SNAP_MODE:
        import numpy as np
        tree, node_ids = _get_or_build_ball_tree(graph)
        point_rad = np.deg2rad([[lat, lon]])
        _, pos = tree.query(point_rad, k=1)
        nearest_id = node_ids[pos[0][0]]
        result = (nearest_id, (graph.nodes[nearest_id]['y'], graph.nodes[nearest_id]['x']))
        _COORD_SNAP_CACHE[coords] = result
        return result

    
    point_geom = Point(lon, lat)
    
    try:
        # Find nearest edge (u, v, key)
        u, v, k = ox.nearest_edges(graph, lon, lat)
        original_data = graph.get_edge_data(u, v, k)
        if original_data is None:
            raise ValueError(f"No edge data found for ({u}, {v}, {k})")
        
        edge_data = original_data.copy()
        
        # Get edge geometry for interpolation
        if 'geometry' in edge_data:
            line = edge_data['geometry']
        else:
            u_node = graph.nodes[u]
            v_node = graph.nodes[v]
            line = LineString([(u_node['x'], u_node['y']), (v_node['x'], v_node['y'])])
            
        # Project house onto edge and find exact intersection point
        projected_dist = line.project(point_geom)
        snapped_point = line.interpolate(projected_dist)
        snapped_coords = (snapped_point.y, snapped_point.x)
        
        # 1. Add the virtual node to the graph
        graph.add_node(vnode_id, x=snapped_coords[1], y=snapped_coords[0], street_count=2)
        
        # 2. Calculate the split ratio for attributes
        line_len = line.length if line.length > 0 else 1.0
        ratio = projected_dist / line_len
        # Clamp ratio to avoid very small edges
        ratio = max(0.01, min(0.99, ratio))
        
        total_len = edge_data.get('length', 1.0)
        
        # 3. Create two new edges by splitting the original
        data1 = edge_data.copy()
        data2 = edge_data.copy()
        
        data1['length'] = total_len * ratio
        data2['length'] = total_len * (1 - ratio)
        
        # Split geometry if it exists
        if 'geometry' in edge_data:
            line = edge_data['geometry']
            # Use shapely.ops.substring for precise splitting
            data1['geometry'] = substring(line, 0, projected_dist)
            data2['geometry'] = substring(line, projected_dist, line.length)
        
        # Approximate travel times
        if 'travel_time' in edge_data:
            data1['travel_time'] = edge_data['travel_time'] * ratio
            data2['travel_time'] = edge_data['travel_time'] * (1 - ratio)
            
        # Add the new edges and remove the original one
        graph.add_edge(u, vnode_id, **data1)
        graph.add_edge(vnode_id, v, **data2)
        
        if graph.has_edge(u, v, k):
            graph.remove_edge(u, v, k)
            
        # 4. Handle the reverse direction (if it exists)
        if graph.has_edge(v, u):
            rev_options = graph.get_edge_data(v, u)
            for rev_k, rev_data_orig in rev_options.items():
                rev_data = rev_data_orig.copy()
                rev_data1 = rev_data.copy()
                rev_data2 = rev_data.copy()
                rev_data1['length'] = rev_data.get('length', 1.0) * (1 - ratio)
                rev_data2['length'] = rev_data.get('length', 1.0) * ratio
                
                if 'travel_time' in rev_data:
                    rev_data1['travel_time'] = rev_data['travel_time'] * (1 - ratio)
                    rev_data2['travel_time'] = rev_data['travel_time'] * ratio

                if 'geometry' in rev_data:
                    rev_line = rev_data['geometry']
                    rev_len = rev_line.length
                    # Split at (rev_len - projected_dist) because rev edge starts at v
                    rev_data1['geometry'] = substring(rev_line, 0, rev_len - projected_dist)
                    rev_data2['geometry'] = substring(rev_line, rev_len - projected_dist, rev_len)
                    
                graph.add_edge(v, vnode_id, **rev_data1)
                graph.add_edge(vnode_id, u, **rev_data2)
                graph.remove_edge(v, u, rev_k)
                break
            
        result = (vnode_id, snapped_coords)
        _COORD_SNAP_CACHE[coords] = result
        return result
        
    except Exception as e:
        # Fallback to nearest node if edge splitting fails
        nearest_id = fast_nearest_node(graph, lon, lat)
        result = (nearest_id, (graph.nodes[nearest_id]['y'], graph.nodes[nearest_id]['x']))
        _COORD_SNAP_CACHE[coords] = result
        return result



# Road class rank for candidate-stop scoring (higher = more bus-useful)
_ROAD_CLASS_RANK = {
    "motorway"     : 6, "trunk"       : 5,
    "primary"      : 4, "secondary"   : 3,
    "tertiary"     : 2, "residential" : 1,
    "living_street": 1, "unclassified": 1,
}

def _candidate_points(graph, node) -> int:
    """Return 0, 1, or 2 — higher means node is a more useful bus-stop candidate.

    +1  if the node is a real intersection (degree >= 3 distinct streets)
    +1  if any adjacent edge is tertiary or higher road class

    Used by find_safe_nodes_within_radius to sort walk-radius candidates so
    intersection / arterial-adjacent nodes are preferred over mid-block
    residential nodes of similar distance.
    """
    pts = 0
    if graph.degree(node) >= 3:
        pts += 1
    best_rank = 0
    for _, _, data in graph.edges(node, data=True):
        hw = data.get("highway", "")
        if isinstance(hw, list):
            hw = hw[0]
        best_rank = max(best_rank, _ROAD_CLASS_RANK.get(hw, 0))
    if best_rank >= 2:   # tertiary or higher
        pts += 1
    return pts

# Cache for pedestrian-safe nodes (stores full BFS result; scoring/truncation applied at call time)
_safe_nodes_cache = {}

def find_safe_nodes_within_radius(coords, graph, radius_meters, walk_distance_limit,
                                   candidate_cfg=None, walk_graph=None,
                                   student_stage=None, student_disabled=False):
    """Find all nodes reachable by walking within *walk_distance_limit* metres.

    Walking semantics
    -----------------
    The BFS is **bidirectional** (pedestrians ignore one-way rules) and only
    traverses edges where ``is_safe_to_cross`` is True.

    When *walk_graph* is provided:
    - BFS is performed on the walk graph (which may include synthetic crossings)
    - Each reachable walk node is mapped to its nearest drive graph node
    - Returns drive graph nodes suitable for bus stops

    When *walk_graph* is None (legacy mode):
    - BFS is performed directly on the drive graph

    The ``is_safe_to_cross`` flag is set per-edge by ``setup_graph``:

    * **Constrained** graph → primary / trunk / secondary are False;
      tertiary / residential / living_street / default are True.
    * **Unconstrained** graph → ALL edges are True.

    This means:
    * On the constrained graph the BFS can walk along residential +
      tertiary streets but cannot cross primary / trunk / secondary.
    * On the unconstrained graph the BFS explores freely.

    Candidate scoring (when *candidate_cfg* is provided)
    ------------------------------------------------------
    The full BFS result is cached, then ranked and truncated:

    * The home snap node (distance 0) is **always** included first.
    * Remaining nodes are sorted by ``(-points, walk_dist)``:
        - 2-point nodes first  (intersection **and** tertiary+)
        - 1-point nodes next   (one criterion met)
        - 0-point nodes last   (mid-block residential)
        - ties broken by ascending walk distance
    * Truncated to ``candidate_cfg["max_candidates_per_student"]`` total.

    Returns a list of ``(node_id, distance_metres)`` tuples.
    """
    lat, lon = coords
    # Include walk_graph identity in cache key to avoid stale results
    walk_graph_id = id(walk_graph) if walk_graph is not None else 0
    stage_key = _resolve_stage_crossing_policy_key(student_stage, student_disabled)
    cache_key = (lat, lon, walk_distance_limit, walk_graph_id, stage_key, bool(student_disabled))
    if cache_key in _safe_nodes_cache:
        all_reachable = _safe_nodes_cache[cache_key]
        # Apply scoring/truncation on the cached full result if config given
        if candidate_cfg:
            return _rank_and_truncate_candidates(
                all_reachable, graph, fast_nearest_node(graph, lon, lat), candidate_cfg
            )
        return all_reachable

    # If walk_graph provided, do BFS on walk graph and map results to drive nodes.
    # Also union with direct drive-graph BFS so candidate coverage doesn't collapse
    # when walk->drive projection is overly coarse in sparse/simplified areas.
    if walk_graph is not None:
        safe_nodes_walk = _bfs_walk_graph_to_drive_nodes(
            coords,
            graph,
            walk_graph,
            walk_distance_limit,
            student_stage=student_stage,
            student_disabled=student_disabled,
        )
        safe_nodes_drive = _bfs_on_drive_graph(coords, graph, walk_distance_limit)

        merged = {}
        for nid, dist in safe_nodes_walk:
            d = float(dist)
            if d < merged.get(nid, float("inf")):
                merged[nid] = d
        for nid, dist in safe_nodes_drive:
            d = float(dist)
            if d < merged.get(nid, float("inf")):
                merged[nid] = d
        safe_nodes = [(nid, dist) for nid, dist in merged.items()]
    else:
        # Legacy mode: BFS directly on drive graph
        safe_nodes = _bfs_on_drive_graph(coords, graph, walk_distance_limit)

    _safe_nodes_cache[cache_key] = safe_nodes

    if candidate_cfg:
        home_node = fast_nearest_node(graph, lon, lat)
        return _rank_and_truncate_candidates(safe_nodes, graph, home_node, candidate_cfg)
    return safe_nodes


def _bfs_on_drive_graph(coords, graph, walk_distance_limit):
    """BFS on drive graph - legacy behavior."""
    lat, lon = coords
    start_node = fast_nearest_node(graph, lon, lat)

    safe_nodes = []
    visited = set()
    queue = [(start_node, 0)]  # (node, distance_so_far)

    while queue:
        current_node, dist_so_far = queue.pop(0)

        if current_node in visited or dist_so_far > walk_distance_limit:
            continue
        visited.add(current_node)

        safe_nodes.append((current_node, dist_so_far))

        # Walk along any edge marked safe (bidirectional)
        for neighbor in graph.successors(current_node):
            edge_data = graph[current_node][neighbor]
            is_safe = False
            edge_length = float('inf')
            for key, data in edge_data.items():
                if data.get('is_safe_to_cross', True):
                    is_safe = True
                    edge_length = min(edge_length, data.get('length', 0))
            if is_safe:
                new_dist = dist_so_far + edge_length
                if new_dist <= walk_distance_limit:
                    queue.append((neighbor, new_dist))

        # Also walk against traffic (pedestrians are bidirectional)
        for predecessor in graph.predecessors(current_node):
            edge_data = graph[predecessor][current_node]
            is_safe = False
            edge_length = float('inf')
            for key, data in edge_data.items():
                if data.get('is_safe_to_cross', True):
                    is_safe = True
                    edge_length = min(edge_length, data.get('length', 0))
            if is_safe:
                new_dist = dist_so_far + edge_length
                if new_dist <= walk_distance_limit:
                    queue.append((predecessor, new_dist))

    return safe_nodes


def _bfs_walk_graph_to_drive_nodes(
    coords,
    drive_graph,
    walk_graph,
    walk_distance_limit,
    student_stage=None,
    student_disabled=False,
):
    """BFS on walk graph, mapping reachable walk nodes to drive nodes.

    This enables students to use synthetic crossings (on walk graph) to reach
    bus stops (on drive graph) on the opposite side of dual carriageways.

    Also tracks debug metrics about synthetic crossing usage.
    """
    global _CROSSING_BFS_STATS
    lat, lon = coords
    crossing_allowed_for_student = bool(
        _allowed_synthetic_crossing_classes(student_stage, student_disabled)
    )

    # Find starting walk node - map from nearest drive node
    drive_start = fast_nearest_node(drive_graph, lon, lat)
    walk_start = _map_to_walk_node(drive_start, drive_graph, walk_graph)
    if walk_start is None:
        # Fallback: direct snap to walk graph (robust to synthetic string node IDs)
        walk_start = _nearest_node_any_id(walk_graph, lon, lat)
    if walk_start is None:
        return []

    # BFS on walk graph
    # Track: (walk_node, distance, crossed_synthetic)
    visited = {}  # walk_node -> (dist, crossed_synthetic)
    queue = [(walk_start, 0, False)]  # (node, distance, crossed_synthetic_to_get_here)

    # Track minimum distance to each drive node and whether it required crossing
    drive_node_min_dist = {}  # drive_node -> dist
    drive_node_via_crossing = set()  # drive nodes reached ONLY via synthetic crossing
    drive_node_without_crossing = set()  # drive nodes reachable without synthetic crossing
    explored_crossing = False

    while queue:
        current_walk_node, dist_so_far, crossed_synthetic = queue.pop(0)

        if dist_so_far > walk_distance_limit:
            continue

        # Check if already visited with a better or equal path
        if current_walk_node in visited:
            prev_dist, prev_crossed = visited[current_walk_node]
            # Skip if we've already found this node at same/shorter distance
            if dist_so_far >= prev_dist:
                continue
        visited[current_walk_node] = (dist_so_far, crossed_synthetic)

        # Map this walk node to a drive node (if not synthetic)
        node_data = walk_graph.nodes.get(current_walk_node, {})
        is_synthetic = node_data.get('synthetic', False) or (
            isinstance(current_walk_node, str) and current_walk_node.startswith('synth_')
        )

        if not is_synthetic:
            drive_node = _nearest_drive_node_for_walk_node(
                current_walk_node, walk_graph, drive_graph
            )
            if drive_node is not None and drive_node in drive_graph:
                # Keep minimum distance for each drive node
                if drive_node not in drive_node_min_dist:
                    drive_node_min_dist[drive_node] = dist_so_far
                else:
                    drive_node_min_dist[drive_node] = min(
                        drive_node_min_dist[drive_node], dist_so_far
                    )

                # Track crossing usage for this drive node
                if crossed_synthetic:
                    drive_node_via_crossing.add(drive_node)
                else:
                    drive_node_without_crossing.add(drive_node)

        def _iter_edge_variants(edge_data_dict):
            if isinstance(edge_data_dict, dict) and 'length' in edge_data_dict:
                yield edge_data_dict
                return
            if isinstance(edge_data_dict, dict):
                for data in edge_data_dict.values():
                    if isinstance(data, dict):
                        yield data

        def _edge_traversal_info(edge_data_dict):
            """Return (is_traversable, edge_length, crossed_synthetic)."""
            best_length = float('inf')
            crossed_synthetic = False
            found = False
            for data in _iter_edge_variants(edge_data_dict):
                if not data.get('is_safe_to_cross', True):
                    continue
                if not _edge_allows_student_crossing(
                    data,
                    student_stage=student_stage,
                    student_disabled=student_disabled,
                ):
                    continue
                edge_length = float(data.get('length', 0.0) or 0.0)
                is_crossing = bool(data.get('synthetic_crossing', False))
                if edge_length < best_length:
                    best_length = edge_length
                    crossed_synthetic = is_crossing
                found = True
            if not found:
                return False, float('inf'), False
            return True, best_length, crossed_synthetic

        # Walk along edges (bidirectional for pedestrians)
        # Handle both directed and undirected graphs
        is_directed = walk_graph.is_directed()
        neighbors_iter = walk_graph.successors(current_walk_node) if is_directed else walk_graph.neighbors(current_walk_node)
        for neighbor in neighbors_iter:
            edge_data = walk_graph[current_walk_node][neighbor]
            is_safe, edge_length, is_crossing_edge = _edge_traversal_info(edge_data)
            if is_safe:
                new_dist = dist_so_far + edge_length
                if new_dist <= walk_distance_limit:
                    new_crossed = crossed_synthetic or is_crossing_edge
                    if is_crossing_edge:
                        explored_crossing = True
                    queue.append((neighbor, new_dist, new_crossed))

        # Also check predecessors (for directed graphs only)
        if is_directed:
            for predecessor in walk_graph.predecessors(current_walk_node):
                edge_data = walk_graph[predecessor][current_walk_node]
                is_safe, edge_length, is_crossing_edge = _edge_traversal_info(edge_data)
                if is_safe:
                    new_dist = dist_so_far + edge_length
                    if new_dist <= walk_distance_limit:
                        new_crossed = crossed_synthetic or is_crossing_edge
                        if is_crossing_edge:
                            explored_crossing = True
                        queue.append((predecessor, new_dist, new_crossed))

    # Calculate crossing-only candidates (nodes reachable ONLY via crossing)
    crossing_only_nodes = drive_node_via_crossing - drive_node_without_crossing

    # Update global stats
    _CROSSING_BFS_STATS["students_checked"] += 1
    _CROSSING_BFS_STATS["candidates_via_crossing"] += len(crossing_only_nodes)
    if crossing_only_nodes:
        _CROSSING_BFS_STATS["students_with_crossing_benefit"] += 1
    if explored_crossing:
        _CROSSING_BFS_STATS["students_explored_crossing"] += 1
    if crossing_allowed_for_student:
        _CROSSING_BFS_STATS["allowed_students_checked"] += 1
        if explored_crossing:
            _CROSSING_BFS_STATS["allowed_students_explored_crossing"] += 1

    # Convert to list of (node, dist) tuples
    return [(node, dist) for node, dist in drive_node_min_dist.items()]


def get_crossing_bfs_stats():
    """Return debug statistics about synthetic crossing usage in BFS."""
    return dict(_CROSSING_BFS_STATS)


def reset_crossing_bfs_stats():
    """Reset the crossing BFS statistics."""
    global _CROSSING_BFS_STATS
    _CROSSING_BFS_STATS = {
        "students_checked": 0,
        "candidates_via_crossing": 0,
        "students_with_crossing_benefit": 0,
        "students_explored_crossing": 0,
        "allowed_students_checked": 0,
        "allowed_students_explored_crossing": 0,
    }


def _rank_and_truncate_candidates(all_nodes, graph, home_node, candidate_cfg):
    """Sort *all_nodes* by (-points, dist) and truncate to max_candidates_per_student.

    The home snap node is always inserted at position 0 regardless of score.
    All other nodes compete on (points desc, distance asc).
    """
    max_k = candidate_cfg.get("max_candidates_per_student", 15)

    home  = [(nid, d) for nid, d in all_nodes if nid == home_node]
    others = [(nid, d) for nid, d in all_nodes if nid != home_node]
    others.sort(key=lambda x: (-_candidate_points(graph, x[0]), x[1]))

    result = home + others
    return result[:max_k]


def _extract_synthetic_crossings_from_path(walk_graph, walk_path):
    """Extract synthetic crossing edges from a walk path.

    Args:
        walk_graph: Walk graph with synthetic crossing edges marked
        walk_path: List of node IDs from walk_path_on_roads (may be empty or single node)

    Returns:
        List of synthetic crossing edge tuples: [(u, v), ...] normalized as (min, max)
    """
    if not walk_path or len(walk_path) < 2:
        return []

    crossings = []
    for i in range(len(walk_path) - 1):
        u, v = walk_path[i], walk_path[i + 1]

        # Get edge data (handle both directed and undirected, multigraph keys)
        edge_data = walk_graph.get_edge_data(u, v)
        if edge_data is None:
            edge_data = walk_graph.get_edge_data(v, u)

        if edge_data is None:
            continue

        # Check if any key in this edge is synthetic
        is_synthetic = False
        if isinstance(edge_data, dict) and 'synthetic_crossing' in edge_data:
            is_synthetic = edge_data.get('synthetic_crossing', False)
        else:
            # MultiGraph: check all keys
            for key, data in edge_data.items() if isinstance(edge_data, dict) else []:
                if data.get('synthetic_crossing', False):
                    is_synthetic = True
                    break

        if is_synthetic:
            # Normalize edge as (min, max) to handle direction independence
            crossing_key = (min(u, v), max(u, v))
            if crossing_key not in crossings:
                crossings.append(crossing_key)

    return crossings


def get_crossing_usage_from_solution(solution, drive_graph, walk_graph):
    """Extract which synthetic crossings were actually used by students' walk paths.

    Args:
        solution: ServiceSolution object with routes and stops
        drive_graph: Drive graph (for path finding)
        walk_graph: Walk graph with synthetic crossings

    Returns:
        Dict: {crossing_key: {
            "students": [student_ids],
            "homes": [(lat, lon), ...],
            "count": n,
            "length_m": distance
        }, ...}
    """
    crossing_usage = {}  # (u, v) -> {students, homes, count}

    if not solution or not solution.routes:
        return crossing_usage

    for route in solution.routes:
        for stop in route.stops:
            # Skip school stops
            if getattr(stop, 'stop_type', None) == 'school':
                continue

            for student in getattr(stop, 'students', []):
                # Get student home node (map from coordinates)
                try:
                    student_home_node = fast_nearest_node(drive_graph, student.coords[1], student.coords[0])
                except Exception:
                    continue

                # Get walk path from home to stop
                try:
                    walk_path = walk_path_on_roads(drive_graph, student_home_node, stop.node_id)
                except Exception:
                    walk_path = []

                # Extract synthetic crossings used in this path
                crossings = _extract_synthetic_crossings_from_path(walk_graph, walk_path)

                for crossing_key in crossings:
                    if crossing_key not in crossing_usage:
                        crossing_usage[crossing_key] = {
                            "students": [],
                            "homes": [],
                        }

                    crossing_usage[crossing_key]["students"].append(student.id)
                    crossing_usage[crossing_key]["homes"].append((student.coords[0], student.coords[1]))

    # Deduplicate and add counts
    for crossing_key, data in crossing_usage.items():
        data["students"] = list(set(data["students"]))
        data["homes"] = list(set(data["homes"]))
        data["count"] = len(data["students"])

    return crossing_usage


def _get_crossing_geometry_and_midpoint(walk_graph, u, v):
    """Extract geometry and calculate midpoint for a crossing edge.

    Args:
        walk_graph: Walk graph containing the edge
        u, v: Edge nodes

    Returns:
        Dict with keys: "coords" (list of lat/lon), "lat", "lon", "length_m"
    """
    coords = []
    length_m = 0.0

    # Get edge data
    edge_data = walk_graph.get_edge_data(u, v)
    if edge_data is None:
        edge_data = walk_graph.get_edge_data(v, u)

    if edge_data is None:
        # Fallback: just use node coordinates
        try:
            lat_u, lon_u = walk_graph.nodes[u]['y'], walk_graph.nodes[u]['x']
            lat_v, lon_v = walk_graph.nodes[v]['y'], walk_graph.nodes[v]['x']
            coords = [(lat_u, lon_u), (lat_v, lon_v)]
            length_m = math.sqrt((lat_v - lat_u)**2 + (lon_v - lon_u)**2) * 111000  # rough meters conversion
        except Exception:
            pass
    else:
        # Extract geometry from edge
        if isinstance(edge_data, dict) and 'geometry' in edge_data:
            # Simple edge with geometry
            try:
                geom = edge_data.get('geometry')
                if geom:
                    coords = [(lat, lon) for lon, lat in geom.coords]
                length_m = edge_data.get('length', 0.0)
            except Exception:
                pass
        else:
            # MultiGraph: find best edge with geometry
            for key, data in edge_data.items() if isinstance(edge_data, dict) else []:
                if 'geometry' in data:
                    try:
                        geom = data.get('geometry')
                        if geom:
                            coords = [(lat, lon) for lon, lat in geom.coords]
                        length_m = data.get('length', 0.0)
                        break
                    except Exception:
                        pass

        # Fallback if no geometry found
        if not coords:
            try:
                lat_u, lon_u = walk_graph.nodes[u]['y'], walk_graph.nodes[u]['x']
                lat_v, lon_v = walk_graph.nodes[v]['y'], walk_graph.nodes[v]['x']
                coords = [(lat_u, lon_u), (lat_v, lon_v)]
                if not length_m:
                    length_m = math.sqrt((lat_v - lat_u)**2 + (lon_v - lon_u)**2) * 111000
            except Exception:
                pass

    # Calculate midpoint
    if coords:
        lat_mid = sum(c[0] for c in coords) / len(coords)
        lon_mid = sum(c[1] for c in coords) / len(coords)
    else:
        lat_mid = lon_mid = 0.0

    return {
        "coords": coords,
        "lat": lat_mid,
        "lon": lon_mid,
        "length_m": round(length_m, 1),
    }


# ============================================================================
# ROUTE ANALYSIS: Calculate distance, time, and safety for routes
# ============================================================================


def calculate_route_distance(route, graph):
    """Calculate total distance of a route in kilometers.
    
    Args:
        route: Route object with ordered stops
        graph: NetworkX road network
        
    Returns:
        float: Total distance in kilometers
    """
    if len(route.stops) < 2:
        return 0.0
    
    # Try fast matrix lookup first (lazy: computes on cache miss)
    fast = calculate_route_distance_from_matrix(route.stops, graph)
    if fast is not None:
        return fast
    
    total_distance_m = 0
    
    # Sum distances between consecutive stops
    for i in range(len(route.stops) - 1):
        from_node = route.stops[i].node_id
        to_node = route.stops[i + 1].node_id
        
        # Check length matrix
        if (from_node, to_node) in _MATRIX_CACHE_LENGTH:
            d = _MATRIX_CACHE_LENGTH[(from_node, to_node)]
            if d < float('inf'):
                total_distance_m += d
            continue
        
        try:
            # Use shortest path in terms of distance (turn-aware)
            dist_val = shortest_path_length_with_turns(
                graph, from_node, to_node, weight='length'
            )
            if dist_val < 1000000:
                total_distance_m += dist_val
        except Exception:
            print(f"Warning: No path between stops {from_node} and {to_node}")
            continue
    
    return total_distance_m / 1000  # Convert to km


def calculate_route_path_and_stats(graph, stops, weight='travel_time'):
    """Calculate the full path and travel time for a sequence of stops.
    
    This function ensures that turns BETWEEN segments (at the stops) are also
    penalized, preventing the bus from doing a 180 at a stop.
    
    Returns:
        tuple: (full_path_nodes, total_time)
    """
    if not stops:
        return [], 0.0
    if len(stops) == 1:
        return [stops[0].node_id], 0.0
        
    full_path = []
    total_time = 0.0
    last_bearing = None
    
    for i in range(len(stops) - 1):
        u_node = stops[i].node_id
        v_node = stops[i+1].node_id
        
        path_segment, segment_time = find_shortest_path_with_turns(
            graph, u_node, v_node, weight=weight, initial_bearing=last_bearing
        )
        
        if not path_segment:
            return None, float('inf')
            
        total_time += segment_time
        
        if not full_path:
            full_path.extend(path_segment)
        else:
            full_path.extend(path_segment[1:])
            
        last_bearing = get_bearing_of_path(graph, path_segment)
        
    return full_path, total_time


def calculate_route_time(route, graph):
    """Calculate total travel time of a route in minutes.
    """
    if len(route.stops) < 2:
        return 0.0
    
    _, total_time = calculate_route_path_and_stats(graph, route.stops)
    return total_time if total_time != float('inf') else 9999.0


def calculate_stops_time(stops, graph):
    """Purely functional travel time calculation for a sequence of Stop objects.
    Does not modify any objects.
    """
    if len(stops) < 2:
        return 0.0
    # Reuses the existing logic that handles turn penalties
    _, total_time = calculate_route_path_and_stats(graph, stops)
    return total_time if total_time != float('inf') else 9999.0


def calculate_student_ride_time(route, graph):
    """Calculate the travel time from the first student boarding to the end (school).
    """
    if len(route.stops) < 2:
        return 0.0
    
    # Find the index of the first stop that has students
    first_student_stop_idx = -1
    for i, stop in enumerate(route.stops):
        if stop.get_student_count() > 0:
            first_student_stop_idx = i
            break
            
    if first_student_stop_idx == -1 or first_student_stop_idx >= len(route.stops) - 1:
        return 0.0
        
    student_stops = route.stops[first_student_stop_idx:]
    _, ride_time = calculate_route_path_and_stats(graph, student_stops)
    return ride_time if ride_time != float('inf') else 9999.0


def calculate_student_ride_time_potential(route, new_stop, insert_position, graph):
    """Calculate what the student ride time WOULD be if a stop were inserted.
    
    Args:
        route: Route object
        new_stop: Stop object to potentially insert
        insert_position: Index where new_stop would be placed
        graph: NetworkX graph
        
    Returns:
        float: Predicted student ride time in minutes
    """
    # Create temporary stop list with new_stop inserted at its candidate position.
    temp_stops = list(route.stops)
    temp_stops.insert(insert_position, new_stop)

    # The new student boards at insert_position.  Their ride time is the sum
    # of legs from that stop to school — NOT from the first occupied stop in
    # the route (which was the previous, incorrect behaviour).
    if insert_position >= len(temp_stops) - 1:
        return 0.0

    ride_time = 0.0
    for i in range(insert_position, len(temp_stops) - 1):
        u = temp_stops[i].node_id
        v = temp_stops[i + 1].node_id
        # Fast matrix lookup first
        if (u, v) in _MATRIX_CACHE:
            t = _MATRIX_CACHE[(u, v)]
            if t == float('inf'):
                return 9999.0
            ride_time += t
        else:
            # Fallback to graph search
            path, time_minutes = find_shortest_path_with_turns(graph, u, v)
            if time_minutes < float('inf'):
                ride_time += time_minutes
            else:
                return 9999.0
    return ride_time


def calculate_afternoon_ride_time_potential(route, new_stop, insert_position, graph,
                                            target_stop=None):
    """Compute the PM ride time for a student in the reversed route after a potential insertion.
    
    Afternoon route = morning route with pickup stops reversed:
        [School, Stop_n, ..., Stop_1, School]
    A student's afternoon ride = time from school to their stop in the reversed sequence.
    Students near school in the morning (last pickup) are dropped off FIRST in the afternoon.
    
    Args:
        route: Route object (morning direction)
        new_stop: Stop being inserted
        insert_position: Position in morning sequence
        graph: NetworkX graph
        target_stop: Which stop to measure for; None → new_stop itself
    
    Returns:
        float: Afternoon ride time in minutes
    """
    temp_stops = list(route.stops)
    temp_stops.insert(insert_position, new_stop)

    # Reverse the interior pickup stops; keep school at start/end
    interior = temp_stops[1:-1][::-1]
    afternoon_stops = [temp_stops[0]] + interior + [temp_stops[-1]]

    target = target_stop if target_stop is not None else new_stop

    # Find target in afternoon sequence (match by identity first, then node_id)
    target_idx = -1
    for i, s in enumerate(afternoon_stops):
        if s is target or (target_stop is None and s.node_id == new_stop.node_id):
            target_idx = i
            break

    if target_idx <= 0:
        return 0.0  # school position or not found

    ride_time = 0.0
    for i in range(target_idx):
        u = afternoon_stops[i].node_id
        v = afternoon_stops[i + 1].node_id
        t = _MATRIX_CACHE.get((u, v), None)
        if t is None:
            # OPTIMIZATION: Cache reverse edges to speed up future PM checks
            path, t = find_shortest_path_with_turns(graph, u, v)
            _MATRIX_CACHE[(u, v)] = t
            
            # Optionally cache length if you have the path (like calculate_route_time_from_matrix does)
            if path:
                dist_m = 0.0
                for pi in range(len(path) - 1):
                    ed = graph.get_edge_data(path[pi], path[pi+1])
                    if ed:
                        d = ed[0] if 0 in ed else list(ed.values())[0]
                        dist_m += d.get('length', 0)
                _MATRIX_CACHE_LENGTH[(u, v)] = dist_m
                
        if t == float('inf'):
            return 9999.0
        ride_time += t
    return ride_time
    """Check if all students on the route can reach their stops safely.
    
    A route is safe if every student can reach their assigned stop via
    a pedestrian path that does not cross any arterial roads, and within
    their maximum walk distance.
    
    Args:
        route: Route object
        graph: NetworkX road network
        walk_distance_limits: Dict mapping stop_id to max walk distance
        
    Returns:
        Tuple of (is_safe: bool, unsafe_students: list)
            - is_safe: True if all students can reach stops safely
            - unsafe_students: List of (student, reason) for unsafe assignments
    """
    unsafe_students = []
    
    for stop in route.stops:
        for student in stop.students:
            # Check 1: Student is within walk distance of stop
            lat_s, lon_s = student.coords
            lat_stop, lon_stop = stop.coords
            
            walk_distance = math.sqrt((lat_s - lat_stop)**2 + (lon_s - lon_stop)**2) * 111000  # meters
            
            if walk_distance > student.walk_radius:
                unsafe_students.append((student, f"Beyond walk radius: {walk_distance}m > {student.walk_radius}m"))
                continue
            
            # Check 2: Path from student to stop is safe (details in find_safe_nodes_within_radius)
            walk_g = _get_walk_graph(graph)  # Use walk graph with crossings if available
            safe_nodes = find_safe_nodes_within_radius(
                student.coords,
                graph,
                500,
                student.walk_radius,
                walk_graph=walk_g,
                student_stage=getattr(student, "school_stage", None),
                student_disabled=bool(getattr(student, "physically_mentally_disabled", False)),
            )
            safe_node_ids = [n[0] for n in safe_nodes]
            
            if stop.node_id not in safe_node_ids:
                unsafe_students.append((student, "No safe pedestrian path to stop (arterial crossing required)"))
    
    return len(unsafe_students) == 0, unsafe_students


# ============================================================================
# INSERTION COST CALCULATION: Core Cheapest Insertion logic
# ============================================================================

def calculate_insertion_cost(new_stop, route, insert_position, graph):
    """
    Calculate the time cost of inserting a stop at a specific position.
    Uses incremental delta logic to speed up ALNS by 100x.
    """
    if insert_position < 1 or insert_position >= len(route.stops):
        return None, False, "Insertion must be between existing stops"
    
    # We assume routes have fixed Start (Depot/School) and End (School)
    u_node = route.stops[insert_position - 1].node_id
    v_node = route.stops[insert_position].node_id
    
    # Calculate detour leg 1: u -> new
    # (Optional: Pass incoming bearing from route.stops[i-2] if you want extreme turn precision)
    dt1 = shortest_path_length_with_turns(graph, u_node, new_stop.node_id)
    
    # Calculate detour leg 2: new -> v
    dt2 = shortest_path_length_with_turns(graph, new_stop.node_id, v_node)
    
    # Current distance between u and v
    dt_old = shortest_path_length_with_turns(graph, u_node, v_node)
    
    if dt1 == float('inf') or dt2 == float('inf'):
        return None, False, "No valid path"
        
    delta_time = dt1 + dt2 - dt_old
    return delta_time, True, "Success"


# ============================================================================
# CONSTRAINT VALIDATORS: Check if detour/insertion is allowed
# ============================================================================

def validate_temporary_detour(new_stop, route, delta_time_minutes, daily_budget=5):
    """Validate if a temporary detour request can be accepted.
    
    A temporary detour is a one-time request for a student to get an
    alternate drop-off. The cumulative time cost of this detour plus all
    prior detours today must not exceed the daily budget (5 minutes max).
    
    Args:
        new_stop: Stop object for the temporary detour
        route: Existing Route object
        delta_time_minutes: Time cost of this specific detour (minutes)
        daily_budget: Maximum cumulative detour time per day (minutes, default 5)
        
    Returns:
        Tuple of (valid: bool, remaining_budget: float, reason: str)
    """
    # Check 1: Would adding this detour exceed the daily budget?
    current_used = route.get_current_detour_time()
    total_if_added = current_used + delta_time_minutes
    
    if total_if_added > daily_budget:
        remaining = daily_budget - current_used
        return False, remaining, f"Detour would exceed daily budget: {total_if_added:.2f} > {daily_budget} min (only {remaining:.2f} min remaining)"
    
    # Check 2: Is there a safe pedestrian path to this location?
    # (This would be validated via find_safe_nodes_within_radius in practice)
    # For now, assume safe path exists; validation happens during insertion.
    
    remaining = daily_budget - total_if_added
    return True, remaining, f"Temporary detour accepted ({total_if_added:.2f}/{daily_budget} min used)"


def compute_direct_time(student, school_node, graph):
    """Compute direct drive time from student's home to school (minutes).
    
    Uses the precomputed distance matrix when available, falling back to
    an on-demand A* search.  Result is cached on the student object so
    subsequent calls are O(1).
    
    Args:
        student: Student object (must have .coords)
        school_node: OSM node ID of the school
        graph: NetworkX road network
    
    Returns:
        float: Travel time in minutes (direct, no detours)
    """
    if student.direct_time_to_school is not None:
        return student.direct_time_to_school

    lat, lon = student.coords
    student_node = fast_nearest_node(graph, lon, lat)

    # Check precomputed matrix first (fast, no graph search)
    cached = _MATRIX_CACHE.get((student_node, school_node), None)
    if cached is not None and cached < float('inf'):
        student.direct_time_to_school = cached
        return cached

    # Fallback: on-demand A* search
    _, t = find_shortest_path_with_turns(graph, student_node, school_node)
    if t == float('inf'):
        # Last-resort: try nearest reachable node
        t = float('inf')
    student.direct_time_to_school = t
    return t


def compute_afternoon_direct_time(student, school_node, graph):
    """Compute direct drive time from school to student's home (afternoon direction).
    
    Due to one-way streets, this may differ from compute_direct_time.
    Cached on student.direct_time_from_school for O(1) repeat calls.
    """
    if getattr(student, 'direct_time_from_school', None) is not None:
        return student.direct_time_from_school

    lat, lon = student.coords
    student_node = fast_nearest_node(graph, lon, lat)

    cached = _MATRIX_CACHE.get((school_node, student_node), None)
    if cached is not None and cached < float('inf'):
        student.direct_time_from_school = cached
        return cached

    _, t = find_shortest_path_with_turns(graph, school_node, student_node)
    student.direct_time_from_school = t
    return t


def compute_student_tmax(student, school_node, G,
                          multiplier=2.5,
                          floor_minutes=45,
                                                    ceiling_minutes=60,
                                                    base_mrt_minutes=None,
                                                    acceptable_offset_minutes=None):
    """
        DMRT per-student ride-time cap (morning direction: home -> school):

                T_max(s) = max(base_mrt_minutes, T_direct + acceptable_offset_minutes)

        Defaults:
            base_mrt_minutes          -> floor_minutes (legacy fallback)
            acceptable_offset_minutes -> ceiling_minutes (legacy fallback)
    """
    t_direct = compute_direct_time(student, school_node, G)

    if t_direct == float('inf'):
        return float('inf')
    if t_direct <= 0:
        return float(base_mrt_minutes if base_mrt_minutes is not None else floor_minutes)

    if base_mrt_minutes is None:
        base_mrt_minutes = floor_minutes
    if acceptable_offset_minutes is None:
        acceptable_offset_minutes = ceiling_minutes

    personal_tmax = max(float(base_mrt_minutes), float(t_direct) + float(acceptable_offset_minutes))

    # Cache on student for visualization and logging
    student.direct_time_to_school = t_direct
    student.personal_tmax         = personal_tmax
    return personal_tmax


def validate_permanent_student(new_stop, route, insert_position, delta_time_minutes, graph,
                               new_student=None):
    """Validate if a permanent student can be added to the route.
    
    Uses per-student ride-time caps when the student object is available:
        T_ride \u2264 max(base_mrt, T_direct + acceptable_offset)
    Falls back to the flat route_tmax when no student object is provided.
    
    Also checks that no existing student on the route has their personal
    cap violated by the insertion (inserting a stop after them increases
    their ride time).
    
    A permanent student assignment is accepted if it doesn't violate any
    ride-time constraint and bus capacity is available.
    
    Args:
        new_stop: Stop object for the new student
        route: Existing Route object
        insert_position: Where the stop is being inserted
        delta_time_minutes: Time cost of this insertion (for info)
        graph: NetworkX road network
        new_student: Optional Student object; enables per-student ride-time caps
        
    Returns:
        Tuple of (valid: bool, student_ride_time: float, reason: str)
    """
    # Check 1: Bus capacity
    if route.get_student_count() >= route.bus.capacity:
        return False, route.total_time, f"Bus at capacity ({route.get_student_count()}/{route.bus.capacity})"

    school_node  = route.stops[-1].node_id
    k            = getattr(route, 'ride_time_multiplier', 2.5)
    floor_min    = getattr(route, 'floor_minutes',        45)
    ceiling_min  = getattr(route, 'ceiling_minutes',      60)  # extra minutes over direct
    dmrt_offset  = getattr(route, 'acceptable_offset_minutes', 30)
    mrt_enabled  = bool(getattr(route, 'mrt_enabled', False))
    mrt_minutes  = getattr(route, 'mrt_minutes', None)
    try:
        mrt_minutes = float(mrt_minutes) if mrt_minutes is not None else None
    except (TypeError, ValueError):
        mrt_minutes = None
    try:
        dmrt_offset = float(dmrt_offset)
    except (TypeError, ValueError):
        dmrt_offset = 30.0
    # DMRT should only use fixed MRT minutes when hard MRT mode is enabled.
    base_mrt = mrt_minutes if (mrt_enabled and mrt_minutes is not None and mrt_minutes > 0) else floor_min

    caps_enabled = getattr(route, 'ride_caps_enabled', True)

    if not caps_enabled:
        # Skip all ride-time checks when caps are disabled.
        new_student_ride_time = calculate_student_ride_time_potential(
            route, new_stop, insert_position, graph
        )
        return True, new_student_ride_time, "Ride-time caps not enforced"

    # Fast-exit for unconstrained / benchmark mode: skip all ride-time checks.
    # When multiplier >= 100 and floor >= 999, no real cap exists; avoid O(N^2) work.
    if k >= 100 and floor_min >= 999:
        return True, 0.0, "Unconstrained accepted"

    # Check 2: New student's personal ride-time cap (bidirectional fairness rule)
    # A route is only rejected for ride time if the constraint is broken in BOTH
    # the morning (home→school) AND the afternoon (school→home reversed route).
    new_student_ride_time = calculate_student_ride_time_potential(route, new_stop, insert_position, graph)

    bidir          = getattr(route, 'bidirectional_check',      True)

    if new_student is not None:
        if mrt_enabled and mrt_minutes is not None and mrt_minutes > 0:
            # Hard MRT mode: reject if either AM or PM exceeds fixed MRT.
            pm_ride = calculate_afternoon_ride_time_potential(route, new_stop, insert_position, graph)
            if new_student_ride_time > mrt_minutes or pm_ride > mrt_minutes:
                return (False, new_student_ride_time,
                        f"Hard MRT exceeded: AM {new_student_ride_time:.1f}, PM {pm_ride:.1f} > {mrt_minutes:.1f} min")
        else:
            morning_cap = compute_student_tmax(
                new_student,
                school_node,
                graph,
                k,
                floor_min,
                ceiling_min,
                base_mrt_minutes=base_mrt,
                acceptable_offset_minutes=dmrt_offset,
            )
            t_direct    = compute_direct_time(new_student, school_node, graph)
            am_violated = new_student_ride_time > morning_cap

            if am_violated:
                if not bidir:
                    # Strict one-direction check — reject immediately
                    return (False, new_student_ride_time,
                            f"AM ride cap exceeded: "
                            f"{new_student_ride_time:.1f}>{morning_cap:.1f} min "
                            f"(direct={t_direct:.1f}, max({base_mrt:.1f}, direct+{dmrt_offset:.1f}))")
                # Bidirectional leniency: only reject if PM is also too long
                pm_ride    = calculate_afternoon_ride_time_potential(route, new_stop, insert_position, graph)
                pm_violated = pm_ride > morning_cap
                if pm_violated:
                    return (False, new_student_ride_time,
                            f"Ride cap exceeded in both directions — AM {new_student_ride_time:.1f} "
                            f"PM {pm_ride:.1f} > {morning_cap:.1f} min")
                # AM violated but PM is within cap → accept under bidirectional leniency
    else:
        # Fallback: flat route_tmax
        if new_student_ride_time > route.route_tmax:
            return (False, new_student_ride_time,
                    f"Student ride time exceeds Tmax: {new_student_ride_time:.1f} > {route.route_tmax} min")

    # Check 3: Existing students whose morning ride increases due to this insertion.
    # Check 3: Existing students whose morning ride increases due to this insertion.
    if new_student is not None:
        # --- SUPER OPTIMIZATION: O(1) Ride Time Check ---
        # Precompute the current time from each stop to the school ONCE (O(N) instead of O(N^3))
        old_ride_times = {}
        accumulated = 0.0
        if route.stops:
            for i in range(len(route.stops) - 1, 0, -1):
                u = route.stops[i-1].node_id
                v = route.stops[i].node_id
                t = _MATRIX_CACHE.get((u, v), float('inf'))
                accumulated += t
                old_ride_times[route.stops[i-1]] = accumulated

        for stop in route.stops:
            if stop.stop_type == 'school':
                continue
                
            stop_idx = route.stops.index(stop) if stop in route.stops else -1
            if stop_idx == -1 or stop_idx >= insert_position:
                continue  # boards AFTER new stop — morning ride unaffected

            # The new ride time is simply their exact old ride time + the detour delta
            old_ride_time = old_ride_times.get(stop, float('inf'))
            morning_ride_check = old_ride_time + delta_time_minutes

            for existing_student in stop.students:
                if mrt_enabled and mrt_minutes is not None and mrt_minutes > 0:
                    pm_ride_ex = calculate_afternoon_ride_time_potential(
                        route, new_stop, insert_position, graph, target_stop=stop)
                    if morning_ride_check > mrt_minutes or pm_ride_ex > mrt_minutes:
                        return (False, morning_ride_check,
                                f"Insertion pushes {existing_student.id} over hard MRT")
                else:
                    existing_cap = compute_student_tmax(
                        existing_student,
                        school_node,
                        graph,
                        k,
                        floor_min,
                        ceiling_min,
                        base_mrt_minutes=base_mrt,
                        acceptable_offset_minutes=dmrt_offset,
                    )

                    if morning_ride_check > existing_cap:
                        if not bidir:
                            return (False, morning_ride_check,
                                    f"Insertion pushes {existing_student.id} over AM cap")

                        # Bidirectional: check PM for this existing student
                        pm_ride_ex  = calculate_afternoon_ride_time_potential(
                            route, new_stop, insert_position, graph, target_stop=stop)
                        if pm_ride_ex > existing_cap:
                            return (False, morning_ride_check,
                                    f"Insertion pushes {existing_student.id} over cap in both directions")

    return True, new_student_ride_time, "Permanent student accepted"
    
    # if new_student is not None:
    #     for stop in route.stops:
    #         if stop.stop_type == 'school':
    #             continue
    #         for existing_student in stop.students:
    #             t_d = compute_direct_time(existing_student, school_node, graph)
    #             if t_d == float('inf') or t_d <= 0:
    #                 continue
    #             ex_floor   = getattr(existing_student, 'floor_minutes',   floor_min)
    #             ex_ceiling = getattr(existing_student, 'ceiling_minutes', ceiling_min)
    #             existing_cap = max(ex_floor, min(k * t_d, t_d + ex_ceiling))

    #             stop_idx = route.stops.index(stop) if stop in route.stops else -1
    #             if stop_idx == -1:
    #                 continue
    #             if stop_idx >= insert_position:
    #                 continue  # boards AFTER new stop — morning ride unaffected

    #             # Build post-insertion list and measure this student's new morning ride
    #             temp_stops = list(route.stops)
    #             temp_stops.insert(insert_position, new_stop)
    #             student_stop_idx_in_temp = next(
    #                 (ti for ti, ts in enumerate(temp_stops) if ts is stop), -1)
    #             if student_stop_idx_in_temp == -1:
    #                 continue

    #             morning_ride_check = 0.0
    #             for si in range(student_stop_idx_in_temp, len(temp_stops) - 1):
    #                 u = temp_stops[si].node_id
    #                 v = temp_stops[si + 1].node_id
    #                 t = _MATRIX_CACHE.get((u, v), None)
    #                 if t is None:
    #                     _, t = find_shortest_path_with_turns(graph, u, v)
    #                 if t == float('inf'):
    #                     morning_ride_check = float('inf')
    #                     break
    #                 morning_ride_check += t

    #             ex_am_violated = morning_ride_check > existing_cap

    #             if ex_am_violated:
    #                 if not bidir:
    #                     return (False, morning_ride_check,
    #                             f"Insertion pushes {existing_student.id} over AM cap: "
    #                             f"{morning_ride_check:.1f}>{existing_cap:.1f} min")
    #                 # Bidirectional: check PM for this existing student
    #                 pm_ride_ex  = calculate_afternoon_ride_time_potential(
    #                     route, new_stop, insert_position, graph, target_stop=stop)
    #                 ex_pm_violated = pm_ride_ex > existing_cap
    #                 if ex_pm_violated:
    #                     return (False, morning_ride_check,
    #                             f"Insertion pushes {existing_student.id} over cap in both directions "
    #                             f"AM {morning_ride_check:.1f} PM {pm_ride_ex:.1f} > {existing_cap:.1f} min")

    # return True, new_student_ride_time, "Permanent student accepted"


# ============================================================================
# CHEAPEST INSERTION ALGORITHM: Find best route and position for student
# ============================================================================

def cheapest_insertion(new_student, existing_routes, graph, detour_type='temporary', 
                       daily_detour_budget=5, student_walk_distance_limit=None):
    """Find the cheapest insertion position across all existing routes.
    
    This is the core algorithm for the Dynamic Detour Engine. It evaluates
    inserting a new student (via a new stop) at every possible position in
    every existing route, returns the minimum-cost option that passes all
    constraints.
    
    Args:
        new_student: Student object to be added
        existing_routes: List of Route objects
        graph: NetworkX road network
        detour_type: 'temporary' or 'permanent'
        daily_detour_budget: Budget for temporary detours (minutes)
        student_walk_distance_limit: Optional override for student walk distance
        
    Returns:
        Tuple of (result_dict or None, reason: str)
            result_dict contains:
                - route: Route object
                - new_stop: Stop object
                - insertion_position: Position in route.stops
                - insertion_cost_minutes: Time delta
                - is_new_stop: Whether a new stop was created or existing used
            reason: Explanation of result
    """
    best_cost = float('inf')
    best_route = None
    best_position = None
    best_stop = None
    best_reason = ""
    
    walk_limit = student_walk_distance_limit or new_student.walk_radius
    
    # Mapping from node_id to specific coordinates (for virtual stops)
    node_coords_mapping = {}
    
    # ALWAYS ensure the absolute frontage (virtual node) is the first candidate
    # This is critical for precisely hitting the front of the house
    frontage_node_id, frontage_coords = snap_address_to_edge(new_student.coords, graph)
    candidate_node_ids = [frontage_node_id]
    node_coords_mapping[frontage_node_id] = frontage_coords
    
    # 2. Find other candidate nodes within walking distance (only if walk_limit > 0)
    if walk_limit > 0:
        walk_g = _get_walk_graph(graph)  # Use walk graph with crossings if available
        safe_nodes = find_safe_nodes_within_radius(
            new_student.coords,
            graph,
            500,
            walk_limit,
            walk_graph=walk_g,
            student_stage=getattr(new_student, "school_stage", None),
            student_disabled=bool(getattr(new_student, "physically_mentally_disabled", False)),
        )
        for node_id, dist in sorted(safe_nodes, key=lambda x: x[1]):
            if node_id not in candidate_node_ids:
                candidate_node_ids.append(node_id)
    
    # 3. Bus-reachable fallback: If frontage is unreachable by bus (one-way / U-turn),
    #    find nearby graph nodes that ARE bus-reachable and within a reasonable walk.
    #    This handles cases where the student's street is one-way AWAY from school.
    frontage_reachable = (
        _MATRIX_CACHE.get((frontage_node_id, existing_routes[0].stops[0].node_id if existing_routes else None), float('inf')) < float('inf')
        and _MATRIX_CACHE.get((existing_routes[0].stops[0].node_id if existing_routes else None, frontage_node_id), float('inf')) < float('inf')
    ) if existing_routes else True
    
    if not frontage_reachable:
        # Find nearest graph nodes that the bus CAN reach via bidirectional BFS
        # (walking is not constrained by one-way streets)
        lat, lon = new_student.coords
        try:
            center_node = fast_nearest_node(graph, lon, lat)
            visited = set()
            bfs_queue = [(center_node, 0)]
            reachable_candidates = []
            max_walk = get_walk_absolute_max(walk_limit)  # Stage-based absolute max
            school_node = existing_routes[0].stops[0].node_id
            while bfs_queue and len(reachable_candidates) < 10:
                node, dist = bfs_queue.pop(0)
                if node in visited or dist > max_walk:
                    continue
                visited.add(node)
                to_school = _MATRIX_CACHE.get((node, school_node), float('inf'))
                from_school = _MATRIX_CACHE.get((school_node, node), float('inf'))
                if to_school < float('inf') and from_school < float('inf'):
                    reachable_candidates.append((node, dist))
                # Expand along out-edges
                for neighbor in graph.successors(node):
                    ed = graph.get_edge_data(node, neighbor)
                    if ed:
                        d = ed[0] if 0 in ed else list(ed.values())[0]
                        new_dist = dist + d.get('length', 0)
                        if new_dist <= max_walk:
                            bfs_queue.append((neighbor, new_dist))
                # Also expand along in-edges (walking is bidirectional)
                for predecessor in graph.predecessors(node):
                    ed = graph.get_edge_data(predecessor, node)
                    if ed:
                        d = ed[0] if 0 in ed else list(ed.values())[0]
                        new_dist = dist + d.get('length', 0)
                        if new_dist <= max_walk:
                            bfs_queue.append((predecessor, new_dist))
            
            for node_id, dist in sorted(reachable_candidates, key=lambda x: x[1]):
                if node_id not in candidate_node_ids:
                    candidate_node_ids.append(node_id)
        except Exception:
            pass
    
    # To keep it efficient, we'll only check up to 8 best candidate nodes
    candidate_node_ids = candidate_node_ids[:8]
    
    # Pre-filter: skip candidates not known to be bus-reachable (avoids cold A* calls)
    if existing_routes:
        school_node = existing_routes[0].stops[0].node_id
        filtered = []
        for cid in candidate_node_ids:
            to_s = _MATRIX_CACHE.get((cid, school_node), None)
            from_s = _MATRIX_CACHE.get((school_node, cid), None)
            if to_s is not None and from_s is not None and to_s < float('inf') and from_s < float('inf'):
                filtered.append(cid)
        candidate_node_ids = filtered if filtered else candidate_node_ids  # Fallback if nothing passes
    
    for route in existing_routes:

        # If route has at least 2 stops (Start and End), strictly insert between them
        if len(route.stops) >= 2:
            start_pos = 1
            end_pos = len(route.stops)
        else:
            start_pos = 0
            end_pos = len(route.stops) + 1
            
        # Try every valid insertion position in this route
        for position in range(start_pos, end_pos):
            
            # Check each candidate node for this position
            for node_id in candidate_node_ids:
                # Check if a stop already exists at this node in this route
                existing_stop = None
                for stop in route.stops:
                    if stop.node_id == node_id:
                        existing_stop = stop
                        break
                
                if existing_stop:
                    eval_stop = existing_stop
                else:
                    # Use virtual snapped coords if available, else use graph node coords
                    if node_id in node_coords_mapping:
                        lat, lon = node_coords_mapping[node_id]
                    else:
                        lat = graph.nodes[node_id]['y']
                        lon = graph.nodes[node_id]['x']
                    
                    eval_stop = Stop(node_id, lat, lon)

                
                # Calculate insertion cost
                delta_time, is_valid, cost_reason = calculate_insertion_cost(
                    eval_stop, route, position, graph
                )
                
                if not is_valid:
                    continue
                
                # Validate constraints based on detour type
                if detour_type == 'temporary':
                    valid, remaining, constraint_reason = validate_temporary_detour(
                        eval_stop, route, delta_time, daily_detour_budget
                    )
                    if not valid:
                        continue
                else:  # permanent
                    valid, new_ride_time, constraint_reason = validate_permanent_student(
                        eval_stop, route, position, delta_time, graph,
                        new_student=new_student
                    )
                    if not valid:
                        continue
                
                # Calculate walk penalty (straight-line distance)
                walk_penalty, walk_m, over_limit = calculate_walk_penalty(
                    new_student, node_id, graph
                )
                if walk_penalty == float('inf'):
                    continue  # Beyond absolute walk maximum
                
                # Total cost = bus insertion time + walk penalty
                total_cost = delta_time + walk_penalty
                
                # Track best option (penalized cost for comparison)
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_route = route
                    best_position = position
                    best_stop = eval_stop
                    best_reason = f"Cost: {delta_time:.2f} min + walk penalty {walk_penalty:.1f} min (walk {walk_m:.0f}m), {constraint_reason}"
    
    # Return result
    if best_route is not None:
        result = {
            'route': best_route,
            'new_stop': best_stop,
            'insertion_position': best_position,
            'insertion_cost_minutes': best_cost,
            'is_new_stop': best_stop not in best_route.stops
        }
        return result, best_reason
    else:
        return None, "No valid insertion found in any existing route"


# ============================================================================
# 2-OPT INTRA-ROUTE OPTIMIZATION
# ============================================================================

def two_opt_improve(route, graph):
    """Apply 2-opt local search to improve a single route's stop ordering.
    
    Repeatedly tries reversing sub-sequences of pickup stops (keeping
    school start/end stops fixed) and keeps each improvement that reduces
    total route time. Runs until no further improvement is found.
    
    Args:
        route: Route object with at least 2 stops (school-start ... school-end)
        graph: NetworkX road graph
        
    Returns:
        float: Total time improvement in minutes (positive = saved)
    """
    if len(route.stops) < 4:
        # Need at least: school-start, 2 pickups, school-end to swap anything
        return 0.0
    
    original_time = calculate_route_time(route, graph)
    improved = True
    
    while improved:
        improved = False
        # Only reverse among interior stops (index 1 .. len-2)
        n = len(route.stops)
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                # Try reversing stops[i..j]
                new_stops = route.stops[:i] + route.stops[i:j+1][::-1] + route.stops[j+1:]
                
                # Evaluate the new ordering
                old_stops = route.stops
                route.stops = new_stops
                new_time = calculate_route_time(route, graph)
                
                if new_time < original_time - 0.01:
                    # Improvement found — keep it
                    original_time = new_time
                    route.total_time = new_time
                    improved = True
                else:
                    # Revert
                    route.stops = old_stops
    
    final_time = calculate_route_time(route, graph)
    route.total_time = final_time
    route.total_distance = calculate_route_distance(route, graph)
    
    return max(0.0, original_time - final_time)


def insert_with_2opt(new_student, existing_routes, graph, detour_type='temporary',
                     daily_detour_budget=5):
    """Insert student via cheapest insertion, then improve with 2-opt.
    
    Combines cheapest_insertion for initial placement with two_opt_improve
    for local search. This is the recommended fast algorithm for Mode 2
    change_location requests.
    
    Args:
        new_student: Student to insert
        existing_routes: List of Route objects
        graph: NetworkX road graph
        detour_type: 'temporary' or 'permanent'
        daily_detour_budget: Budget for temporary detours
        
    Returns:
        Same as process_detour_request: (success, route_or_none, message)
    """
    if not existing_routes:
        return False, None, "No existing routes available"
    
    # Step 1: Cheapest Insertion
    result, reason = cheapest_insertion(
        new_student, existing_routes, graph, detour_type, daily_detour_budget
    )
    
    if result is None:
        return False, None, f"Insertion failed: {reason}"
    
    route = result['route']
    new_stop = result['new_stop']
    position = result['insertion_position']
    delta_time = result['insertion_cost_minutes']
    
    # Insert the stop
    if result['is_new_stop']:
        route.stops.insert(position, new_stop)
    
    # Set student assignment type
    new_student.assignment = detour_type  # "temporary" or "permanent"
    new_stop.add_student(new_student)
    
    # Step 2: 2-opt improvement on the affected route
    time_saved = two_opt_improve(route, graph)
    
    # Update accounting
    if detour_type == 'temporary':
        route.add_detour_time(max(0, delta_time - time_saved))
    
    route.total_distance = calculate_route_distance(route, graph)
    route.total_time = calculate_route_time(route, graph)
    
    message = (f"2-opt insert: {new_student.id} -> Route {route.route_id}, "
               f"Stop {new_stop.node_id}, Cost +{delta_time:.2f} min, "
               f"2-opt saved {time_saved:.2f} min")
    return True, route, message


# ============================================================================
# MAIN DETOUR REQUEST HANDLER
# ============================================================================

def process_detour_request(student, existing_routes, graph, detour_type='temporary', 
                          daily_detour_budget=5):
    """Process a student's request for a detour/alternate drop-off location.
    
    This is the main entry point for handling dynamic detour requests. It:
    1. Snaps the student to the road network
    2. Finds the cheapest insertion across all routes
    3. Validates safety and time constraints
    4. Updates the route if accepted
    
    Args:
        student: Student object with coords and walk_radius set
        existing_routes: List of Route objects operating today
        graph: NetworkX road network with 'travel_time' and 'is_safe_to_cross' attributes
        detour_type: 'temporary' (one-time request) or 'permanent' (add to system)
        daily_detour_budget: Daily budget for temporary detours (minutes, default 5)
        
    Returns:
        Tuple of (success: bool, route_updated: Route or None, message: str)
    """
    if not existing_routes:
        return False, None, "No existing routes available"
    
    # Run cheapest insertion algorithm
    result, reason = cheapest_insertion(
        student, existing_routes, graph, detour_type, daily_detour_budget
    )
    
    if result is None:
        return False, None, f"Detour rejected: {reason}"
    
    # Unpack result
    route = result['route']
    new_stop = result['new_stop']
    position = result['insertion_position']
    delta_time = result['insertion_cost_minutes']
    
    # Update route
    if result['is_new_stop']:
        route.stops.insert(position, new_stop)
    
    # Set student assignment type
    student.assignment = detour_type  # "temporary" or "permanent"
    new_stop.add_student(student)
    
    # Update accounting
    if detour_type == 'temporary':
        route.add_detour_time(delta_time)
    
    # Recalculate route totals
    route.total_distance = calculate_route_distance(route, graph)
    route.total_time = calculate_route_time(route, graph)
    
    message = f"Detour accepted: {student.id} -> Route {route.route_id}, Stop {new_stop.node_id}, Cost +{delta_time:.2f} min"
    return True, route, message


# for OSRM integration: Precompute the full distance matrix for all nodes in the graph

def precalculate_distance_matrix_osrm(G, nodes_list):
    """
    Replaces the memory-crashing Python A* matrix calculation.
    Queries a local OSRM Docker container to get all distances instantly.
    """
    print(f"Asking local OSRM to calculate matrix for {len(nodes_list)} nodes...")
    
    # 1. Map node IDs to their Longitude/Latitude
    # OSRM expects the format: lon,lat
    coords_str_list = []
    for node in nodes_list:
        lat = G.nodes[node]['y']
        lon = G.nodes[node]['x']
        coords_str_list.append(f"{lon},{lat}")
        
    coords_string = ";".join(coords_str_list)
    
    # 2. Make the HTTP request to the local OSRM Docker container
    # We ask for both 'duration' and 'distance' annotations
    url = f"http://localhost:5000/table/v1/driving/{coords_string}?annotations=duration,distance"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"OSRM Error: Is the Docker container running? {e}")
        return
        
    durations = data.get('durations', [])
    distances = data.get('distances', [])
    
    # 3. Save the results into the exact cache format the ALNS engine uses
    for i, origin_node in enumerate(nodes_list):
        for j, dest_node in enumerate(nodes_list):
            
            # --- HANDLE DURATIONS ---
            if durations[i][j] is not None:
                # Convert OSRM seconds to minutes
                _MATRIX_CACHE[(origin_node, dest_node)] = durations[i][j] / 60.0
            else:
                # CRITICAL FIX: Prevent A* Death Spiral by caching infinity
                _MATRIX_CACHE[(origin_node, dest_node)] = float('inf')
                
            # --- HANDLE DISTANCES ---
            if distances[i][j] is not None:
                _MATRIX_CACHE_LENGTH[(origin_node, dest_node)] = distances[i][j]
            else:
                _MATRIX_CACHE_LENGTH[(origin_node, dest_node)] = float('inf')
                
    print("OSRM Matrix calculation complete! Cache populated.")