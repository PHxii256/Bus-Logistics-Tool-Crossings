"""
Safety-Aware Bus Optimization  Main Entry Point

Dispatches based on the 'mode' field in the input JSON:
  - generate_routes : Full ALNS optimization from scratch
  - change_location : Insert/move a single student into existing routes
"""

import sys
import os
import json
import time as _t
import hashlib
import pickle
import shutil
import osmnx as ox
import networkx as nx
import argparse

from data_loader import (
    load_json, load_mode1_input, load_mode2_input,
    serialize_routes, print_input_summary
)
import detour_engine as _det_eng
import alns_engine as _alns
from detour_engine import (
    calculate_route_distance, calculate_route_time,
    cheapest_insertion, process_detour_request, insert_with_2opt,
    snap_address_to_edge, precalculate_distance_matrix,
    find_safe_nodes_within_radius, find_shortest_path_with_turns, _get_walk_graph,
    get_walk_absolute_max, haversine_walk_distance,
    _MATRIX_CACHE, _MATRIX_CACHE_LENGTH, _path_cache
)
from visualization import create_route_map
from solution_state import ServiceSolution
from alns_engine import ALNSEngine
from entities import Student, School_Stage

# ============================================================================
# DEFAULT WALK LIMITS PER SCHOOL STAGE  (metres)
# ============================================================================
DEFAULT_STAGE_WALK_LIMITS = {
    "KG":         0,
    "ELEMENTARY": 0,
    "MIDDLE":     150,
    "HIGH":       200,
}

# ============================================================================
# RUN HISTORY: Save each run to runs_history/{mode}_{school}_{hash8}/
# ============================================================================

def _input_hash(data: dict) -> str:
    """Stable 8-char hash of the input so same input  same folder."""
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=True)
    return hashlib.md5(canonical.encode()).hexdigest()[:8]

def save_run(input_data: dict, output_data: dict, report_data: dict,
            map_files: dict = None):
    run_hash = _input_hash(input_data)
    run_dir  = os.path.join('runs_history', run_hash)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'input.json'),  'w') as f:
        json.dump(input_data,  f, indent=2)
    with open(os.path.join(run_dir, 'output.json'), 'w') as f:
        json.dump(output_data, f, indent=2)
    with open(os.path.join(run_dir, 'report.json'), 'w') as f:
        json.dump(report_data, f, indent=2)
    if map_files:
        for dest_name, src_path in map_files.items():
            if os.path.exists(src_path):
                shutil.copy2(src_path, os.path.join(run_dir, dest_name))
    print(f"  Run saved to '{run_dir}/'")
    return run_dir

# ============================================================================
# GRAPH SETUP
# ============================================================================

_DEFAULT_BBOX = [31.229084, 29.925630, 31.331909, 29.991682]
_ROAD_SPEEDS_CONFIG_PATH = 'road_speeds_config.json'

def _load_road_speeds(override: dict = None) -> dict:
    builtin = {
        'default_speed_kph': 30,
        'road_types': {
            'primary':       {'speed_multiplier': 0.8, 'safe_to_cross': False},
            'trunk':         {'speed_multiplier': 0.8, 'safe_to_cross': False},
            'secondary':     {'speed_multiplier': 0.6, 'safe_to_cross': False},
            'tertiary':      {'speed_multiplier': 0.6, 'safe_to_cross': True},
            'residential':   {'speed_multiplier': 0.3, 'safe_to_cross': True},
            'living_street': {'speed_multiplier': 0.3, 'safe_to_cross': True},
            'default':       {'speed_multiplier': 0.2, 'safe_to_cross': True},
        }
    }
    try:
        with open(_ROAD_SPEEDS_CONFIG_PATH) as f:
            file_cfg = json.load(f)
        builtin.update({k: v for k, v in file_cfg.items() if not k.startswith('_')})
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    if override:
        builtin.update(override)
    return builtin

def setup_graph(meta: dict = None, unconstrained: bool = False):
    graph_cfg    = (meta or {}).get('graph', {})
    bbox         = graph_cfg.get('bbox', _DEFAULT_BBOX)
    
    # Simple hash of bbox for caching
    bbox_hash = hashlib.md5(str(bbox).encode()).hexdigest()[:8]
    cache_dir   = 'cache'
    pkl_file    = os.path.join(cache_dir, f"graph_{bbox_hash}.pkl")
    cache_file  = os.path.join(cache_dir, f"graph_{bbox_hash}.graphml")
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(pkl_file):
        import pickle
        print(f"Loading cached road network (pickle): {pkl_file}")
        with open(pkl_file, 'rb') as fh:
            G = pickle.load(fh)
    elif os.path.exists(cache_file):
        print(f"Loading cached road network: {cache_file}")
        G = ox.load_graphml(cache_file)
        # Save as pickle for faster future loads
        import pickle
        print(f"Saving pickle cache for faster future loads...")
        with open(pkl_file, 'wb') as fh:
            pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Downloading road network...")
        north, south, east, west = bbox[3], bbox[1], bbox[2], bbox[0]
        # OSMnx 2.0+ expects a single tuple (north, south, east, west)
        G = ox.graph_from_bbox((north, south, east, west), network_type='drive')
        ox.save_graphml(G, cache_file)
        import pickle
        with open(pkl_file, 'wb') as fh:
            pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)

    road_cfg     = _load_road_speeds((meta or {}).get('road_speeds'))
    road_types   = road_cfg['road_types']
    default_spd  = road_cfg.get('default_speed_kph', 30)

    print("Applying road speed config...")
    for u, v, k, data in G.edges(keys=True, data=True):
        maxspeed = data.get('maxspeed', default_spd)
        if isinstance(maxspeed, list):
            try:    base_speed = float(maxspeed[0])
            except: base_speed = default_spd
        else:
            try:    base_speed = float(maxspeed)
            except: base_speed = default_spd
        highway = data.get('highway', 'unclassified')
        if isinstance(highway, list): highway = highway[0]
        cfg = road_types.get(highway, road_types.get('default', {'speed_multiplier': 0.2, 'safe_to_cross': True}))
        data['speed_kph']       = base_speed * cfg['speed_multiplier']
        data['is_safe_to_cross'] = True if unconstrained else cfg['safe_to_cross']
        meters_per_min = (data['speed_kph'] * 1000) / 60
        data['travel_time'] = data['length'] / meters_per_min
    print("Adding edge bearings for turn-penalty calculations...")
    G = ox.bearing.add_edge_bearings(G)
    print(f"Graph ready: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")
    return G


def setup_walk_graph(meta: dict = None, center: tuple = None, radius_m: float = None):
    """Build/load a walking graph (pedestrian network).

    This is cached separately from the drive graph so it can be reused
    across runs and used for walk-path visualization or walking BFS.
    """
    graph_cfg = (meta or {}).get('graph', {})
    bbox = graph_cfg.get('bbox', _DEFAULT_BBOX)
    if center is not None and radius_m is not None:
        cache_seed = f"center={center}|radius={int(radius_m)}"
    else:
        cache_seed = str(bbox)
    bbox_hash = hashlib.md5(cache_seed.encode()).hexdigest()[:8]
    cache_dir = 'cache'
    pkl_file = os.path.join(cache_dir, f"graph_walk_{bbox_hash}.pkl")
    cache_file = os.path.join(cache_dir, f"graph_walk_{bbox_hash}.graphml")
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(pkl_file):
        import pickle
        print(f"Loading cached walking network (pickle): {pkl_file}")
        with open(pkl_file, 'rb') as fh:
            G_walk = pickle.load(fh)
    elif os.path.exists(cache_file):
        print(f"Loading cached walking network: {cache_file}")
        G_walk = ox.load_graphml(cache_file)
        import pickle
        print("Saving walk pickle cache for faster future loads...")
        with open(pkl_file, 'wb') as fh:
            pickle.dump(G_walk, fh, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Downloading walking network...")
        if center is not None and radius_m is not None:
            # Radius-based walk graph is much smaller than full-bbox graph.
            G_walk = ox.graph_from_point(center, dist=radius_m, network_type='walk', simplify=True)
        else:
            north, south, east, west = bbox[3], bbox[1], bbox[2], bbox[0]
            G_walk = ox.graph_from_bbox((north, south, east, west), network_type='walk')
        ox.save_graphml(G_walk, cache_file)
        import pickle
        with open(pkl_file, 'wb') as fh:
            pickle.dump(G_walk, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Walk graph ready: {G_walk.number_of_nodes()} nodes, {G_walk.number_of_edges()} edges\n")
    return G_walk

# ============================================================================
# MATRIX PRECOMPUTATION
# ============================================================================

_LAST_MATRIX_PRECOMPUTE_STATS = {}


def get_last_matrix_precompute_stats():
    return dict(_LAST_MATRIX_PRECOMPUTE_STATS)


def _set_last_matrix_precompute_stats(stats):
    global _LAST_MATRIX_PRECOMPUTE_STATS
    _LAST_MATRIX_PRECOMPUTE_STATS = dict(stats or {})

def precompute_matrix(students, routes, G, fast_mode=None, G_drive=None,
                      max_candidates=15, matrix_cache_pkl_path=None,
                      matrix_cache_min_finite_ratio=0.0001):
    """Build the distance matrix for ALNS.

    Parameters
    ----------
    G       : graph used for walking BFS (may be constrained)
    G_drive : graph used for bus driving distances (should always be the
              full unconstrained network).  Falls back to *G* if not given,
              preserving backward-compatibility.
    max_candidates : int
        Include up to this many walk-reachable candidate nodes per student
        in the precomputed matrix.  Should match or exceed
        ``max_candidates_per_student`` used by the ALNS engine (default 15)
        so that insertion-cost checks never fall back to A*.
    """
    if G_drive is None:
        G_drive = G
    print("[Optimization] Preparing distance matrix...")
    _t_start = _t.time()
    _stats = {
        "source": "computed_osrm",
        "loaded_from_pkl": False,
        "saved_to_pkl": False,
        "matrix_cache_pkl_path": matrix_cache_pkl_path,
        "load_time_s": 0.0,
        "compute_time_s": 0.0,
        "save_time_s": 0.0,
        "total_time_s": 0.0,
        "critical_nodes_count": 0,
        "matrix_nodes_count": 0,
        "cache_key_prefix": None,
        "cache_entry_node_count": None,
        "cache_loaded_finite_ratio": None,
        "cache_loaded_inf_pairs": None,
        "cache_loaded_finite_pairs": None,
        "cache_guard_min_finite_ratio": float(matrix_cache_min_finite_ratio),
        "cache_load_rejected": False,
        "cache_load_reject_reason": None,
        "cache_recomputed_after_reject": False,
    }
    critical_nodes = set()
    student_frontages = {}
    # Collect ALL candidate nodes ALNS will actually use so the precomputed
    # matrix covers every node the optimizer can insert at.  The old [:5]
    # limit caused massive A* fallback spikes on 600K-node graphs.
    for s in students:
        node_id, _ = snap_address_to_edge(s.coords, G)
        critical_nodes.add(node_id)
        student_frontages[s.id] = node_id
        if s.walk_radius > 0:
            walk_g = _get_walk_graph(G)  # Use walk graph with crossings if available
            safe_nodes = find_safe_nodes_within_radius(s.coords, G, 500, s.walk_radius, walk_graph=walk_g)
            for safe_node_id, _ in safe_nodes[:max_candidates]:
                critical_nodes.add(safe_node_id)
    school_node = None
    for route in routes:
        for stop in route.stops:
            if stop.node_id in G_drive:
                critical_nodes.add(stop.node_id)
                if school_node is None: school_node = stop.node_id
            else:
                nearest = _det_eng.fast_nearest_node(G_drive, stop.coords[1], stop.coords[0])
                stop.node_id = nearest
                stop.coords = (G_drive.nodes[nearest]['y'], G_drive.nodes[nearest]['x'])
                critical_nodes.add(nearest)
                if school_node is None: school_node = nearest
    # Auto-select fast mode for large graphs (>50K nodes) to avoid minutes-long precomputes
    if fast_mode is None:
        fast_mode = G_drive.number_of_nodes() > 50_000
    _stats["critical_nodes_count"] = len(critical_nodes)

    # Persistent matrix cache (optional): load full sub-matrix for this exact
    # critical-node set and driving graph snapshot.
    matrix_nodes = sorted(critical_nodes, key=lambda x: str(x))
    _stats["matrix_nodes_count"] = len(matrix_nodes)
    if matrix_cache_pkl_path:
        cache_key = _build_matrix_cache_key(G_drive, matrix_nodes)
        _stats["cache_key_prefix"] = cache_key[:12]
        _tl = _t.time()
        loaded = _load_matrix_cache_from_disk(matrix_cache_pkl_path, cache_key)
        _stats["load_time_s"] = round(_t.time() - _tl, 4)
        if loaded:
            _stats["source"] = "loaded_from_pkl"
            _stats["loaded_from_pkl"] = True
            if isinstance(loaded, dict):
                _stats["cache_entry_node_count"] = loaded.get("node_count")
                _stats["cache_loaded_finite_ratio"] = loaded.get("finite_ratio")
                _stats["cache_loaded_inf_pairs"] = loaded.get("inf_pairs")
                _stats["cache_loaded_finite_pairs"] = loaded.get("finite_pairs")

            finite_ratio = loaded.get("finite_ratio") if isinstance(loaded, dict) else None
            finite_pairs = loaded.get("finite_pairs") if isinstance(loaded, dict) else None
            reject_reason = None

            if finite_pairs == 0:
                reject_reason = "loaded cache has zero finite pairs (all inf)"
            elif (
                finite_ratio is not None
                and float(finite_ratio) < float(matrix_cache_min_finite_ratio)
            ):
                reject_reason = (
                    f"loaded cache finite_ratio={finite_ratio} below minimum "
                    f"{float(matrix_cache_min_finite_ratio):.6f}"
                )

            if reject_reason:
                _stats["cache_load_rejected"] = True
                _stats["cache_load_reject_reason"] = reject_reason
                _stats["cache_recomputed_after_reject"] = True
                _stats["loaded_from_pkl"] = False
                _stats["source"] = "computed_osrm"
                print(
                    "[Optimization] Warning: rejecting persisted matrix cache "
                    f"({reject_reason}). Recomputing via OSRM."
                )
                # Defensive clear: avoid stale/poisoned pairs affecting this solve.
                _MATRIX_CACHE.clear()
                _MATRIX_CACHE_LENGTH.clear()
            else:
                _stats["total_time_s"] = round(_t.time() - _t_start, 4)
                _set_last_matrix_precompute_stats(_stats)
                print(
                    "[Optimization] Loaded persisted matrix cache from: "
                    f"{matrix_cache_pkl_path} "
                    f"(finite_ratio={finite_ratio}, finite_pairs={finite_pairs})"
                )
                return critical_nodes, student_frontages
    # Bus distance matrix ALWAYS uses the full driving graph
    # precalculate_distance_matrix(G_drive, list(critical_nodes), fast_mode=fast_mode)
    
    # OSRM-based precomputation: much faster on large graphs, but requires a local OSRM instance running with the same graph data.  Falls back to in-memory if OSRM fails for any reason (e.g. not running, different graph, etc.) — in that case a warning is printed and the function behaves like the old version, precomputing only the critical nodes with in-memory Dijkstra.
    from detour_engine import precalculate_distance_matrix_osrm
    _tc = _t.time()
    precalculate_distance_matrix_osrm(G_drive, list(critical_nodes))
    _stats["compute_time_s"] = round(_t.time() - _tc, 4)

    if matrix_cache_pkl_path:
        cache_key = _build_matrix_cache_key(G_drive, matrix_nodes)
        _ts = _t.time()
        _stats["saved_to_pkl"] = _save_matrix_cache_to_disk(matrix_cache_pkl_path, cache_key, matrix_nodes)
        _stats["save_time_s"] = round(_t.time() - _ts, 4)

    _stats["total_time_s"] = round(_t.time() - _t_start, 4)
    _set_last_matrix_precompute_stats(_stats)
    return critical_nodes, student_frontages


def _build_matrix_cache_key(graph, node_ids):
    graph_sig = f"n={graph.number_of_nodes()}|e={graph.number_of_edges()}|k={len(node_ids)}"
    node_sig = "|".join(str(n) for n in node_ids)
    return hashlib.sha1(f"{graph_sig}|{node_sig}".encode("utf-8")).hexdigest()


def _load_matrix_cache_from_disk(pkl_path, cache_key):
    try:
        if not pkl_path or not os.path.exists(pkl_path):
            return False
        with open(pkl_path, "rb") as fh:
            payload = pickle.load(fh)
        if not isinstance(payload, dict):
            return False
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            return False
        entry = entries.get(cache_key)
        if not isinstance(entry, dict):
            return False
        node_ids = entry.get("node_ids")
        durations = entry.get("durations_min")
        distances = entry.get("distances_m")
        if not (isinstance(node_ids, list) and isinstance(durations, list) and isinstance(distances, list)):
            return False

        size = len(node_ids)
        if size == 0 or len(durations) != size or len(distances) != size:
            return False

        finite_pairs = 0
        inf_pairs = 0

        for i, src in enumerate(node_ids):
            row_t = durations[i]
            row_d = distances[i]
            if not (isinstance(row_t, list) and isinstance(row_d, list)):
                return False
            if len(row_t) != size or len(row_d) != size:
                return False
            for j, dst in enumerate(node_ids):
                if src == dst:
                    continue
                t_val = row_t[j]
                d_val = row_d[j]
                _MATRIX_CACHE[(src, dst)] = t_val
                _MATRIX_CACHE_LENGTH[(src, dst)] = d_val
                if t_val == float("inf") or d_val == float("inf"):
                    inf_pairs += 1
                else:
                    finite_pairs += 1

        total_pairs = finite_pairs + inf_pairs
        finite_ratio = round((finite_pairs / total_pairs), 4) if total_pairs > 0 else None
        return {
            "node_count": size,
            "finite_pairs": finite_pairs,
            "inf_pairs": inf_pairs,
            "finite_ratio": finite_ratio,
        }
    except Exception as e:
        print(f"[Optimization] Warning: failed to load matrix cache '{pkl_path}': {e}")
        return False


def _save_matrix_cache_to_disk(pkl_path, cache_key, node_ids):
    try:
        if not pkl_path:
            return False
        folder = os.path.dirname(pkl_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        size = len(node_ids)
        durations = []
        distances = []
        for src in node_ids:
            row_t = []
            row_d = []
            for dst in node_ids:
                if src == dst:
                    row_t.append(0.0)
                    row_d.append(0.0)
                else:
                    row_t.append(_MATRIX_CACHE.get((src, dst), float("inf")))
                    row_d.append(_MATRIX_CACHE_LENGTH.get((src, dst), float("inf")))
            durations.append(row_t)
            distances.append(row_d)

        payload = {"version": 1, "entries": {}}
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, "rb") as fh:
                    existing = pickle.load(fh)
                if isinstance(existing, dict):
                    payload = existing
                    payload.setdefault("entries", {})
            except Exception:
                pass

        payload["entries"][cache_key] = {
            "node_ids": node_ids,
            "durations_min": durations,
            "distances_m": distances,
            "created_unix": _t.time(),
            "size": size,
        }

        with open(pkl_path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Optimization] Saved matrix cache to: {pkl_path}")
        return True
    except Exception as e:
        print(f"[Optimization] Warning: failed to save matrix cache '{pkl_path}': {e}")
        return False

# ============================================================================
# MODE 1: generate_routes
# ============================================================================

def run_generate_routes(data, G, input_file_path):
    _run_start = _t.time()
    students, buses, routes, school_coords, constraints, algo_config = load_mode1_input(data, G)
    print_input_summary(students, buses, routes, school_coords)
    precompute_matrix(students, routes, G)
    print(f"\nRUNNING ALNS OPTIMIZATION ({algo_config.get('iterations', 60)} iters)")
    initial_sol = ServiceSolution(students, routes, G)
    optimizer = ALNSEngine(initial_sol, iterations=algo_config.get('iterations', 60))
    _alns_start = _t.time()
    best_sol = optimizer.run()
    _alns_elapsed = _t.time() - _alns_start
    for r in best_sol.routes:
        r.total_distance = calculate_route_distance(r, G)
        r.total_time = calculate_route_time(r, G)
    routes_with_students = [r for r in best_sol.routes if r.get_student_count() > 0]
    if routes_with_students:
        create_route_map(G, routes_with_students, all_students=best_sol.students,
                         school_coords=school_coords, output_file='route_map.html',
                         solution=best_sol, show_crossing_usage=True)
    unserved = [s for s in best_sol.students if not s.is_served]
    output = serialize_routes(best_sol.routes, buses, school_coords, unserved, G)
    output["meta"] = {
        "students_served": len(best_sol.students) - len(unserved),
        "total_time_minutes": sum(r.total_time for r in best_sol.routes),
        "objective": round(best_sol.calculate_objective(), 2)
    }
    with open('output_data.json', 'w') as f: json.dump(output, f, indent=2)
    _te = _t.time() - _run_start
    report = {
        "mode": "generate_routes", "input_file": input_file_path,
        "total_runtime_seconds": round(_te, 2), "optimization_time_seconds": round(_alns_elapsed, 2),
        "students_total": len(best_sol.students), "students_served": len(best_sol.students) - len(unserved),
        "routes_created": len(routes_with_students), "final_objective": round(best_sol.calculate_objective(), 2),
    }
    save_run(data, output, report, map_files={'route_map.html': 'route_map.html'})
    return output


# ============================================================================
# CALLABLE API  (used by experiments/comparison/run_comparison.py)
# ============================================================================

def run_algorithm(data: dict, G, iterations: int = None,
                  stage_walk_limits: dict = None, save=False,
                  G_drive=None, time_budget_seconds: float = None,
                  matrix_cache_pkl_path: str = None,
                  matrix_cache_min_finite_ratio: float = 0.0001):
    """Run ALNS on *data* using graph *G* and return (best_solution, stats_dict, school_coords).

    Parameters
    ----------
    data : dict            – standard input dict  ({"meta": …, "data": …})
    G    : networkx.Graph  – graph for **walking BFS** (may be constrained)
    G_drive : networkx.Graph – graph for **bus driving** distances (should be
                               the full unconstrained network).  Falls back to
                               *G* when not given, preserving backward compat.
    iterations : int       – override ALNS iterations (None = use data["meta"]["algorithm"]["iterations"])
    stage_walk_limits : dict – override walk limits *after* students are created
                               e.g. {"KG": 0, "MIDDLE": 150, "HIGH": 200}
    save : bool            – persist run artefacts to runs_history/

    Returns
    -------
    tuple : (ServiceSolution, stats_dict, school_coords_dict)
    """
    if G_drive is None:
        G_drive = G
    import time as _time
    students, buses, routes, school_coords, constraints, algo_cfg = load_mode1_input(data, G)

    # Apply stage-specific walk limits if provided
    if stage_walk_limits:
        _stage_map = {
            "KG":         School_Stage.KG,
            "ELEMENTARY": School_Stage.ELEMENTARY,
            "MIDDLE":     School_Stage.MIDDLE,
            "HIGH":       School_Stage.HIGH,
        }
        for s in students:
            stage_name = s.school_stage.name
            if stage_name in stage_walk_limits:
                s.walk_radius = stage_walk_limits[stage_name]

    iters  = iterations or algo_cfg.get("iterations", 60)
    budget = time_budget_seconds or algo_cfg.get("time_budget_seconds", None)
    max_cands = max(6, int(algo_cfg.get("max_candidates_per_student", 15) or 15))
    early_stop_patience = algo_cfg.get("early_stop_patience", None)
    min_improvement = algo_cfg.get("early_stop_min_improvement", 1e-6)
    freeze_temp_threshold = algo_cfg.get("early_stop_freeze_temp", 0.05)
    freeze_patience = algo_cfg.get("early_stop_freeze_patience", None)
    merge_tail_iterations = algo_cfg.get("merge_tail_iterations", 30)
    worst_cost_sample_ratio = algo_cfg.get("worst_cost_sample_ratio", 0.35)
    regret_share_early = algo_cfg.get("regret_share_early", 0.4)
    regret_share_mid = algo_cfg.get("regret_share_mid", 0.3)
    regret_share_late = algo_cfg.get("regret_share_late", 0.22)
    regret_stagnation_bonus = algo_cfg.get("regret_stagnation_bonus", 0.08)
    regret_share_min = algo_cfg.get("regret_share_min", 0.15)
    regret_share_max = algo_cfg.get("regret_share_max", 0.6)
    # Walking BFS uses G (may be constrained); bus routing uses G_drive (unconstrained)
    precompute_matrix(
        students,
        routes,
        G,
        G_drive=G_drive,
        max_candidates=max_cands,
        matrix_cache_pkl_path=matrix_cache_pkl_path,
        matrix_cache_min_finite_ratio=matrix_cache_min_finite_ratio,
    )
    matrix_precompute = get_last_matrix_precompute_stats()

    initial = ServiceSolution(students, routes, G_drive)
    engine  = ALNSEngine(initial, iterations=iters, time_budget_seconds=budget,
                         max_candidates_per_student=max_cands,
                         early_stop_patience=early_stop_patience,
                         min_improvement=min_improvement,
                         freeze_temp_threshold=freeze_temp_threshold,
                         freeze_patience=freeze_patience,
                         merge_tail_iterations=merge_tail_iterations,
                         worst_cost_sample_ratio=worst_cost_sample_ratio,
                         regret_share_early=regret_share_early,
                         regret_share_mid=regret_share_mid,
                         regret_share_late=regret_share_late,
                         regret_stagnation_bonus=regret_stagnation_bonus,
                         regret_share_min=regret_share_min,
                         regret_share_max=regret_share_max)
    t0      = _time.time()
    best    = engine.run()
    elapsed = _time.time() - t0

    for r in best.routes:
        from detour_engine import (
            calculate_route_time_from_matrix,
            calculate_route_distance_from_matrix,
        )
        t = calculate_route_time_from_matrix(r.stops, G_drive)
        r.total_time = t if t is not None else 0.0
        d = calculate_route_distance_from_matrix(r.stops, G_drive)
        r.total_distance = d if d is not None else 0.0

    served = sum(1 for s in best.students if s.is_served)
    total  = len(best.students)
    active = [r for r in best.routes if r.get_student_count() > 0]
    total_time = sum(r.total_time for r in active)
    total_dist = sum(r.total_distance for r in active)

    stats = {
        "served": served, "total": total,
        "routes": len(active),
        "total_time": round(total_time, 2),
        "total_dist": round(total_dist, 2),
        "objective": round(best.calculate_objective(), 2),
        "runtime": round(elapsed, 2),
        "alns_iteration_log": list(getattr(engine, "iteration_log", [])),
        "operator_performance": dict(getattr(engine, "operator_stats_summary", {})),
        "alns_diagnostics": dict(getattr(engine, "run_diagnostics", {})),
        "insertion_debug": _alns.get_insertion_debug_stats(),
        "matrix_precompute": matrix_precompute,
    }

    return best, stats, school_coords


def find_minimum_fleet(data: dict, G, iterations: int = None,
                       stage_walk_limits: dict = None,
                       G_drive=None, time_budget_seconds: float = None,
                       matrix_cache_pkl_path: str = None,
                       matrix_cache_min_finite_ratio: float = 0.0001):
    """Search for the smallest fleet size that can serve every student.

    Iterates from the theoretical minimum number of buses (⌈students/capacity⌉)
    upward, stopping as soon as a fleet size achieves 100 % service rate.  If no
    fleet size within the available buses serves everyone, the result with the
    highest service count is kept.

    Parameters
    ----------
    data : dict   – standard input dict; ``data["data"]["buses"]`` is sliced to
                    select fleet size.
    G / G_drive   – passed through to :func:`run_algorithm`.
    iterations, stage_walk_limits, time_budget_seconds – passed through.

    Returns
    -------
    tuple : (best_k, ServiceSolution, stats_dict, school_coords)
        ``best_k`` is the minimum fleet size found.
        ``stats["buses_used"]`` is set to *best_k*.
    """
    import copy as _copy

    base_buses  = data["data"]["buses"]
    n_students  = len(data["data"]["students"])
    capacity    = base_buses[0].get("capacity", 60) if base_buses else 60
    k_max       = len(base_buses)
    # Ceiling division without math module
    k_min = max(1, -(-n_students // capacity))

    best_k, best_sol, best_stats, best_school = k_max, None, None, None

    print(f"\n[FleetSearch] {n_students} students, capacity {capacity}, "
          f"searching k={k_min}..{k_max}")

    constraints = data.get("meta", {}).get("constraints", {})
    algo_cfg = data.get("meta", {}).get("algorithm", {})
    base_budget_s = time_budget_seconds if time_budget_seconds is not None else algo_cfg.get("time_budget_seconds", None)
    first_k_budget_scale = float(algo_cfg.get("fleet_search_first_k_budget_scale", 1.0))
    followup_k_budget_scale = float(algo_cfg.get("fleet_search_followup_k_budget_scale", 0.65))
    trailing_early_stop_ratio = float(algo_cfg.get("fleet_search_trailing_early_stop_ratio", 0.9))
    trailing_min_gap = int(algo_cfg.get("fleet_search_trailing_min_served_gap", 1))
    max_per_k_s_cfg = algo_cfg.get("fleet_search_max_per_k_seconds", None)
    max_per_k_s = float(max_per_k_s_cfg) if max_per_k_s_cfg is not None else None

    fleet_log   = []
    total_fleet_search_runtime = 0.0

    for k in range(k_min, k_max + 1):
        k_budget_s = None
        if base_budget_s is not None:
            scale = first_k_budget_scale if k == k_min else followup_k_budget_scale
            scale = max(0.05, float(scale))
            k_budget_s = max(1.0, float(base_budget_s) * scale)
            if max_per_k_s is not None:
                k_budget_s = min(k_budget_s, max_per_k_s)

        trial = _copy.deepcopy(data)
        trial["data"]["buses"] = trial["data"]["buses"][:k]

        sol, stats, school = run_algorithm(
            trial, G,
            iterations=iterations,
            stage_walk_limits=stage_walk_limits,
            G_drive=G_drive,
            time_budget_seconds=k_budget_s,
            matrix_cache_pkl_path=matrix_cache_pkl_path,
            matrix_cache_min_finite_ratio=matrix_cache_min_finite_ratio,
        )

        total_fleet_search_runtime += stats.get("runtime", 0.0)

        served  = stats["served"]
        total   = stats["total"]
        capacity_k = trial["data"]["buses"][0].get("capacity", 60)

        unserved_students = [s for s in sol.students if not s.is_served]
        reasons = _diagnose_unserved(unserved_students, sol, capacity_k, constraints)

        fleet_log.append({
            "k":                k,
            "served":           served,
            "unserved":         total - served,
            "feasible":         served == total,
            "runtime_s":        stats["runtime"],
            "time_budget_s":    round(k_budget_s, 2) if k_budget_s is not None else None,
            "matrix_precompute": stats.get("matrix_precompute"),
            "rejection_reasons": reasons,
        })

        diag = "  ".join(f"{r}: {c}" for r, c in reasons.items()) if reasons else "—"
        print(f"  Fleet {k}: {served}/{total} served  [{diag}]")

        if best_sol is None or served > best_stats["served"]:
            best_k, best_sol, best_stats, best_school = k, sol, stats, school

        if served == total:
            print(f"  → All students served with {k} bus(es) — minimum found.")
            break

        # If a larger fleet size trails the best served count and has already
        # spent most of its capped budget, stop expanding k to avoid runaway time.
        if best_stats is not None and k > best_k:
            served_gap = best_stats["served"] - served
            used_most_budget = (
                k_budget_s is not None
                and float(stats.get("runtime", 0.0)) >= (float(k_budget_s) * trailing_early_stop_ratio)
            )
            if served_gap >= trailing_min_gap and used_most_budget:
                print(
                    f"  Early stop fleet search at k={k}: trailing best by {served_gap} served "
                    f"after using {stats.get('runtime', 0.0):.2f}s/{k_budget_s:.2f}s budget."
                )
                break

    best_stats["buses_used"]           = best_k
    best_stats["total_fleet_search_runtime"] = total_fleet_search_runtime
    best_stats["fleet_search_log"]     = fleet_log
    best_stats["fleet_search_summary"] = _summarise_fleet_search(fleet_log)
    best_stats["fleet_search_budget_policy"] = {
        "base_time_budget_seconds": base_budget_s,
        "first_k_budget_scale": first_k_budget_scale,
        "followup_k_budget_scale": followup_k_budget_scale,
        "max_per_k_seconds": max_per_k_s,
        "trailing_early_stop_ratio": trailing_early_stop_ratio,
        "trailing_min_served_gap": trailing_min_gap,
    }
    return best_k, best_sol, best_stats, best_school


def _diagnose_unserved(unserved_students, sol, capacity, constraints):
    """Categorise why each unserved student wasn't placed.
    Returns {reason_key: count} with zero-count keys omitted.
    """
    import math as _math
    if not unserved_students:
        return {}

    con     = constraints or {}
    enabled = bool(con.get("enabled", True))
    k_mult  = float(con.get("ride_time_multiplier", 2.5))
    fl      = float(con.get("floor_minutes", 45))
    ce      = float(con.get("ceiling_minutes", 60))

    all_full = all(r.get_student_count() >= capacity for r in sol.routes)

    reasons = {}

    school_nodes = []
    for route in sol.routes:
        if route.stops:
            school_nodes.append(route.stops[0].node_id)

    def _diagnose_zero_walk_student(student):
        try:
            frontage_node_id, frontage_coords = snap_address_to_edge(student.coords, sol.graph)
        except Exception:
            return "zero_walk_radius_snap_failed"

        # If frontage cannot reach any school node in matrix, ALNS won't be able to insert.
        if school_nodes:
            reachable = False
            for school_node in school_nodes:
                to_school = _MATRIX_CACHE.get((frontage_node_id, school_node), float("inf"))
                from_school = _MATRIX_CACHE.get((school_node, frontage_node_id), float("inf"))
                if to_school < float("inf") and from_school < float("inf"):
                    reachable = True
                    break
            if not reachable:
                return "zero_walk_radius_frontage_unreachable"

        # Probe insertion feasibility directly: this is diagnostic-only, not expensive at tiny unserved counts.
        try:
            for route in sol.routes:
                options = _alns._get_insertions_for_route(
                    student, route, sol.graph, (frontage_node_id, frontage_coords)
                )
                if options:
                    return "search_budget_exhausted"
            return "zero_walk_radius_no_valid_insertion"
        except Exception:
            return "zero_walk_radius_diagnostic_error"

    for s in unserved_students:
        if all_full:
            reasons["all_routes_at_capacity"] = reasons.get("all_routes_at_capacity", 0) + 1
            continue

        if enabled:
            dt = getattr(s, "direct_time_to_school", None)
            if dt is not None and _math.isfinite(dt) and dt > 0:
                cap = max(fl, min(k_mult * dt, dt + ce))
                if cap < 20:
                    reasons["ride_time_cap_too_tight"] = reasons.get("ride_time_cap_too_tight", 0) + 1
                    continue

        if getattr(s, "walk_radius", 0) == 0:
            z_reason = _diagnose_zero_walk_student(s)
            reasons[z_reason] = reasons.get(z_reason, 0) + 1
            continue

        reasons["search_budget_exhausted"] = reasons.get("search_budget_exhausted", 0) + 1

    return {k: v for k, v in reasons.items() if v > 0}


def _summarise_fleet_search(fleet_log):
    """Human-readable explanation of the fleet search outcome."""
    if not fleet_log:
        return "no search performed"

    feasible = [e for e in fleet_log if e["feasible"]]
    if feasible:
        k = feasible[0]["k"]
        if len(fleet_log) == 1 and fleet_log[0]["feasible"]:
            return f"k={k} is the theoretical minimum and already serves all students"
        return f"k={k} is the minimum feasible fleet size"

    last  = max(fleet_log, key=lambda e: e["served"])
    parts = [
        f"No fleet size in range {fleet_log[0]['k']}..{fleet_log[-1]['k']} served all students. "
        f"Best: k={last['k']} with {last['served']}/{last['served'] + last['unserved']} served, "
        f"{last['unserved']} unserved."
    ]
    reasons = last.get("rejection_reasons", {})
    if reasons.get("ride_time_cap_too_tight"):
        n = reasons["ride_time_cap_too_tight"]
        parts.append(
            f"{n} student(s) have ride-time caps too tight to fit into any multi-stop route — "
            f"early-boarding students accumulate too much ride time at this fleet size."
        )
    if reasons.get("all_routes_at_capacity"):
        n = reasons["all_routes_at_capacity"]
        parts.append(
            f"{n} student(s) could not be placed because all routes were at seating capacity."
        )
    if reasons.get("zero_walk_radius_no_candidates"):
        n = reasons["zero_walk_radius_no_candidates"]
        parts.append(
            f"{n} student(s) have walk_radius=0 with no candidate stop found."
        )
    if reasons.get("zero_walk_radius_frontage_unreachable"):
        n = reasons["zero_walk_radius_frontage_unreachable"]
        parts.append(
            f"{n} student(s) have walk_radius=0 and frontage node is unreachable from school in the matrix cache."
        )
    if reasons.get("zero_walk_radius_no_valid_insertion"):
        n = reasons["zero_walk_radius_no_valid_insertion"]
        parts.append(
            f"{n} student(s) have walk_radius=0 but no valid insertion was found under current constraints/pruning."
        )
    if reasons.get("zero_walk_radius_snap_failed"):
        n = reasons["zero_walk_radius_snap_failed"]
        parts.append(
            f"{n} student(s) with walk_radius=0 failed frontage-node snapping during diagnostics."
        )
    if reasons.get("search_budget_exhausted"):
        n = reasons["search_budget_exhausted"]
        parts.append(
            f"{n} student(s) likely unserved due to ALNS budget exhaustion — "
            f"try increasing time_budget_seconds."
        )
    return " ".join(parts)


# ============================================================================
# MODE 2: change_location
# ============================================================================

def run_change_location(data, G, input_file_path):
    _run_start = _t.time()
    (student_id, new_coords, change_type, valid_from, valid_until,
     algo_config, routes, all_students, buses, school_coords) = load_mode2_input(data, G)
    method = algo_config.get('method', 'cheapest_insertion')
    daily_budget = data.get('constraints', {}).get('daily_detour_budget_minutes', 5)
    target_student = next((s for s in all_students if s.id == student_id), None)
    if target_student and target_student.is_served:
        if target_student.assigned_stop: target_student.assigned_stop.remove_student(target_student)
    if not target_student:
        from data_loader import school_stage_from_string
        new_loc = data.get('new_location', {})
        target_student = Student(id=student_id, lat=new_coords[0], lon=new_coords[1],
            age=new_loc.get('age', 10), school_stage=school_stage_from_string(new_loc.get('school_stage', 'ELEMENTARY')),
            fee=new_loc.get('fee', 100), assignment=change_type, valid_from=valid_from, valid_until=valid_until)
    else:
        target_student.coords = new_coords
        target_student.assignment = change_type
    precompute_matrix([target_student], routes, G)
    if method == '2opt': success, updated_route, message = insert_with_2opt(target_student, routes, G, change_type, daily_budget)
    elif method == 'alns':
        if target_student not in all_students: all_students.append(target_student)
        optimizer = ALNSEngine(ServiceSolution(all_students, routes, G), iterations=algo_config.get('iterations', 30))
        best_sol = optimizer.run()
        routes = best_sol.routes
        target = next((s for s in best_sol.students if s.id == student_id), None)
        success = target and target.is_served
        message = f"ALNS: {'placed' if success else 'failed'}"
        updated_route = next((r for r in routes if any(any(st.id == student_id for st in stp.students) for stp in r.stops)), None)
    else: success, updated_route, message = process_detour_request(target_student, routes, G, change_type, daily_budget)
    for r in routes: r.total_distance = calculate_route_distance(r, G); r.total_time = calculate_route_time(r, G)
    if os.path.exists('route_map.html'): shutil.copy2('route_map.html', 'route_map_old.html')
    if success:
        create_route_map(G, [r for r in routes if r.get_student_count() > 0], all_students=all_students,
                         school_coords=school_coords, output_file='route_map_new.html',
                         solution=None, show_crossing_usage=False)
    unserved = [s for s in all_students if not s.is_served]
    output = serialize_routes(routes, buses, school_coords, unserved, G)
    if not success: output = {"status": "failed", "student_id": student_id, "reason": message}
    with open('output_data.json', 'w') as f: json.dump(output, f, indent=2)
    report = {"mode": "change_location", "status": output.get('status', 'success'), "students_total": len(all_students)}
    save_run(data, output, report, map_files={'route_map_old.html': 'route_map_old.html', 'route_map_new.html': 'route_map_new.html'})
    return output

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Safety-Aware Bus Optimization")
    parser.add_argument('input', nargs='?', default='api_requests/generate_routes_input.json', help="Input JSON")
    parser.add_argument('--unconstrained', action='store_true', help="Disable safety constraints")
    parser.add_argument('--iterations', type=int, default=None, help="Override ALNS iters")
    args = parser.parse_args()
    data = load_json(args.input)
    if args.unconstrained:
        if 'data' in data and 'students' in data['data']:
            for s in data['data']['students']: s['walk_radius_override'] = 400
        if 'meta' in data:
            if 'constraints' not in data['meta']: data['meta']['constraints'] = {}
            data['meta']['constraints'].update({"ride_time_multiplier": 999, "floor_minutes": 999, "ceiling_minutes": 999})
    if args.iterations: data['meta'].setdefault('algorithm', {})['iterations'] = args.iterations
    G = setup_graph(data['meta'], unconstrained=args.unconstrained)
    if data['meta']['mode'] == 'generate_routes': run_generate_routes(data, G, args.input)
    else: run_change_location(data, G, args.input)
