import random
import math
import time
import numpy as np
import copy
from detour_engine import (
    cheapest_insertion, 
    calculate_route_time, 
    calculate_route_distance,
    calculate_stops_time,
    calculate_insertion_cost,
    validate_permanent_student,
    snap_address_to_edge,
    calculate_route_time_from_matrix,
    calculate_route_distance_from_matrix,
    calculate_walk_penalty,
    get_walk_absolute_max
)
from entities import Stop

# ============================================================================
# DESTROY OPERATORS
# ============================================================================

def random_removal(solution, n):
    """Removes n random students from the solution."""
    served_students = [s for s in solution.students if s.is_served]
    n = min(n, len(served_students))
    if n == 0:
        return []
    
    removed = random.sample(served_students, n)
    for student in removed:
        _remove_student_from_solution(solution, student)
    return removed

def worst_cost_removal(solution, n):
    """Removes students who add the most travel time to their routes."""
    served_students = [s for s in solution.students if s.is_served]
    n = min(n, len(served_students))
    if n == 0:
        return []

    removal_candidates = []
    for student in served_students:
        stop = student.assigned_stop
        route = next((r for r in solution.routes if stop in r.stops), None)
        
        if not route:
            continue
            
        old_time = route.total_time
        
        # Calculate time saved if this student is removed
        temp_stops = list(route.stops)
        # Find the correct stop in temp_stops (since they were cloned)
        current_stop = next((s for s in temp_stops if s.node_id == stop.node_id), None)
        
        if current_stop:
            # We want to know the time of the route WITHOUT this specific student
            # If they are the only student at the stop, the stop is removed
            if len(current_stop.students) <= 1:
                temp_stops.remove(current_stop)
            
            # Use fast matrix lookup (lazy: computes on cache miss)
            new_time = calculate_route_time_from_matrix(temp_stops, solution.graph)
            if new_time is None:
                new_time = calculate_stops_time(temp_stops, solution.graph)
            delta_time = old_time - new_time
            removal_candidates.append((student, delta_time))
    
    # Sort by time saved (highest first)
    removal_candidates.sort(key=lambda x: x[1], reverse=True)
    removed = [c[0] for c in removal_candidates[:n]]
    
    for student in removed:
        _remove_student_from_solution(solution, student)
    return removed


def _haversine_m(a, b):
    lat1, lon1 = a
    lat2, lon2 = b
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    x = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2.0) ** 2
    return r * (2.0 * math.atan2(math.sqrt(x), math.sqrt(max(1e-12, 1.0 - x))))


def _build_student_route_map(solution):
    out = {}
    for route in solution.routes:
        for stop in route.stops:
            if stop.stop_type == 'school':
                continue
            for student in stop.students:
                out[student.id] = route.route_id
    return out


def shaw_related_removal(solution, n):
    """R13-style removal: pick a seed student and remove related students.

    Relatedness combines pickup proximity, same-route affinity, and stage similarity.
    """
    served_students = [s for s in solution.students if s.is_served and s.assigned_stop is not None]
    n = min(n, len(served_students))
    if n == 0:
        return []

    route_of = _build_student_route_map(solution)
    seed = random.choice(served_students)
    seed_route = route_of.get(seed.id)
    seed_stage = getattr(seed.school_stage, 'name', str(seed.school_stage))
    seed_xy = seed.assigned_stop.coords

    ranked = []
    for s in served_students:
        if s.id == seed.id or s.assigned_stop is None:
            continue
        dist = _haversine_m(seed_xy, s.assigned_stop.coords)
        same_route_bonus = 250.0 if route_of.get(s.id) == seed_route else 0.0
        stage_bonus = 50.0 if getattr(s.school_stage, 'name', str(s.school_stage)) == seed_stage else 0.0
        relatedness = dist - same_route_bonus - stage_bonus
        ranked.append((relatedness, s))

    ranked.sort(key=lambda x: x[0])
    removed = [seed]
    for _, s in ranked:
        if len(removed) >= n:
            break
        removed.append(s)

    for student in removed:
        _remove_student_from_solution(solution, student)
    return removed


def sequence_removal(solution, n):
    """Sequence-based removal: remove students from a contiguous stop subsequence."""
    active_routes = [r for r in solution.routes if r.get_student_count() > 0]
    if not active_routes or n <= 0:
        return []

    removed = []
    attempts = 0
    max_attempts = max(3, len(active_routes) * 2)

    while len(removed) < n and attempts < max_attempts:
        attempts += 1
        route = random.choice(active_routes)
        pickup_stops = [s for s in route.stops if s.stop_type != 'school' and len(s.students) > 0]
        if not pickup_stops:
            continue

        max_seq = min(len(pickup_stops), max(1, n - len(removed)))
        seq_len = random.randint(1, max_seq)
        start = random.randint(0, len(pickup_stops) - seq_len)
        seq = pickup_stops[start:start + seq_len]

        for stop in seq:
            for student in list(stop.students):
                if len(removed) >= n:
                    break
                _remove_student_from_solution(solution, student)
                removed.append(student)
            if len(removed) >= n:
                break

    return removed

def route_merge_removal(solution, n):
    """Empties the least-populated active route so ALNS must consolidate
    its students into the remaining routes.  Drives fleet reduction without
    changing the number of Route objects in the solution."""
    active = [r for r in solution.routes
              if r.get_student_count() > 0]
    if len(active) < 2:
        return []
    # Target the route with the fewest students (easiest to absorb)
    active.sort(key=lambda r: r.get_student_count())
    target = active[0]
    removed = []
    for stop in list(target.stops):
        if stop.stop_type == 'school':
            continue
        for student in list(stop.students):
            _remove_student_from_solution(solution, student)
            removed.append(student)
    return removed

def _remove_student_from_solution(solution, student):
    """Helper to safely decouple student from stop and route."""
    stop = student.assigned_stop
    if not stop: return
    
    route = next((r for r in solution.routes if stop in r.stops), None)
    if not route: return
        
    stop.remove_student(student)
    
    # If stop becomes empty, remove it from the route entirely
    if len(stop.students) == 0:
        route.stops.remove(stop)
        
    # Update route metrics after removal (lazy matrix lookup)
    fast_time = calculate_route_time_from_matrix(route.stops, solution.graph)
    route.total_time = fast_time if fast_time is not None else float('inf')
    fast_dist = calculate_route_distance_from_matrix(route.stops, solution.graph)
    route.total_distance = fast_dist if fast_dist is not None else calculate_route_distance(route, solution.graph)


# ============================================================================
# REPAIR OPERATORS
# ============================================================================

# def greedy_repair(solution):
#     """Inserts all unassigned students using the cheapest available insertion point."""
#     unassigned = [s for s in solution.students if not s.is_served]
#     random.shuffle(unassigned) # Shuffle to provide variation across calls
    
#     for student in unassigned:
#         # Reuse existing logic from detour_engine
#         result, _ = cheapest_insertion(student, solution.routes, solution.graph, detour_type='permanent')
#         if result:
#             _apply_insertion(solution, student, result)


def random_order_best_repair(solution):
    """I5-style insertion: random customer order, best insertion position."""
    unassigned = [s for s in solution.students if not s.is_served]
    if not unassigned:
        return

    random.shuffle(unassigned)

    student_frontages = {}
    for s in unassigned:
        student_frontages[s.id] = snap_address_to_edge(s.coords, solution.graph)

    for student in unassigned:
        all_options = []
        for route in solution.routes:
            all_options.extend(
                _get_insertions_for_route(student, route, solution.graph, student_frontages[student.id])
            )
        if not all_options:
            continue
        best = min(all_options, key=lambda x: x['insertion_cost_minutes'])
        _apply_insertion(solution, student, best)


def greedy_repair(solution):
    """Backwards-compatible alias for the random-order best-position insertion."""
    random_order_best_repair(solution)

def regret_repair(solution, k=2):
    """Inserts students with the highest 'regret' cost between best and k-best options.
    Optimized to minimize redundant calculations.
    """
    unassigned = [s for s in solution.students if not s.is_served]
    if not unassigned:
        return

    # 1. Pre-calculate frontage nodes to avoid repeated snapping logic
    student_frontages = {}
    for s in unassigned:
        node_id, coords = snap_address_to_edge(s.coords, solution.graph)
        student_frontages[s.id] = (node_id, coords)

    # 2. Initial calculation for all unassigned students
    # student_route_options[student_id][route_id] = list of insertions
    student_route_options = {}
    for s in unassigned:
        student_route_options[s.id] = {}
        for route in solution.routes:
            student_route_options[s.id][route.route_id] = _get_insertions_for_route(
                s, route, solution.graph, student_frontages[s.id]
            )

    while unassigned:
        best_regret = -1
        target_student = None
        target_insertion = None
        
        for student in unassigned:
            # Flatten all valid options across all routes
            all_options = []
            for r_id in student_route_options[student.id]:
                all_options.extend(student_route_options[student.id][r_id])
            
            if not all_options:
                continue
            
            all_options.sort(key=lambda x: x['insertion_cost_minutes'])
            
            # Regret calculation
            if len(all_options) >= k:
                regret = all_options[k-1]['insertion_cost_minutes'] - all_options[0]['insertion_cost_minutes']
            else:
                regret = 2000 - all_options[0]['insertion_cost_minutes']
                
            if regret > best_regret:
                best_regret = regret
                target_student = student
                target_insertion = all_options[0]
        
        if target_student and target_insertion:
            # Apply insertion
            affected_route = target_insertion['route']
            _apply_insertion(solution, target_student, target_insertion)
            
            # Remove from unassigned
            unassigned.remove(target_student)
            
            # Update only the affected route's options for all remaining unassigned students
            for s in unassigned:
                student_route_options[s.id][affected_route.route_id] = _get_insertions_for_route(
                    s, affected_route, solution.graph, student_frontages[s.id]
                )
        else:
            break

# Cache candidate nodes per student (BFS + safe_nodes don't change between iterations)
_student_candidate_cache = {}  # student_id -> list of (node_id, coords)
_student_candidate_dist   = {}  # student_id -> {node_id: walk_dist_m}  (0 for frontage)

# Candidate configuration set by ALNSEngine before each run (max_candidates_per_student, etc.)
_alns_candidate_cfg = {}


def _reorder_candidates_with_shared_boost(student_id, candidate_nodes):
    """Promote nodes that appear in candidate lists of multiple students.

    If a node is present for at least two different students, it gets a shared
    boost and is moved ahead of non-shared nodes. This encourages ALNS to
    consider common pickup nodes earlier, improving consolidation potential.
    """
    if not candidate_nodes:
        return candidate_nodes

    # Build node -> set(student_ids) from cached students, then include current.
    node_students = {}
    for sid, nodes in _student_candidate_cache.items():
        for nid, _ in nodes:
            node_students.setdefault(nid, set()).add(sid)

    for nid, _ in candidate_nodes:
        node_students.setdefault(nid, set()).add(student_id)

    popularity = {nid: len(sids) for nid, sids in node_students.items()}

    def _boost_sort(nodes):
        # Stable sort: shared first, then by popularity, then original order.
        ranked = list(enumerate(nodes))
        ranked.sort(
            key=lambda item: (
                -int(popularity.get(item[1][0], 0) >= 2),
                -popularity.get(item[1][0], 0),
                item[0],
            )
        )
        return [node for _, node in ranked]

    boosted_current = _boost_sort(candidate_nodes)

    # Also boost previously cached students that share any of these now-shared nodes.
    shared_nodes = {nid for nid, _ in candidate_nodes if popularity.get(nid, 0) >= 2}
    if shared_nodes:
        for sid, nodes in list(_student_candidate_cache.items()):
            if sid == student_id:
                continue
            if any(nid in shared_nodes for nid, _ in nodes):
                _student_candidate_cache[sid] = _boost_sort(nodes)

    return boosted_current

def _get_insertions_for_route(student, route, graph, frontage_info):
    """Helper to find all possible valid insertion points for a student in ONE route.
    Tries both the frontage node AND walk/reachability candidates.
    """
    from detour_engine import _MATRIX_CACHE
    options = []
    frontage_node_id, frontage_coords = frontage_info
    
    # Use cached candidates if available (graph doesn't change between iterations)
    if student.id in _student_candidate_cache:
        candidate_nodes = _student_candidate_cache[student.id]
    else:
        max_k = _alns_candidate_cfg.get("max_candidates_per_student", 15)
        # Build candidate list: frontage node + walk candidates (if applicable)
        candidate_nodes = [(frontage_node_id, frontage_coords)]
        dist_map = {frontage_node_id: 0.0}  # node_id -> walk distance (metres)
        
        if student.walk_radius > 0:
            from detour_engine import find_safe_nodes_within_radius, _get_walk_graph
            # Pass candidate_cfg so results are scored (intersections/arterials preferred)
            # and already returned in (-points, dist) order.
            cand_cfg = _alns_candidate_cfg if _alns_candidate_cfg else None
            walk_g = _get_walk_graph(graph)  # Use walk graph with crossings if available
            safe_nodes = find_safe_nodes_within_radius(
                student.coords, graph, 500, student.walk_radius, candidate_cfg=cand_cfg, walk_graph=walk_g
            )
            for node_id, dist in safe_nodes:  # already sorted by scoring function
                if node_id != frontage_node_id:
                    coords = (graph.nodes[node_id]['y'], graph.nodes[node_id]['x'])
                    candidate_nodes.append((node_id, coords))
                    dist_map[node_id] = float(dist)
        
        # Bus-reachable fallback: if frontage is unreachable, find nearby reachable nodes
        # via bidirectional BFS (walking ignores one-way constraints)
        school_node = route.stops[0].node_id if route.stops else None
        if school_node:
            to_school = _MATRIX_CACHE.get((frontage_node_id, school_node), float('inf'))
            from_school = _MATRIX_CACHE.get((school_node, frontage_node_id), float('inf'))
            if to_school == float('inf') or from_school == float('inf'):
                from detour_engine import fast_nearest_node
                lat, lon = student.coords
                center_node = fast_nearest_node(graph, lon, lat)
                max_walk = get_walk_absolute_max(student.walk_radius)  # Stage-based
                visited = set()
                bfs_queue = [(center_node, 0)]
                while bfs_queue and len(candidate_nodes) < max_k:
                    node, dist = bfs_queue.pop(0)
                    if node in visited or dist > max_walk:
                        continue
                    visited.add(node)
                    ts = _MATRIX_CACHE.get((node, school_node), float('inf'))
                    fs = _MATRIX_CACHE.get((school_node, node), float('inf'))
                    if ts < float('inf') and fs < float('inf'):
                        if not any(c[0] == node for c in candidate_nodes):
                            coords = (graph.nodes[node]['y'], graph.nodes[node]['x'])
                            candidate_nodes.append((node, coords))
                            dist_map[node] = float(dist)
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
        
        # Shared-node boost: if a node appears for multiple students, prioritize it.
        candidate_nodes = _reorder_candidates_with_shared_boost(student.id, candidate_nodes)
        candidate_nodes = candidate_nodes[:max_k]
        _student_candidate_cache[student.id] = candidate_nodes
        _student_candidate_dist[student.id]   = dist_map
    
    # Start and end stops are fixed (Depot/School), strictly insert between
    start_pos = 1 if len(route.stops) >= 2 else 0
    end_pos = len(route.stops) if len(route.stops) >= 2 else len(route.stops) + 1
    
    # Pre-filter: only keep candidates with KNOWN bus-reachability (both directions in cache)
    school_node = route.stops[0].node_id if route.stops else None
    reachable_candidates = []
    for cand_node_id, cand_coords in candidate_nodes:
        if school_node:
            to_s = _MATRIX_CACHE.get((cand_node_id, school_node), None)
            from_s = _MATRIX_CACHE.get((school_node, cand_node_id), None)
            # Skip if not in matrix at all (never precomputed) or known unreachable
            if to_s is None or from_s is None or to_s == float('inf') or from_s == float('inf'):
                continue
        reachable_candidates.append((cand_node_id, cand_coords))
        
    for pos in range(start_pos, end_pos):
        for cand_node_id, cand_coords in reachable_candidates:
            # Check if an existing stop at this node can be reused
            existing_stop = next((s for s in route.stops if s.node_id == cand_node_id), None)
            eval_stop = existing_stop if existing_stop else Stop(cand_node_id, cand_coords[0], cand_coords[1])
            
            res = calculate_insertion_cost(eval_stop, route, pos, graph)
            if res is None: continue
            
            cost, is_valid, _ = res
            if not is_valid: continue
            
            valid, _, _ = validate_permanent_student(eval_stop, route, pos, cost, graph,
                                                     new_student=student)
            if valid:
                # Add walk penalty to insertion cost
                walk_penalty, walk_m, over_limit = calculate_walk_penalty(
                    student, cand_node_id, graph
                )
                if walk_penalty == float('inf'):
                    continue  # Beyond absolute walk maximum
                
                penalized_cost = cost + walk_penalty
                options.append({
                    'route': route,
                    'new_stop': eval_stop,
                    'insertion_position': pos,
                    'insertion_cost_minutes': penalized_cost,
                    'is_new_stop': existing_stop is None
                })
    return options

def _get_all_valid_insertions(student, routes, graph):
    """Legacy helper (still needed for greedy_repair)"""
    node_id, coords = snap_address_to_edge(student.coords, graph)
    all_options = []
    for route in routes:
        all_options.extend(_get_insertions_for_route(student, route, graph, (node_id, coords)))
    return all_options

def _apply_insertion(solution, student, result):
    """Actually update the route and student state based on insertion search."""
    route = result['route']
    new_stop = result['new_stop']
    
    if result.get('is_new_stop', True) and new_stop not in route.stops:
        route.stops.insert(result['insertion_position'], new_stop)
    
    new_stop.add_student(student)
    # Lazy matrix lookup (computes A* on cache miss)
    fast_time = calculate_route_time_from_matrix(route.stops, solution.graph)
    route.total_time = fast_time if fast_time is not None else float('inf')  
    fast_dist = calculate_route_distance_from_matrix(route.stops, solution.graph)
    route.total_distance = fast_dist if fast_dist is not None else calculate_route_distance(route, solution.graph)


# ============================================================================
# ALNS ENGINE
# ============================================================================

class ALNSEngine:
    def __init__(self, initial_solution, iterations=100, temp=1000, cooling=0.98,
                 time_budget_seconds=None, max_candidates_per_student=None,
                 early_stop_patience=None, min_improvement=1e-6,
                 freeze_temp_threshold=0.05, freeze_patience=None):
        # Configure module-level candidate settings.
        # NOTE: do NOT clear _student_candidate_cache here — the cache is
        # keyed by student-id and stays valid across fleet-search iterations
        # (same students, same graph, same walk radii).  Clearing is handled
        # by _reset_caches() in run_comparison.py between MODES, not between
        # fleet-search k values.
        global _alns_candidate_cfg
        _alns_candidate_cfg = {}
        if max_candidates_per_student is not None:
            _alns_candidate_cfg["max_candidates_per_student"] = max_candidates_per_student

        self.curr_sol = initial_solution.clone()
        self.best_sol = initial_solution.clone()
        self.iterations = iterations
        self.temp = temp
        self.cooling = cooling
        self.time_budget_seconds = time_budget_seconds  # wall-clock budget (None = use iterations only)
        self.early_stop_patience = int(early_stop_patience) if early_stop_patience else None
        self.min_improvement = float(min_improvement) if min_improvement is not None else 1e-6
        self.freeze_temp_threshold = float(freeze_temp_threshold)
        self.freeze_patience = int(freeze_patience) if freeze_patience else None
        
        self.destroy_ops = [
            random_removal,
            worst_cost_removal,
            shaw_related_removal,
            sequence_removal,
            route_merge_removal,
        ]
        self.repair_ops = [random_order_best_repair, regret_repair]
        
        # Weights for operator selection
        self.d_weights = np.ones(len(self.destroy_ops))
        self.r_weights = np.ones(len(self.repair_ops))
        
        # Reward scores
        self.s1 = 30 # New global best
        self.s2 = 15 # Better than current
        self.s3 = 5  # Accepted (Simulated Annealing)

        # Iteration diagnostics — populated during run()
        self.iteration_log = []
        self.operator_stats = {
            "destroy": {op.__name__: self._new_op_stats() for op in self.destroy_ops},
            "repair": {op.__name__: self._new_op_stats() for op in self.repair_ops},
        }
        self.operator_stats_summary = {}

    @staticmethod
    def _new_op_stats():
        return {
            "count": 0,
            "total_time_s": 0.0,
            "min_time_s": None,
            "max_time_s": None,
        }

    def _record_op_timing(self, family, op_name, elapsed_s):
        stats = self.operator_stats.get(family, {}).get(op_name)
        if stats is None:
            return
        stats["count"] += 1
        stats["total_time_s"] += float(elapsed_s)
        if stats["min_time_s"] is None or elapsed_s < stats["min_time_s"]:
            stats["min_time_s"] = float(elapsed_s)
        if stats["max_time_s"] is None or elapsed_s > stats["max_time_s"]:
            stats["max_time_s"] = float(elapsed_s)

    def _finalize_operator_stats(self, executed_iterations):
        out = {"executed_iterations": int(executed_iterations), "destroy": {}, "repair": {}}
        denom = max(1, int(executed_iterations))
        for family in ("destroy", "repair"):
            for op_name, st in self.operator_stats.get(family, {}).items():
                count = int(st.get("count", 0))
                total = float(st.get("total_time_s", 0.0))
                avg = (total / count) if count > 0 else 0.0
                out[family][op_name] = {
                    "count": count,
                    "frequency_pct": round((count / denom) * 100.0, 2),
                    "avg_time_s": round(avg, 6),
                    "best_time_s": round(float(st.get("min_time_s", 0.0) or 0.0), 6),
                    "worst_time_s": round(float(st.get("max_time_s", 0.0) or 0.0), 6),
                    "total_time_s": round(total, 6),
                }
        self.operator_stats_summary = out
        
    def run(self):
        t = self.temp
        start_time = time.time()
        block_start_time = start_time
        best_obj = self.best_sol.calculate_objective()
        no_improve_iters = 0
        executed_iters = 0

        if self.time_budget_seconds:
            print(f"Starting ALNS Optimization (time budget: {self.time_budget_seconds}s, "
                  f"max {self.iterations} iterations)...")
        else:
            print(f"Starting ALNS Optimization with {self.iterations} iterations...")
        print(f"Initial State: {self.curr_sol}")

        for i in range(self.iterations):
            # ── Time-budget early exit ──
            if self.time_budget_seconds and (time.time() - start_time) >= self.time_budget_seconds:
                print(f"  Time budget of {self.time_budget_seconds}s reached at iteration {i+1} — stopping.")
                break
            # Selection
            d_idx = self._select_op(self.d_weights)
            r_idx = self._select_op(self.r_weights)
            executed_iters = i + 1
            
            new_sol = self.curr_sol.clone()
            
            # Destroy: Remove between 5% and 25% of students
            n_remove = max(1, int(len(new_sol.students) * random.uniform(0.05, 0.25)))
            _td = time.time()
            self.destroy_ops[d_idx](new_sol, n_remove)
            self._record_op_timing("destroy", self.destroy_ops[d_idx].__name__, time.time() - _td)
            
            # Repair
            _tr = time.time()
            self.repair_ops[r_idx](new_sol)
            self._record_op_timing("repair", self.repair_ops[r_idx].__name__, time.time() - _tr)
            
            # Score calculation
            new_obj = new_sol.calculate_objective()
            curr_obj = self.curr_sol.calculate_objective()
            improved_best = False
            
            reward = 0
            if new_obj > best_obj + self.min_improvement:
                self.best_sol = new_sol.clone()
                self.curr_sol = new_sol
                best_obj = new_obj
                reward = self.s1
                improved_best = True
            elif new_obj > curr_obj + self.min_improvement:
                self.curr_sol = new_sol
                reward = self.s2
            else:
                # Simulated Annealing acceptance criteria
                # We use (new - old) because we are MAXIMIZING
                diff = new_obj - curr_obj # will be negative
                p = math.exp(diff / t) if t > 0 else 0
                if random.random() < p:
                    self.curr_sol = new_sol
                    reward = self.s3
            
            # Update weights (Adaptive)
            alpha = 0.7
            self.d_weights[d_idx] = alpha * self.d_weights[d_idx] + (1-alpha) * reward
            self.r_weights[r_idx] = alpha * self.r_weights[r_idx] + (1-alpha) * reward

            if improved_best:
                no_improve_iters = 0
            else:
                no_improve_iters += 1
            
            t *= self.cooling

            if self.early_stop_patience and no_improve_iters >= self.early_stop_patience:
                print(
                    f"  Early stop: no best-objective improvement for "
                    f"{no_improve_iters} iterations."
                )
                break

            if (
                self.freeze_patience
                and t <= self.freeze_temp_threshold
                and no_improve_iters >= self.freeze_patience
            ):
                print(
                    f"  Early stop: temperature <= {self.freeze_temp_threshold:g} and "
                    f"no improvement for {no_improve_iters} iterations."
                )
                break
            
            if (i+1) % 10 == 0:
                block_end_time = time.time()
                block_elapsed = block_end_time - block_start_time
                print(f"Iteration {i+1}: Best Obj = {best_obj:.2f}, Temp = {t:.1f}, Last 10 iter: {block_elapsed:.2f}s")
                self.iteration_log.append({
                    "iteration":             i + 1,
                    "temperature":           round(float(t), 6),
                    "objective_value":       round(float(best_obj), 2),
                    "best_objective":        round(best_obj, 2),
                    "students_served":       sum(1 for s in self.best_sol.students if s.is_served),
                    "block_elapsed_seconds": round(block_elapsed, 3)
                })
                block_start_time = block_end_time

        total_elapsed = time.time() - start_time
        print(f"Optimization Complete.")
        print(f"Total Time: {total_elapsed:.2f}s")
        print(f"Final State: {self.best_sol}")

        # ── Final repair pass: rescue any remaining unserved students ──
        unserved_count = sum(1 for s in self.best_sol.students if not s.is_served)
        if unserved_count > 0:
            print(f"Running final repair pass on {unserved_count} unserved students...")
            greedy_repair(self.best_sol)
            rescued = unserved_count - sum(1 for s in self.best_sol.students if not s.is_served)
            if rescued > 0:
                print(f"  Repair pass rescued {rescued} student(s).")
            else:
                print(f"  Repair pass: no additional students could be inserted.")

        self._finalize_operator_stats(executed_iters)

        return self.best_sol

    def _select_op(self, weights):
        probs = weights / np.sum(weights)
        return np.random.choice(len(weights), p=probs)
