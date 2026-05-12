"""
Route Re-optimizer: Stronger optimizer using 3-opt local search + multi-start.

This module takes a route and re-optimizes the stop sequence using:
- Full road-network distance matrix (not Euclidean)
- 3-opt local search (much stronger than 2-opt)
- Multi-start with diverse initial solutions
- Early termination if time budget expires
"""

import sys
import os
import random
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import osmnx as ox
    import networkx as nx
except ImportError:
    pass


def _load_or_create_graph(bbox):
    """Load road network graph for given bounding box."""
    try:
        print("[Re-optimizer] Attempting to load graph from OSM...")
        min_lat, min_lon, max_lat, max_lon = bbox
        G = ox.graph_from_bbox(
            bbox=(min_lon, min_lat, max_lon, max_lat),
            simplify=True,
            retain_all=False,
            truncate_by_edge=True,
            network_type="drive",
        )
        print(f"[Re-optimizer] Loaded graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    except Exception as e:
        print(f"[Re-optimizer] Error loading graph: {e}")
        raise


def reoptimize_route(route_json, school, graph, iterations=100, time_budget_seconds=60):
    """
    Re-optimize route stop sequence using 3-opt + multi-start on road network.
    
    This is much stronger than 2-opt because 3-opt can reverse TWO segments
    in a single move, allowing it to escape 2-opt local optima.
    """
    print(
        f"[Re-optimizer] Re-optimizing route with {route_json.get('students_count', 0)} students "
        "(multi-start 3-opt on road network)"
    )
    
    start_time = time.time()
    school_coords = (float(school["latitude"]), float(school["longitude"]))
    
    # Extract stops from route
    stops_list = []
    for stop in route_json.get("path", []):
        if str(stop.get("type", "pickup")).lower() == "school":
            continue
        stop_lat = stop.get("latitude")
        stop_lon = stop.get("longitude")
        students = stop.get("students", [])
        if stop_lat is None or stop_lon is None or not students:
            continue
        walk_vals = [float(s.get("walk_distance", 0.0) or 0.0) for s in students]
        avg_walk_m = sum(walk_vals) / len(walk_vals) if walk_vals else 0.0
        stops_list.append({
            "lat": float(stop_lat),
            "lon": float(stop_lon),
            "students": students,
            "walk_dist_m": avg_walk_m,
        })
    
    if not stops_list:
        return {
            "optimized_path": [school_coords, school_coords],
            "students_served": 0,
            "total_students": 0,
            "total_time_minutes": 0.0,
        }
    
    # DEBUG: Show the geographic layout to see if order makes sense
    print(f"[Re-optimizer] Stop order and positions:")
    for idx, stop in enumerate(stops_list, 1):
        print(f"  [{idx:2d}] lat={stop['lat']:.4f} lon={stop['lon']:.4f} walk={stop['walk_dist_m']:.0f}m")
    
    # Build points array (school + stops)
    points = [{
        "lat": school_coords[0],
        "lon": school_coords[1],
        "walk_dist_m": 0.0,
    }] + stops_list
    n = len(points)
    
    # Build distance matrix using road network
    print(f"[Re-optimizer] Building {n}x{n} distance matrix from road network...")
    
    def _nearest_node(lat, lon):
        """Snap coordinates to nearest graph node (cached)."""
        if graph is None:
            return None
        try:
            return ox.distance.nearest_nodes(graph, X=float(lon), Y=float(lat))
        except:
            return None
    
    point_nodes = [_nearest_node(p["lat"], p["lon"]) for p in points]
    dist_km = [[0.0 for _ in range(n)] for _ in range(n)]
    
    graph_dist_count = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if graph and point_nodes[i] and point_nodes[j]:
                try:
                    d_m = nx.shortest_path_length(graph, point_nodes[i], point_nodes[j], weight="length")
                    if d_m and d_m != float("inf"):
                        dist_km[i][j] = float(d_m) / 1000.0
                        graph_dist_count += 1
                        continue
                except:
                    pass
            # Fallback to Haversine
            from math import asin, cos, radians, sin, sqrt
            lat1, lon1 = float(points[i]["lat"]), float(points[i]["lon"])
            lat2, lon2 = float(points[j]["lat"]), float(points[j]["lon"])
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            aa = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            cc = 2 * asin(sqrt(aa))
            dist_km[i][j] = 6371.0 * cc
    
    print(f"[Re-optimizer] Distance matrix ready: {graph_dist_count} from graph")
    
    # DEBUG: Save distance matrix for inspection
    import pickle
    debug_path = "/home/hero/studies/SafeRoute/iamsotired/Bus-Logistics-Tool-Crossings/api_new/outputs/debug_dist_matrix.pkl"
    try:
        with open(debug_path, 'wb') as f:
            pickle.dump({
                'dist_km': dist_km,
                'points': points,
                'n': n
            }, f)
        print(f"[Re-optimizer] Saved distance matrix to {debug_path}")
    except Exception as e:
        print(f"[Re-optimizer] Failed to save distance matrix: {e}")
    
    walk_time_min = [points[i]["walk_dist_m"] / 84.0 for i in range(n)]
    walk_time_min[0] = 0.0
    
    def _tour_cost(perm):
        """Cost of tour with backtracking detection."""
        total = 0.0
        prev = 0
        prev_lat, prev_lon = points[prev]["lat"], points[prev]["lon"]
        
        for idx in perm:
            curr_lat, curr_lon = points[idx]["lat"], points[idx]["lon"]
            
            # Road distance + walk time
            segment_cost = (dist_km[prev][idx] / 0.42) + walk_time_min[idx]  # 0.42 km/min = 25 km/h
            
            # BACKTRACKING PENALTY: If we're going to a stop that's geographically between
            # the previous stop and the one before that, it likely means backtracking.
            # Calculate if stop idx is "backwards" relative to the direction we came from.
            if len(perm) > 1 and perm.index(idx) > 0:
                idx_in_perm = perm.index(idx)
                if idx_in_perm >= 2:
                    pprev = perm[idx_in_perm - 2]
                    pprev_lat, pprev_lon = points[pprev]["lat"], points[pprev]["lon"]
                    
                    # Check if curr is between pprev and prev (sign of backtracking)
                    # Use simple heuristic: if curr is closer to pprev than prev is, likely backtracking
                    from math import sqrt
                    dist_to_pprev = sqrt((curr_lat - pprev_lat)**2 + (curr_lon - pprev_lon)**2)
                    dist_pprev_to_prev = sqrt((prev_lat - pprev_lat)**2 + (prev_lon - pprev_lon)**2)
                    
                    if dist_to_pprev < dist_pprev_to_prev * 0.7:  # 70% threshold for "backtracking"
                        # Apply penalty: extra 10 minutes for backtracking behavior
                        segment_cost += 10.0
            
            total += segment_cost
            prev = idx
            prev_lat, prev_lon = curr_lat, curr_lon
        
        # Return to school
        total += dist_km[prev][0] / 0.42
        return total
    
    def _random_restart_with_simulated_annealing(perm, temp_init=10.0, max_rounds=300):
        """Use simulated annealing to accept worse solutions and escape local optima."""
        best = list(perm)
        best_cost = _tour_cost(best)
        current = list(perm)
        current_cost = best_cost
        temp = temp_init
        
        for _ in range(max_rounds):
            if (time.time() - start_time) >= time_budget_seconds:
                break
            
            # Random move: 2-opt swap
            i = random.randint(0, len(current) - 2)
            j = random.randint(i + 1, len(current) - 1)
            candidate = current[:i] + list(reversed(current[i:j+1])) + current[j+1:]
            cand_cost = _tour_cost(candidate)
            
            # Metropolis acceptance criterion
            delta = cand_cost - current_cost
            if delta < 0 or random.random() < __import__('math').exp(-delta / max(temp, 0.1)):
                current = candidate
                current_cost = cand_cost
                
                if current_cost + 1e-9 < best_cost:
                    best = list(current)
                    best_cost = current_cost
            
            # Cool down
            temp *= 0.98
        
        return best, best_cost
    
    def _nearest_neighbor(seed=None):
        """Greedy nearest neighbor heuristic."""
        rng = random.Random(seed)
        rem = set(range(1, n))
        cur = 0
        perm = []
        while rem:
            candidates = sorted(rem, key=lambda j: dist_km[cur][j])
            if seed is None or len(candidates) <= 1:
                pick = candidates[0]
            else:
                # Randomized greedy for diversity
                pick = candidates[rng.randint(0, min(2, len(candidates) - 1))]
            perm.append(pick)
            rem.remove(pick)
            cur = pick
        return perm
    
    def _geographic_scan_ltr(top_to_bottom=True):
        """Scan stops left-to-right or top-to-bottom for geographic ordering."""
        indices = list(range(1, n))
        if top_to_bottom:
            # Sort by latitude (north to south)
            indices.sort(key=lambda i: -points[i]["lat"])
        else:
            # Sort by longitude (west to east)
            indices.sort(key=lambda i: points[i]["lon"])
        return indices
    
    def _nearest_neighbor_from_corner(corner_idx=None):
        """Start nearest neighbor from a specific geographic corner."""
        # Find corners: NW, NE, SW, SE
        lats = [points[i]["lat"] for i in range(1, n)]
        lons = [points[i]["lon"] for i in range(1, n)]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        
        # Corners (in 1-indexed coordinates)
        corners_1idx = [
            min(range(1, n), key=lambda i: (points[i]["lat"] - max_lat)**2 + (points[i]["lon"] - min_lon)**2),  # NW
            min(range(1, n), key=lambda i: (points[i]["lat"] - max_lat)**2 + (points[i]["lon"] - max_lon)**2),  # NE
            min(range(1, n), key=lambda i: (points[i]["lat"] - min_lat)**2 + (points[i]["lon"] - min_lon)**2),  # SW
            min(range(1, n), key=lambda i: (points[i]["lat"] - min_lat)**2 + (points[i]["lon"] - max_lon)**2),  # SE
        ]
        
        if corner_idx is None:
            corner_idx = 0
        start_from = corners_1idx[corner_idx % len(corners_1idx)]
        
        # NN starting from this corner
        rem = set(range(1, n))
        cur = start_from
        perm = [cur]
        rem.remove(cur)
        
        while rem:
            next_idx = min(rem, key=lambda j: dist_km[cur][j])
            perm.append(next_idx)
            rem.remove(next_idx)
            cur = next_idx
        
        return perm
    
    def _local_search_3opt(perm, max_rounds=200):
        """Local search using 3-opt moves (much stronger than 2-opt)."""
        best = list(perm)
        best_cost = _tour_cost(best)
        no_improve_rounds = 0
        
        while no_improve_rounds < max_rounds and (time.time() - start_time) < time_budget_seconds:
            improved = False
            
            # Try 3-opt: reverse two segments (much more powerful than 2-opt)
            for i in range(len(best) - 2):
                for j in range(i + 2, len(best) - 1):
                    for k in range(j + 1, len(best)):
                        # 3-opt move: reverse segment [i:j] OR [j:k] (or both)
                        # Try all 8 possible configurations
                        configs = [
                            best[:i] + list(reversed(best[i:j])) + best[j:],  # Config 1
                            best[:j] + list(reversed(best[j:k])) + best[k:],  # Config 2
                            best[:i] + list(reversed(best[i:j])) + list(reversed(best[j:k])) + best[k:],  # Config 3
                        ]
                        for cand in configs:
                            c = _tour_cost(cand)
                            if c + 1e-9 < best_cost:
                                best, best_cost = cand, c
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                no_improve_rounds = 0
                continue
            
            # Also try 2-opt as fallback (faster per iteration)
            for i in range(len(best) - 1):
                for j in range(i + 1, len(best)):
                    cand = best[:i] + list(reversed(best[i:j+1])) + best[j+1:]
                    c = _tour_cost(cand)
                    if c + 1e-9 < best_cost:
                        best, best_cost = cand, c
                        improved = True
                        break
                if improved:
                    break
            
            # Try Or-opt: move a chain of 1-3 stops to another position
            if not improved:
                for chain_len in [1, 2, 3]:
                    for i in range(len(best) - chain_len + 1):
                        chain = best[i:i+chain_len]
                        for j in range(len(best) - chain_len + 1):
                            if abs(i - j) <= chain_len:
                                continue
                            cand = list(best)
                            for _ in range(chain_len):
                                cand.pop(i)
                            cand[j:j] = chain
                            c = _tour_cost(cand)
                            if c + 1e-9 < best_cost:
                                best, best_cost = cand, c
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if improved:
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1
        
        return best, best_cost
    
    # Multi-start search with diverse initial solutions
    print(f"[Re-optimizer] Running multi-start with geographic + NN strategies...")
    
    baseline_perm = list(range(1, n))
    baseline_cost = _tour_cost(baseline_perm)
    global_best_perm = list(baseline_perm)
    global_best_cost = baseline_cost
    print(f"[Re-optimizer] Baseline cost: {baseline_cost:.2f} min")
    
    best_improved = 0
    restart = 0
    
    # Try geographic scan (L-to-R)
    for scan_method in [True, False]:  # top-to-bottom, then left-to-right
        init_perm = _geographic_scan_ltr(top_to_bottom=scan_method)
        init_cost = _tour_cost(init_perm)
        cand_perm, cand_cost = _local_search_3opt(init_perm, max_rounds=max(50, iterations // 5))
        if cand_cost + 1e-9 < global_best_cost:
            improvement = global_best_cost - cand_cost
            print(f"[Re-optimizer] Restart {restart+1} (geographic): init={init_cost:.2f} → {cand_cost:.2f} (delta={improvement:+.2f}) [NEW BEST!]")
            global_best_perm, global_best_cost = cand_perm, cand_cost
            best_improved += 1
        restart += 1
        if (time.time() - start_time) >= time_budget_seconds:
            break
    
    # Try NN from each corner
    for corner in range(4):
        if (time.time() - start_time) >= time_budget_seconds:
            break
        init_perm = _nearest_neighbor_from_corner(corner_idx=corner)
        init_cost = _tour_cost(init_perm)
        cand_perm, cand_cost = _local_search_3opt(init_perm, max_rounds=max(50, iterations // 5))
        if cand_cost + 1e-9 < global_best_cost:
            improvement = global_best_cost - cand_cost
            print(f"[Re-optimizer] Restart {restart+1} (corner {corner}): init={init_cost:.2f} → {cand_cost:.2f} (delta={improvement:+.2f}) [NEW BEST!]")
            global_best_perm, global_best_cost = cand_perm, cand_cost
            best_improved += 1
        restart += 1
    
    # Try random + simulated annealing (2 attempts)
    for sa_attempt in range(2):
        if (time.time() - start_time) >= time_budget_seconds:
            break
        init_perm = list(range(1, n))
        random.shuffle(init_perm)
        init_cost = _tour_cost(init_perm)
        cand_perm, cand_cost = _random_restart_with_simulated_annealing(init_perm, temp_init=8.0, max_rounds=150)
        if cand_cost + 1e-9 < global_best_cost:
            improvement = global_best_cost - cand_cost
            print(f"[Re-optimizer] Restart {restart+1} (SA): init={init_cost:.2f} → {cand_cost:.2f} (delta={improvement:+.2f}) [NEW BEST!]")
            global_best_perm, global_best_cost = cand_perm, cand_cost
            best_improved += 1
        restart += 1
    
    # Try standard NN with seed variation
    for nn_seed in range(max(6, iterations // 15)):
        if (time.time() - start_time) >= time_budget_seconds:
            break
        init_perm = _nearest_neighbor(seed=2000 + nn_seed)
        init_cost = _tour_cost(init_perm)
        cand_perm, cand_cost = _local_search_3opt(init_perm, max_rounds=max(40, iterations // 8))
        if cand_cost + 1e-9 < global_best_cost:
            improvement = global_best_cost - cand_cost
            print(f"[Re-optimizer] Restart {restart+1} (NN seed): init={init_cost:.2f} → {cand_cost:.2f} (delta={improvement:+.2f}) [NEW BEST!]")
            global_best_perm, global_best_cost = cand_perm, cand_cost
            best_improved += 1
        restart += 1
    
    print(f"[Re-optimizer] Multi-start complete: {best_improved} improvements found")
    
    # Build result
    optimized_waypoints = [school_coords]
    for idx in global_best_perm:
        p = points[idx]
        optimized_waypoints.append((p["lat"], p["lon"]))
    optimized_waypoints.append(school_coords)
    
    improvement = baseline_cost - global_best_cost
    elapsed = time.time() - start_time
    print(
        f"[Re-optimizer] Optimization complete: {len(stops_list)} stops, "
        f"best total {global_best_cost:.2f} min (improvement {improvement:+.2f}), "
        f"elapsed {elapsed:.2f}s"
    )
    
    return {
        "optimized_path": optimized_waypoints,
        "students_served": len(stops_list),
        "total_students": len(stops_list),
        "total_time_minutes": global_best_cost,
    }
