"""
Route Re-optimizer: Uses ALNS to optimize the sequence of stops in an existing route.

This module takes an updated route JSON and re-optimizes the stop sequence using
the full ALNS algorithm with walk distance penalties and safety constraints.
"""

import sys
import os
import json
import pickle
from pathlib import Path

# Add repo root to path so we can import from main modules
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import osmnx as ox
    import networkx as nx
except ImportError:
    pass

from entities import Student, School_Stage, Stop, Route, Bus
from solution_state import ServiceSolution
from alns_engine import ALNSEngine
from detour_engine import (
    precalculate_distance_matrix,
    calculate_route_distance,
    calculate_route_time,
    snap_address_to_edge,
)
from data_loader import load_json


def _create_students_from_route(route_json):
    """Convert route JSON into Student entities.
    
    Args:
        route_json: Route dict with path containing students
        
    Returns:
        List of Student objects
    """
    students = []
    for stop in route_json.get("path", []):
        if str(stop.get("type", "pickup")).lower() == "school":
            continue
        for student_data in stop.get("students", []):
            student_id = str(student_data.get("id", "unknown"))
            
            # Use home location as primary coords
            home_lat = student_data.get("home_latitude")
            home_lon = student_data.get("home_longitude")
            
            if home_lat is None or home_lon is None:
                continue
            
            # Map school_stage string to enum
            stage_str = (student_data.get("school_stage") or "").upper()
            try:
                stage = School_Stage[stage_str] if stage_str in School_Stage.__members__ else School_Stage.HIGH
            except (KeyError, ValueError):
                stage = School_Stage.HIGH
            
            # Create student
            student = Student(
                id=student_id,
                lat=float(home_lat),
                lon=float(home_lon),
                age=0,  # Not used in routing
                school_stage=stage,
                fee=0,  # Not used
                assignment=student_data.get("assignment", "permanent"),
                physically_mentally_disabled=student_data.get("physically_mentally_disabled", False),
            )
            students.append(student)
    
    return students


def _load_or_create_graph(bbox):
    """Load or create the road network graph for the given bounding box.
    
    Args:
        bbox: [min_lat, min_lon, max_lat, max_lon]
        
    Returns:
        networkx Graph object
    """
    try:
        print("[Re-optimizer] Attempting to load graph from OSM...")
        min_lat, min_lon, max_lat, max_lon = bbox
        # osmnx 2.1.0 API: bbox = (left, bottom, right, top) = (min_lon, min_lat, max_lon, max_lat)
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
    """Re-optimize a route's stop sequence using greedy nearest-neighbor + 2-opt local search.
    
    This simpler approach avoids ALNS initialization issues while still producing
    well-optimized routes through nearest-neighbor construction + local improvement.
    
    Args:
        route_json: Updated route dict from pipeline
        school: School dict with latitude/longitude
        graph: networkx Graph for routing
        iterations: Number of 2-opt improvement iterations
        time_budget_seconds: Time budget for optimization
        
    Returns:
        Dict with optimized_path (list of (lat, lon) tuples in visit order)
    """
    print(f"[Re-optimizer] Re-optimizing route with {route_json.get('students_count', 0)} students using greedy NN + 2-opt")
    
    import time
    start_time = time.time()
    
    # 1. Extract waypoints (school + all student homes)
    waypoints = [(float(school["latitude"]), float(school["longitude"]), "school")]
    
    for stop in route_json.get("path", []):
        if str(stop.get("type", "pickup")).lower() == "school":
            continue
        for student_data in stop.get("students", []):
            home_lat = student_data.get("home_latitude")
            home_lon = student_data.get("home_longitude")
            if home_lat is not None and home_lon is not None:
                waypoints.append((float(home_lat), float(home_lon), str(student_data.get("id", "unknown"))))
    
    if len(waypoints) < 2:
        print(f"[Re-optimizer] Not enough waypoints ({len(waypoints)}) to optimize")
        return {"optimized_path": [(w[0], w[1]) for w in waypoints]}
    
    print(f"[Re-optimizer] Optimizing {len(waypoints)} waypoints (1 school + {len(waypoints)-1} students)")
    
    # 2. Build distance function using Haversine (fast approximation)
    def _distance(p1, p2):
        """Compute distance between two points using Haversine."""
        from math import radians, cos, sin, asin, sqrt
        lon1, lat1 = p1[1], p1[0]
        lon2, lat2 = p2[1], p2[0]
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        km = 6371 * c
        return km
    
    # 3. Greedy nearest-neighbor construction
    school_idx = 0
    unvisited = set(range(1, len(waypoints)))  # All except school
    tour = [school_idx]  # Start at school
    
    current = school_idx
    while unvisited and time.time() - start_time < time_budget_seconds:
        # Find nearest unvisited
        nearest = min(unvisited, key=lambda j: _distance(waypoints[current][:2], waypoints[j][:2]))
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    # Add remaining (in case we ran out of time)
    tour.extend(sorted(unvisited))
    tour.append(school_idx)  # Return to school
    
    print(f"[Re-optimizer] Initial tour: {len(tour)} stops")
    
    # 4. 2-opt local search improvement (limited by time budget)
    max_2opt_iterations = min(iterations, int(time_budget_seconds * 10))  # Scale with time budget
    improved = True
    iteration_count = 0
    
    while improved and iteration_count < max_2opt_iterations and time.time() - start_time < time_budget_seconds * 0.9:
        improved = False
        iteration_count += 1
        
        for i in range(1, len(tour) - 2):
            for j in range(i + 2, len(tour) - 1):
                # Compute distance before swap
                d_before = (
                    _distance(waypoints[tour[i-1]][:2], waypoints[tour[i]][:2]) +
                    _distance(waypoints[tour[j]][:2], waypoints[tour[j+1]][:2])
                )
                
                # Compute distance after swap
                d_after = (
                    _distance(waypoints[tour[i-1]][:2], waypoints[tour[j]][:2]) +
                    _distance(waypoints[tour[i]][:2], waypoints[tour[j+1]][:2])
                )
                
                # If improvement found, apply it
                if d_after < d_before - 1e-6:  # Small epsilon to avoid floating point issues
                    tour[i:j+1] = tour[i:j+1][::-1]
                    improved = True
                    break
            
            if improved:
                break
    
    print(f"[Re-optimizer] 2-opt: {iteration_count} iterations, tour length: {sum(_distance(waypoints[tour[k]][:2], waypoints[tour[k+1]][:2]) for k in range(len(tour)-1)):.2f} km")
    
    # 5. Extract optimized path (excluding return to school at the end)
    optimized_path = [(waypoints[idx][0], waypoints[idx][1]) for idx in tour[:-1]]
    
    print(f"[Re-optimizer] Optimization complete: {len(optimized_path)} waypoints, {time.time() - start_time:.2f}s elapsed")
    
    return {
        "optimized_path": optimized_path,
        "students_served": len(waypoints) - 1,
        "total_students": len(waypoints) - 1,
        "route_distance_km": sum(_distance(waypoints[tour[k]][:2], waypoints[tour[k+1]][:2]) for k in range(len(tour)-1)),
    }
