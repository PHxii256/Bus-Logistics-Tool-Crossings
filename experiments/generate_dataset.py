"""
generate_dataset.py — Synthetic student dataset generator for Cairo graph.

Generates students following a Gaussian Annulus distribution around a school,
filtering out locations on major highways, military zones, industrial areas,
and locations without a residential road within 200 m.

Usage:
    python generate_dataset.py --n_students 50 --output my_dataset.json
"""

import json
import random
import math
import argparse
import os
import pickle
import csv
from collections import deque
from typing import Optional
import osmnx as ox
import numpy as np
import networkx as nx
from shapely.geometry import Point, Polygon

# Default school (Victory College School)
DEFAULT_SCHOOL = {
    "name": "Victory College School",
    "latitude": 29.964406,
    "longitude": 31.270319
}

# Highway types to avoid for student addresses
# 'unclassified' is intentionally excluded from valid placement roads —
# OSM uses it as a catch-all for ambiguous/unknown road types which often
# map onto non-residential land (military areas, rural tracks, etc.)
FORBIDDEN_HIGHWAYS = {'motorway', 'motorway_link', 'trunk', 'trunk_link', 'unclassified'}

# Road types that count as "residential context" (unclassified deliberately excluded)
RESIDENTIAL_TYPES = {'residential', 'living_street', 'tertiary', 'service'}

# Residential proximity threshold (metres)
RESIDENTIAL_RADIUS_M = 200


def _get_restricted_zones(center_lat, center_lon, dist=6500):
    """Download military / industrial / quarry landuse polygons for the area.
    Returns a list of Shapely Polygon/MultiPolygon objects.
    An empty list is returned on any error so generation can still proceed.
    """
    restricted_polys = []
    try:
        tags = {
            'landuse': ['military', 'industrial', 'quarry', 'mining'],
            'military': True,
        }
        features = ox.features_from_point((center_lat, center_lon), tags=tags, dist=dist)
        for _, row in features.iterrows():
            geom = row.geometry
            if geom is not None and geom.geom_type in ('Polygon', 'MultiPolygon'):
                restricted_polys.append(geom)
        print(f"  Loaded {len(restricted_polys)} restricted-zone polygons.")
    except Exception as e:
        print(f"  Warning: Could not fetch restricted zones ({e}). Skipping landuse filter.")
    return restricted_polys


def _in_restricted_zone(lat, lon, restricted_polys):
    """Return True if (lat, lon) falls inside any restricted landuse polygon."""
    if not restricted_polys:
        return False
    pt = Point(lon, lat)  # Shapely uses (x=lon, y=lat)
    return any(pt.within(poly) for poly in restricted_polys)


def _has_residential_nearby(node_id, G, radius_m=RESIDENTIAL_RADIUS_M):
    """BFS over road graph edges to find at least one residential/local road
    within *radius_m* metres of *node_id*.  Returns True if found.
    """
    visited = {node_id}
    queue = deque([(node_id, 0.0)])
    while queue:
        cur, dist_so_far = queue.popleft()
        for nb in list(G.successors(cur)) + list(G.predecessors(cur)):
            if nb in visited:
                continue
            edge_data = G.get_edge_data(cur, nb) or G.get_edge_data(nb, cur)
            if not edge_data:
                continue
            for edata in edge_data.values():
                hw = edata.get('highway', '')
                if isinstance(hw, list):
                    hw = hw[0]
                if hw in RESIDENTIAL_TYPES:
                    return True
                new_dist = dist_so_far + edata.get('length', 0)
                if new_dist <= radius_m:
                    visited.add(nb)
                    queue.append((nb, new_dist))
                    break
    return False


def _build_residential_reachability_index(G, radius_m=RESIDENTIAL_RADIUS_M):
    """Return nodes that can reach a residential-type edge within *radius_m*.

    This converts the per-candidate BFS check into a one-time precomputation.
    """
    residential_sources = set()
    for u, v, _k, data in G.edges(keys=True, data=True):
        hw = data.get('highway', '')
        hw_values = hw if isinstance(hw, list) else [hw]
        if any(h in RESIDENTIAL_TYPES for h in hw_values):
            residential_sources.add(u)
            residential_sources.add(v)

    if not residential_sources:
        return set()

    undirected = G.to_undirected(as_view=True)
    reachable = nx.multi_source_dijkstra_path_length(
        undirected,
        sources=residential_sources,
        cutoff=float(radius_m),
        weight='length',
    )
    return set(reachable.keys())


def _load_boundary_polygon_points_from_csv(csv_path: str):
    points = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames and {"lat", "lon"}.issubset(set(reader.fieldnames)):
            for row in reader:
                points.append((float(row["lat"]), float(row["lon"])))
        else:
            fh.seek(0)
            plain = csv.reader(fh)
            for row in plain:
                if len(row) < 2:
                    continue
                try:
                    lon = float(row[0])
                    lat = float(row[1])
                except Exception:
                    continue
                points.append((lat, lon))

    if len(points) < 3:
        raise ValueError(f"Boundary polygon CSV must contain at least 3 points: {csv_path}")
    if points[0] != points[-1]:
        points.append(points[0])
    return points


def _resolve_boundary_polygon_path(raw_path: str):
    if not raw_path:
        return None
    if os.path.isabs(raw_path):
        return raw_path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.abspath(os.path.join(repo_root, raw_path)),
        os.path.abspath(raw_path),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]

def gaussian_annulus_sample(center_lat, center_lon, peak_km, sigma_km, min_km, max_km):
    """
    Samples a (lat, lon) following a Gaussian distribution of distances from center.
    """
    while True:
        dist = random.gauss(peak_km, sigma_km)
        if min_km <= dist <= max_km:
            break
            
    angle = random.uniform(0, 2 * math.pi)
    
    # 1 degree lat ≈ 111 km
    d_lat = (dist * math.sin(angle)) / 111.0
    # 1 degree lon ≈ 111 * cos(lat) km
    d_lon = (dist * math.cos(angle)) / (111.0 * math.cos(math.radians(center_lat)))
    
    return center_lat + d_lat, center_lon + d_lon

# ── Default stage distribution (uniform across all 4 stages) ──
DEFAULT_STAGE_DIST = {"KG": 0.25, "ELEMENTARY": 0.25, "MIDDLE": 0.25, "HIGH": 0.25}

# ── Age ranges per stage (used when stage_dist controls stage selection) ──
_STAGE_AGE_RANGES = {
    "KG":         (4, 6),
    "ELEMENTARY": (7, 11),
    "MIDDLE":     (12, 14),
    "HIGH":       (15, 17),
}


def _pick_stage(stage_dist):
    """Weighted random choice of stage from a distribution dict."""
    # Filter out any non-stage keys (e.g. _comment)
    valid_stages = {k: v for k, v in stage_dist.items()
                    if k in ("KG", "ELEMENTARY", "MIDDLE", "HIGH") and v > 0}
    stages = list(valid_stages.keys())
    weights = [valid_stages[s] for s in stages]
    return random.choices(stages, weights=weights, k=1)[0]


def generate_dataset(
    n_students: int = 40,
    seed: int = 42,
    school: Optional[dict] = None,
    stage_dist: Optional[dict] = None,
    annulus: Optional[dict] = None,
    graph_bbox: Optional[list] = None,
    graph_boundary_mode: Optional[str] = None,
    graph_boundary_polygon: Optional[list] = None,
    graph_boundary_polygon_path: Optional[str] = None,
    graph_pkl_path: Optional[str] = None,
    buses_count: int = 4,
    bus_capacity: int = 60,
    constraints: Optional[dict] = None,
    iterations: int = 200,
    disabled_percentage: float = 0.0,
    G: Optional[nx.MultiDiGraph] = None,
) -> dict:
    """Generate a synthetic student dataset as a dict (same format as experiment JSONs).

    Parameters
    ----------
    n_students : int
    seed : int
    school : dict   – {name, latitude, longitude}
    stage_dist : dict – e.g. {"MIDDLE": 0.5, "HIGH": 0.5} — weights (will be normalised)
    annulus : dict  – {peak_km, sigma_km, min_km, max_km}
    graph_bbox : list[float] – [min_lat, min_lon, max_lat, max_lon] for consistent area coverage
    graph_boundary_mode : str – "bbox" (default) or "polygon"
    graph_boundary_polygon : list – optional polygon vertices [[lat, lon], ...]
    graph_boundary_polygon_path : str – optional CSV path with lon,lat or lat,lon headers
    graph_pkl_path : str – optional path to prebuilt drive graph pickle (networkx graph or payload with "graph")
    buses_count : int
    bus_capacity : int
    constraints : dict – ride_time_multiplier, floor_minutes, ceiling_minutes, etc.
    iterations : int – ALNS iterations (stored in meta)
    disabled_percentage : float – percent of students flagged physically/mentally disabled
    G : networkx.Graph – optional pre-built graph (avoids re-download)

    Returns
    -------
    dict : ready-to-use input data (``{"meta": …, "data": …}``)
    """
    random.seed(seed)
    np.random.seed(seed)

    school = school or DEFAULT_SCHOOL
    stage_dist = stage_dist or DEFAULT_STAGE_DIST
    annulus = annulus or {}
    constraints = constraints or {
        "ride_time_multiplier": 2.5,
        "floor_minutes": 45,
        "ceiling_minutes": 60,
        "daily_detour_budget_minutes": 5,
    }

    peak_km  = annulus.get("peak_km",  2.0)
    sigma_km = annulus.get("sigma_km", 1.0)
    min_km   = annulus.get("min_km",   0.4)
    max_km_config   = annulus.get("max_km",   5.0)
    disabled_pct = max(0.0, min(100.0, float(disabled_percentage or 0.0)))
    disabled_count = int(round((n_students * disabled_pct) / 100.0))
    disabled_student_idx = set(random.sample(range(n_students), disabled_count)) if disabled_count > 0 else set()

    center_lat, center_lon = school["latitude"], school["longitude"]
    print(f"Generating {n_students} students around {school.get('name', 'School')}...")
    print(f"  Accessibility profile: {disabled_count}/{n_students} physically/mentally disabled ({disabled_pct:.2f}%)")

    boundary_mode = str(graph_boundary_mode or "bbox").strip().lower()
    boundary_points = None
    boundary_polygon_geom = None

    if boundary_mode == "polygon":
        if isinstance(graph_boundary_polygon, list) and len(graph_boundary_polygon) >= 3:
            parsed = []
            for pt in graph_boundary_polygon:
                if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                    continue
                parsed.append((float(pt[0]), float(pt[1])))  # [lat, lon]
            if len(parsed) >= 3:
                if parsed[0] != parsed[-1]:
                    parsed.append(parsed[0])
                boundary_points = parsed
        elif graph_boundary_polygon_path:
            polygon_path = _resolve_boundary_polygon_path(str(graph_boundary_polygon_path))
            if polygon_path and os.path.exists(polygon_path):
                boundary_points = _load_boundary_polygon_points_from_csv(polygon_path)
            else:
                print(f"Warning: graph_boundary_polygon_path not found ({polygon_path}); falling back to bbox mode.")

        if boundary_points and len(boundary_points) >= 3:
            boundary_polygon_geom = Polygon([(lon, lat) for lat, lon in boundary_points])
            if not boundary_polygon_geom.is_valid:
                boundary_polygon_geom = boundary_polygon_geom.buffer(0)
            min_lon, min_lat, max_lon, max_lat = boundary_polygon_geom.bounds
            graph_bbox = [float(min_lat), float(min_lon), float(max_lat), float(max_lon)]
        else:
            print("Warning: boundary_mode='polygon' requested but polygon points are unavailable; using bbox mode.")
            boundary_mode = "bbox"

    if G is None and graph_pkl_path:
        pkl_abs = os.path.abspath(graph_pkl_path)
        if os.path.exists(pkl_abs):
            print(f"Loading drive graph from pickle: {pkl_abs}")
            with open(pkl_abs, "rb") as fh:
                loaded = pickle.load(fh)
            if isinstance(loaded, dict) and "graph" in loaded:
                loaded_graph = loaded["graph"]
            else:
                loaded_graph = loaded
            if not isinstance(loaded_graph, nx.MultiDiGraph):
                raise TypeError(f"Unsupported graph pickle format: {pkl_abs}")
            G = loaded_graph
        else:
            print(f"Warning: graph_pkl_path not found ({pkl_abs}); falling back to OSM download.")

    # ── Graph ──
    if G is None:
        if boundary_mode == "polygon" and boundary_polygon_geom is not None:
            print("Downloading graph from configured boundary polygon...")
            G = ox.graph_from_polygon(boundary_polygon_geom, network_type='drive', simplify=False)
        elif isinstance(graph_bbox, (list, tuple)) and len(graph_bbox) == 4:
            print("Downloading graph from configured bbox...")
            west, south, east, north = graph_bbox[1], graph_bbox[0], graph_bbox[3], graph_bbox[2]
            G = ox.graph_from_bbox((west, south, east, north), network_type='drive', simplify=False)
        else:
            print("Downloading graph around school (5 km radius)...")
            try:
                G = ox.graph_from_point((center_lat, center_lon), dist=5000,
                                        network_type='drive', simplify=False)
            except Exception as e:
                print(f"Graph download failed: {e}. Trying bbox fallback...")
                west, south, east, north = 31.24, 29.93, 31.30, 29.99
                G = ox.graph_from_bbox((west, south, east, north),
                                       network_type='drive', simplify=False)

    if G is None:
        raise RuntimeError("Drive graph could not be initialized")

    # ── Optional dynamic max_km based on graph bbox ──
    # When enabled, clamp max_km by graph extent so students never spawn outside.
    # When disabled, keep static max_km from input JSON.
    from math import radians, cos, sin, asin, sqrt
    
    def haversine_km(lat1, lon1, lat2, lon2):
        """Calculate great-circle distance in km between two lat/lon points."""
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return 6371.0 * c  # Earth radius in km
    
    # Use intended boundary bbox when available; otherwise infer from graph nodes.
    if isinstance(graph_bbox, (list, tuple)) and len(graph_bbox) == 4:
        bbox_south, bbox_west, bbox_north, bbox_east = [float(v) for v in graph_bbox]
    else:
        lats = [G.nodes[n]['y'] for n in G.nodes]
        lons = [G.nodes[n]['x'] for n in G.nodes]
        bbox_north, bbox_south = max(lats), min(lats)
        bbox_east, bbox_west = max(lons), min(lons)

    def _inside_boundary(lat, lon):
        if boundary_mode == "polygon" and boundary_polygon_geom is not None:
            p = Point(float(lon), float(lat))
            return boundary_polygon_geom.contains(p) or boundary_polygon_geom.touches(p)
        return bbox_south <= lat <= bbox_north and bbox_west <= lon <= bbox_east
    
    # Calculate distance to each corner
    corners = [
        (bbox_north, bbox_east),
        (bbox_north, bbox_west),
        (bbox_south, bbox_east),
        (bbox_south, bbox_west),
    ]
    bbox_max_dist = max(haversine_km(center_lat, center_lon, clat, clon) 
                        for clat, clon in corners)
    
    use_auto_bbox_circle = bool(annulus.get("auto_from_bbox_circle", False))
    use_dynamic_max_km = bool(annulus.get("dynamic_max_km", False))

    if use_auto_bbox_circle:
        bbox_radius_scale = float(annulus.get("bbox_radius_scale", 1.0) or 1.0)
        max_km = max(0.1, bbox_max_dist * bbox_radius_scale)
        min_km = max(0.0, min(min_km, max_km))

        if bool(annulus.get("auto_peak_km", True)):
            peak_ratio = float(annulus.get("peak_ratio", 0.65) or 0.65)
            peak_km = max(min_km, min(max_km, max_km * peak_ratio))
        if bool(annulus.get("auto_sigma_km", True)):
            sigma_ratio = float(annulus.get("sigma_ratio", 0.25) or 0.25)
            sigma_km = max(0.1, (max_km - min_km) * sigma_ratio)
    else:
        # Set max_km to 95% of bbox max distance (slight margin for safety) only when enabled.
        max_km_dynamic = min(max_km_config, bbox_max_dist * 0.95)
        max_km = max_km_dynamic if use_dynamic_max_km else max_km_config

    print(f"  Graph bbox: N={bbox_north:.4f}, S={bbox_south:.4f}, E={bbox_east:.4f}, W={bbox_west:.4f}")
    print(f"  Max distance to bbox corner: {bbox_max_dist:.2f} km")
    if use_auto_bbox_circle:
        print(
            "  Annulus auto-from-bbox-circle: "
            f"max_km={max_km:.2f}, peak_km={peak_km:.2f}, sigma_km={sigma_km:.2f}, min_km={min_km:.2f}"
        )
    elif use_dynamic_max_km:
        print(f"  Annulus max_km: configured={max_km_config:.1f} km, auto-adjusted={max_km:.2f} km (bbox constraint)")
    else:
        print(f"  Annulus max_km: static={max_km:.2f} km (dynamic_max_km=false)")

    # ── Restricted zones ──
    print("Fetching restricted landuse zones...")
    restricted_polys = _get_restricted_zones(center_lat, center_lon, dist=6500)

    # ── Forbidden / allowed nodes ──
    forbidden_nodes = set()
    allowed_nodes = set()
    for u, v, _k, data in G.edges(keys=True, data=True):
        h_type = data.get('highway', '')
        if isinstance(h_type, list):
            is_forbidden = any(t in FORBIDDEN_HIGHWAYS for t in h_type)
        else:
            is_forbidden = h_type in FORBIDDEN_HIGHWAYS
        if is_forbidden:
            forbidden_nodes.add(u)
            forbidden_nodes.add(v)
        else:
            allowed_nodes.add(u)
            allowed_nodes.add(v)

    final_forbidden = forbidden_nodes - allowed_nodes
    print(f"Found {len(final_forbidden)} nodes on restricted highways.")

    print("Building residential reachability index...")
    residential_reachable_nodes = _build_residential_reachability_index(
        G,
        radius_m=RESIDENTIAL_RADIUS_M,
    )
    print(
        f"  Residential context nodes (<= {RESIDENTIAL_RADIUS_M}m): "
        f"{len(residential_reachable_nodes)}"
    )

    # ── Sample students ──
    students = []
    rejected_restricted = 0
    rejected_no_residential = 0
    rejected_outside_bbox = 0
    rejected_forbidden = 0
    max_attempts_per_student = int(annulus.get("max_attempts_per_student", 12000) or 12000)
    attempts_log_every = int(annulus.get("attempts_log_every", 25000) or 25000)
    total_attempts = 0
    for i in range(n_students):
        attempts_this_student = 0
        while True:
            attempts_this_student += 1
            total_attempts += 1
            if attempts_log_every > 0 and total_attempts % attempts_log_every == 0:
                print(
                    "  Sampling progress: "
                    f"student={i+1}/{n_students}, total_attempts={total_attempts}, "
                    f"rejected_outside_bbox={rejected_outside_bbox}, "
                    f"rejected_forbidden={rejected_forbidden}, "
                    f"rejected_restricted={rejected_restricted}, "
                    f"rejected_no_residential={rejected_no_residential}"
                )

            if attempts_this_student > max_attempts_per_student:
                raise RuntimeError(
                    "Student placement exceeded max attempts. "
                    f"student_index={i+1}, attempts={attempts_this_student}, "
                    f"max_attempts_per_student={max_attempts_per_student}, "
                    f"rejected_outside_bbox={rejected_outside_bbox}, "
                    f"rejected_forbidden={rejected_forbidden}, "
                    f"rejected_restricted={rejected_restricted}, "
                    f"rejected_no_residential={rejected_no_residential}. "
                    "Try lowering peak_ratio/sigma_ratio or relaxing filters."
                )

            lat, lon = gaussian_annulus_sample(center_lat, center_lon,
                                              peak_km, sigma_km, min_km, max_km)
            if not _inside_boundary(lat, lon):
                rejected_outside_bbox += 1
                continue
            nearest_node = ox.nearest_nodes(G, lon, lat)
            if nearest_node in final_forbidden:
                rejected_forbidden += 1
                continue
            node_data = G.nodes[nearest_node]
            s_lat, s_lon = node_data['y'], node_data['x']
            if not _inside_boundary(s_lat, s_lon):
                rejected_outside_bbox += 1
                continue
            if _in_restricted_zone(s_lat, s_lon, restricted_polys):
                rejected_restricted += 1
                continue
            if nearest_node not in residential_reachable_nodes:
                rejected_no_residential += 1
                continue

            # Stage selection via distribution
            stage = _pick_stage(stage_dist)
            lo, hi = _STAGE_AGE_RANGES[stage]
            age = random.randint(lo, hi)

            students.append({
                "id": f"S{i+1:03d}",
                "latitude": s_lat,
                "longitude": s_lon,
                "age": age,
                "school_stage": stage,
                "physically_mentally_disabled": i in disabled_student_idx,
                "fee": 100.0,
            })
            break

    print(
        f"Placement stats: {rejected_restricted} restricted-zone rejections, "
        f"{rejected_no_residential} no-residential rejections, "
        f"{rejected_outside_bbox} outside-boundary rejections, "
        f"{rejected_forbidden} forbidden-highway rejections, "
        f"total_attempts={total_attempts}."
    )

    output_data = {
        "meta": {
            "mode": "generate_routes",
            "city": "Cairo",
            "description": f"Synthetic dataset - {n_students} students, seed {seed}",
            "disabled_percentage": disabled_pct,
            "disabled_students": disabled_count,
            "constraints": constraints,
            "algorithm": {"method": "alns", "iterations": iterations},
        },
        "data": {
            "school": school,
            "buses": [
                {"id": f"BUS_{i+1}", "type": "Standard",
                 "capacity": bus_capacity, "fixed_cost": 50, "var_cost_km": 1.0}
                for i in range(buses_count)
            ],
            "students": students,
        },
    }
    return output_data


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic student dataset")
    parser.add_argument('--n_students', type=int, default=40, help="Number of students to generate")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--output', default='synthetic_dataset.json', help="Output JSON path")
    parser.add_argument('--peak_km', type=float, default=2.0)
    parser.add_argument('--sigma_km', type=float, default=1.0)
    parser.add_argument('--min_km', type=float, default=0.4)
    parser.add_argument('--max_km', type=float, default=5.0)
    parser.add_argument('--disabled_percentage', type=float, default=0.0)
    args = parser.parse_args()

    data = generate_dataset(
        n_students=args.n_students,
        seed=args.seed,
        annulus={"peak_km": args.peak_km, "sigma_km": args.sigma_km,
                 "min_km": args.min_km, "max_km": args.max_km},
        disabled_percentage=args.disabled_percentage,
    )

    with open(args.output, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Successfully generated {len(data['data']['students'])} students → {args.output}")


if __name__ == '__main__':
    main()
