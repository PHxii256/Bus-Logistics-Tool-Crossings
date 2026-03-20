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
import osmnx as ox
import numpy as np
from shapely.geometry import Point

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
    queue = [(node_id, 0.0)]
    while queue:
        cur, dist_so_far = queue.pop(0)
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
    school: dict = None,
    stage_dist: dict = None,
    annulus: dict = None,
    buses_count: int = 4,
    bus_capacity: int = 60,
    constraints: dict = None,
    iterations: int = 200,
    G=None,
) -> dict:
    """Generate a synthetic student dataset as a dict (same format as experiment JSONs).

    Parameters
    ----------
    n_students : int
    seed : int
    school : dict   – {name, latitude, longitude}
    stage_dist : dict – e.g. {"MIDDLE": 0.5, "HIGH": 0.5} — weights (will be normalised)
    annulus : dict  – {peak_km, sigma_km, min_km, max_km}
    buses_count : int
    bus_capacity : int
    constraints : dict – ride_time_multiplier, floor_minutes, ceiling_minutes, etc.
    iterations : int – ALNS iterations (stored in meta)
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
    max_km   = annulus.get("max_km",   5.0)

    center_lat, center_lon = school["latitude"], school["longitude"]
    print(f"Generating {n_students} students around {school.get('name', 'School')}...")

    # ── Graph ──
    if G is None:
        print(f"Downloading graph around school (5 km radius)...")
        try:
            G = ox.graph_from_point((center_lat, center_lon), dist=5000,
                                    network_type='drive', simplify=False)
        except Exception as e:
            print(f"Graph download failed: {e}. Trying bbox fallback...")
            north, south, east, west = 29.99, 29.93, 31.30, 31.24
            try:
                G = ox.graph_from_bbox(bbox=(north, south, east, west),
                                       network_type='drive', simplify=False)
            except:
                G = ox.graph_from_bbox(north, south, east, west,
                                       network_type='drive', simplify=False)

    # ── Restricted zones ──
    print("Fetching restricted landuse zones...")
    restricted_polys = _get_restricted_zones(center_lat, center_lon, dist=6500)

    # ── Forbidden / allowed nodes ──
    forbidden_nodes = set()
    for u, v, k, data in G.edges(keys=True, data=True):
        h_type = data.get('highway', '')
        if isinstance(h_type, list):
            is_forbidden = any(t in FORBIDDEN_HIGHWAYS for t in h_type)
        else:
            is_forbidden = h_type in FORBIDDEN_HIGHWAYS
        if is_forbidden:
            forbidden_nodes.add(u)
            forbidden_nodes.add(v)

    allowed_nodes = set()
    for u, v, k, data in G.edges(keys=True, data=True):
        h_type = data.get('highway', '')
        if isinstance(h_type, list):
            is_forbidden = any(t in FORBIDDEN_HIGHWAYS for t in h_type)
        else:
            is_forbidden = h_type in FORBIDDEN_HIGHWAYS
        if not is_forbidden:
            allowed_nodes.add(u)
            allowed_nodes.add(v)

    final_forbidden = forbidden_nodes - allowed_nodes
    print(f"Found {len(final_forbidden)} nodes on restricted highways.")

    # ── Sample students ──
    students = []
    rejected_restricted = 0
    rejected_no_residential = 0
    for i in range(n_students):
        while True:
            lat, lon = gaussian_annulus_sample(center_lat, center_lon,
                                              peak_km, sigma_km, min_km, max_km)
            nearest_node = ox.nearest_nodes(G, lon, lat)
            if nearest_node in final_forbidden:
                continue
            node_data = G.nodes[nearest_node]
            s_lat, s_lon = node_data['y'], node_data['x']
            if _in_restricted_zone(s_lat, s_lon, restricted_polys):
                rejected_restricted += 1
                continue
            if not _has_residential_nearby(nearest_node, G, radius_m=RESIDENTIAL_RADIUS_M):
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
                "fee": 100.0,
            })
            break

    print(f"Placement stats: {rejected_restricted} restricted-zone rejections, "
          f"{rejected_no_residential} no-residential rejections.")

    output_data = {
        "meta": {
            "mode": "generate_routes",
            "city": "Cairo",
            "description": f"Synthetic dataset - {n_students} students, seed {seed}",
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
    args = parser.parse_args()

    data = generate_dataset(
        n_students=args.n_students,
        seed=args.seed,
        annulus={"peak_km": args.peak_km, "sigma_km": args.sigma_km,
                 "min_km": args.min_km, "max_km": args.max_km},
    )

    with open(args.output, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Successfully generated {len(data['data']['students'])} students → {args.output}")


if __name__ == '__main__':
    main()
