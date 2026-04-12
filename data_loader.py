"""
Data loader for the Safety-Aware Bus Optimization algorithm

Handles two service modes:
  Mode 1 (generate_routes): Load students, buses, constraints -> create empty routes
  Mode 2 (change_location): Load existing routes (unified schema) -> reconstruct objects

Also provides serialization to the unified route schema for output.
"""

import json
from entities import Student, Bus, Route, Stop, School_Stage
import networkx as nx
import osmnx as ox


# ============================================================================
# HELPERS
# ============================================================================

def school_stage_from_string(stage_str):
    """Convert string to School_Stage enum."""
    stage_map = {
        'KG': School_Stage.KG,
        'ELEMENTARY': School_Stage.ELEMENTARY,
        'MIDDLE': School_Stage.MIDDLE,
        'HIGH': School_Stage.HIGH
    }
    return stage_map.get(stage_str.upper(), School_Stage.ELEMENTARY)


def load_json(json_file):
    """Load raw JSON from file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def _create_buses_dict(bus_data_list):
    """Create dict mapping bus_id -> Bus object from JSON bus array."""
    buses = {}
    for data in bus_data_list:
        bus = Bus(
            bus_type=data.get('type'),
            capacity=data.get('capacity'),
            fixed_cost=data.get('fixed_cost'),
            var_cost_km=data.get('var_cost_km')
        )
        bus.bus_id = data.get('id')   # store for reporting / serialization
        buses[data.get('id')] = bus
    return buses


def _create_students(student_data_list):
    """Create Student objects from JSON array."""
    students = []
    for data in student_data_list:
        student = Student(
            id=data.get('id'),
            lat=data.get('latitude'),
            lon=data.get('longitude'),
            age=data.get('age'),
            school_stage=school_stage_from_string(data.get('school_stage', 'ELEMENTARY')),
            fee=data.get('fee', 100.0),
            assignment=data.get('assignment', 'permanent'),
            valid_from=data.get('valid_from'),
            valid_until=data.get('valid_until'),
            physically_mentally_disabled=data.get('physically_mentally_disabled', False),
        )
        if 'walk_radius_override' in data:
            student.walk_radius = data['walk_radius_override']
        students.append(student)
    return students


# ============================================================================
# MODE 1: Generate Routes (initial optimization)
# ============================================================================

def _unpack_input(data):
    """Return (meta, dataset) from the {meta, data} input format."""
    if 'meta' not in data or 'data' not in data:
        raise ValueError("Input file must have top-level 'meta' and 'data' fields.")
    return data['meta'], data['data']


def load_mode1_input(data, G):
    """Load Mode 1 (generate_routes) input.
    
    Args:
        data: Parsed JSON dict with mode="generate_routes"
        G: NetworkX road graph
        
    Returns:
        Tuple of (students, buses_dict, routes, school_coords, constraints, algo_config)
    """
    meta, dataset = _unpack_input(data)

    school = dataset.get('school', {})
    school_coords = {
        'latitude': school.get('latitude'),
        'longitude': school.get('longitude'),
        'name': school.get('name', 'School')
    }
    
    buses = _create_buses_dict(dataset.get('buses', []))
    students = _create_students(dataset.get('students', []))
    constraints = meta.get('constraints', {})
    algo_config = meta.get('algorithm', {'method': 'alns', 'iterations': 60})

    # Per-student ride-time constraints:
    #   DMRT cap = max(base_mrt, T_direct + acceptable_offset_minutes)
    ride_time_multiplier = constraints.get('ride_time_multiplier', 2.5)
    floor_minutes        = constraints.get('floor_minutes',        45)
    ceiling_minutes      = constraints.get('ceiling_minutes',      60)
    try:
        acceptable_offset_minutes = float(constraints.get('acceptable_offset_minutes', 30))
    except (TypeError, ValueError):
        acceptable_offset_minutes = 30.0
    bidirectional_check  = constraints.get('bidirectional_check',  True)
    mrt_enabled          = bool(constraints.get('mrt_enabled', constraints.get('mrt enabled', False)))
    mrt_minutes_raw      = constraints.get('mrt', None)
    try:
        mrt_minutes = float(mrt_minutes_raw) if mrt_minutes_raw is not None else None
    except (TypeError, ValueError):
        mrt_minutes = None
    caps_enabled         = constraints.get('enabled',              True)
    # Legacy flat cap — kept as fallback (= ceiling)
    route_tmax = constraints.get('route_tmax', ceiling_minutes)
    if mrt_enabled and mrt_minutes is not None and mrt_minutes > 0:
        route_tmax = mrt_minutes
    
    # Create one route per bus, each starting with school start/end stops
    routes = []
    
    # Snap school ONCE (not per bus) — use BallTree if already built
    import detour_engine as _det
    school_node = _det.fast_nearest_node(G, school_coords['longitude'], school_coords['latitude'])
    school_node_lat = G.nodes[school_node]['y']
    school_node_lon = G.nodes[school_node]['x']
    
    for i, (bus_id, bus) in enumerate(buses.items()):
        route = Route(
            bus=bus,
            route_id=f"R{i+1}",
            route_tmax=route_tmax,
            ride_time_multiplier=ride_time_multiplier,
            floor_minutes=floor_minutes,
            ceiling_minutes=ceiling_minutes,
            bidirectional_check=bidirectional_check,
            mrt_enabled=mrt_enabled,
            mrt_minutes=mrt_minutes,
            acceptable_offset_minutes=acceptable_offset_minutes,
        )
        # Flag used by validate_permanent_student for cap enforcement behavior
        route.ride_caps_enabled = caps_enabled
        
        node_lat = school_node_lat
        node_lon = school_node_lon
        
        start_stop = Stop(school_node, node_lat, node_lon,
                          stop_id=f"R{i+1}-Start", stop_type="school")
        end_stop = Stop(school_node, node_lat, node_lon,
                        stop_id=f"R{i+1}-End", stop_type="school")
        route.stops = [start_stop, end_stop]
        routes.append(route)
    
    return students, buses, routes, school_coords, constraints, algo_config


# ============================================================================
# MODE 2: Change Location (single student re-assignment)
# ============================================================================

def load_mode2_input(data, G):
    """Load Mode 2 (change_location) input.
    
    Reconstructs Route/Stop/Student objects from the unified route schema,
    then extracts the change request details.
    
    Args:
        data: Parsed JSON dict with mode="change_location"
        G: NetworkX road graph
        
    Returns:
        Tuple of (student_id, new_coords, change_type, valid_from, valid_until,
                  algo_config, routes, all_students, buses_dict, school_coords)
    """
    meta, dataset = _unpack_input(data)

    school = dataset.get('school', {})
    school_coords = {
        'latitude': school.get('latitude'),
        'longitude': school.get('longitude'),
        'name': school.get('name', 'School')
    }
    
    buses = _create_buses_dict(dataset.get('buses', []))
    
    # Reconstruct routes from unified schema
    routes, all_students = _reconstruct_routes(dataset.get('routes', []), buses, G)
    
    # Extract change request (all from meta)
    student_id  = meta.get('student_id')
    new_loc     = meta.get('new_location', {})
    new_coords  = (new_loc.get('latitude'), new_loc.get('longitude'))
    change_type = meta.get('change_type', 'temporary')
    valid_from  = meta.get('valid_from')
    valid_until = meta.get('valid_until')
    algo_config = meta.get('algorithm', {'method': 'cheapest_insertion'})
    
    return (student_id, new_coords, change_type, valid_from, valid_until,
            algo_config, routes, all_students, buses, school_coords)


def _reconstruct_routes(routes_json, buses_dict, G):
    """Rebuild Route/Stop/Student objects from the unified route schema.
    
    Args:
        routes_json: List of route dicts from the unified schema
        buses_dict: Dict of bus_id -> Bus
        G: NetworkX graph (for node validation)
        
    Returns:
        Tuple of (routes_list, all_students_list)
    """
    routes = []
    all_students = []
    
    for route_data in routes_json:
        bus_id = route_data.get('bus_id')
        bus = buses_dict.get(bus_id)
        if not bus:
            print(f"Warning: Bus '{bus_id}' not found, skipping route {route_data.get('id')}")
            continue
        
        route = Route(
            bus=bus,
            route_id=route_data.get('id'),
            route_tmax=route_data.get('route_tmax', 75),
            ride_time_multiplier=route_data.get('ride_time_multiplier', 2.5),
            floor_minutes=route_data.get('floor_minutes',   45),
            ceiling_minutes=route_data.get('ceiling_minutes', 60),
            bidirectional_check=route_data.get('bidirectional_check', True),
            mrt_enabled=route_data.get('mrt_enabled', route_data.get('mrt enabled', False)),
            mrt_minutes=route_data.get('mrt', None),
            acceptable_offset_minutes=route_data.get('acceptable_offset_minutes', 30),
        )
        route.total_distance = route_data.get('total_distance_km', 0)
        route.total_time = route_data.get('total_time_minutes', 0)
        route.detour_time_used = route_data.get('detour_time_used_today', 0)
        
        for stop_data in route_data.get('path', []):
            stop = Stop(
                node_id=stop_data.get('node_id'),
                lat=stop_data.get('latitude'),
                lon=stop_data.get('longitude'),
                stop_type=stop_data.get('type', 'pickup')
            )
            
            # Reconstruct students at this stop
            for s_data in stop_data.get('students', []):
                student = Student(
                    id=s_data.get('id'),
                    lat=s_data.get('home_latitude'),
                    lon=s_data.get('home_longitude'),
                    age=s_data.get('age', 0),
                    school_stage=school_stage_from_string(s_data.get('school_stage', 'ELEMENTARY')),
                    fee=s_data.get('fee', 0),
                    assignment=s_data.get('assignment', 'permanent'),
                    valid_from=s_data.get('valid_from'),
                    valid_until=s_data.get('valid_until'),
                    physically_mentally_disabled=s_data.get('physically_mentally_disabled', False),
                )
                stop.add_student(student)
                all_students.append(student)
            
            route.stops.append(stop)
        
        routes.append(route)
    
    return routes, all_students


# ============================================================================
# OUTPUT SERIALIZATION: Unified Route Schema
# ============================================================================

def serialize_routes(routes, buses, school_coords, unserved_students=None, graph=None):
    """Serialize Route/Stop/Student objects to the unified route schema.
    
    This is the single output format used by both Mode 1 and Mode 2.
    
    Args:
        routes: List of Route objects
        buses: Dict of bus_id -> Bus
        school_coords: Dict with name, latitude, longitude
        unserved_students: Optional list of Student objects that weren't placed
        graph: Optional graph for walk distance computation
        
    Returns:
        Dict in the unified route schema format
    """
    from detour_engine import haversine_walk_distance
    
    # Build bus_id lookup (reverse: Bus object -> id)
    bus_to_id = {}
    for bid, bus_obj in buses.items():
        bus_to_id[id(bus_obj)] = bid
    
    output = {
        "school": {
            "name": school_coords.get('name', 'School'),
            "latitude": school_coords.get('latitude'),
            "longitude": school_coords.get('longitude')
        },
        "buses": [
            {
                "id": bus_id,
                "type": b.bus_type,
                "capacity": b.capacity,
                "fixed_cost": b.fixed_cost,
                "var_cost_km": b.var_cost_km
            } for bus_id, b in buses.items()
        ],
        "routes": [],
        "unserved_students": []
    }
    
    for route in routes:
        # Skip routes that have no pickup stops (school-only routes are empty)
        if not any(stop.stop_type != 'school' for stop in route.stops):
            continue

        route_bus_id = bus_to_id.get(id(route.bus), "UNKNOWN")
        route_capacity_total = int(getattr(route.bus, 'capacity', 0) or 0)

        path_data = []
        for stop in route.stops:
            students_data = []
            for s in stop.students:
                # Calculate walk distance (haversine, meters)
                walk_d = haversine_walk_distance(s.coords[0], s.coords[1],
                                                 stop.coords[0], stop.coords[1])
                
                students_data.append({
                    "id": s.id,
                    "home_latitude": s.coords[0],
                    "home_longitude": s.coords[1],
                    "school_stage": s.school_stage.name,
                    "physically_mentally_disabled": bool(getattr(s, 'physically_mentally_disabled', False)),
                    "age": s.age,
                    "fee": s.fee,
                    "assignment": s.assignment,
                    "valid_from": s.valid_from,
                    "valid_until": s.valid_until,
                    "walk_distance": round(walk_d, 1)
                })
            
            path_data.append({
                "node_id": stop.node_id,
                "latitude": stop.coords[0],
                "longitude": stop.coords[1],
                "type": stop.stop_type,
                "students_count": len(students_data),
                "students": students_data
            })

        route_students_count = int(route.get_student_count())
        pickup_stops_count = sum(1 for s in route.stops if s.stop_type != 'school')
        capacity_used = route_students_count
        capacity_remaining = max(0, route_capacity_total - capacity_used)
        occupancy_pct = round((capacity_used / route_capacity_total) * 100.0, 1) if route_capacity_total > 0 else 0.0
        
        output["routes"].append({
            "id": route.route_id,
            "bus_id": route_bus_id,
            "route_tmax": route.route_tmax,
            "ride_time_multiplier": getattr(route, 'ride_time_multiplier', 2.5),
            "floor_minutes":        getattr(route, 'floor_minutes',        45),
            "ceiling_minutes":      getattr(route, 'ceiling_minutes',      60),
            "acceptable_offset_minutes": getattr(route, 'acceptable_offset_minutes', 30),
            "bidirectional_check":  getattr(route, 'bidirectional_check',  True),
            "mrt_enabled":          getattr(route, 'mrt_enabled',          False),
            "mrt":                  getattr(route, 'mrt_minutes',          None),
            "total_distance_km": round(route.total_distance, 2),
            "total_time_minutes": round(route.total_time, 2),
            "detour_time_used_today": round(route.detour_time_used, 2),
            "students_count": route_students_count,
            "pickup_stops_count": pickup_stops_count,
            "capacity_total": route_capacity_total,
            "capacity_used": capacity_used,
            "capacity_remaining": capacity_remaining,
            "occupancy_pct": occupancy_pct,
            "path": path_data
        })
    
    # Add unserved students
    if unserved_students:
        for s in unserved_students:
            output["unserved_students"].append({
                "id": s.id,
                "home_latitude": s.coords[0],
                "home_longitude": s.coords[1],
                "school_stage": s.school_stage.name,
                "physically_mentally_disabled": bool(getattr(s, 'physically_mentally_disabled', False)),
                "age": s.age,
                "fee": s.fee,
                "reason": s.failure_reason or "No valid insertion found"
            })
    
    return output


# ============================================================================
# PRINTING HELPERS
# ============================================================================

def print_input_summary(students, buses, routes, school_coords):
    """Print a summary of loaded data."""
    print(f"\n{'='*80}")
    print("INPUT DATA SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"School: {school_coords.get('name', 'School')}")
    print(f"  Location: ({school_coords['latitude']}, {school_coords['longitude']})\n")
    
    print(f"Students: {len(students)}")
    for student in students:
        print(f"  {student.id}: {student.school_stage.name}, walk={student.walk_radius}m, fee=${student.fee}")
    
    print(f"\nBuses: {len(buses)}")
    for bus_id, bus in buses.items():
        print(f"  {bus_id}: {bus.bus_type}, capacity={bus.capacity}, fixed=${bus.fixed_cost}, var=${bus.var_cost_km}/km")
    
    print(f"\nRoutes: {len(routes)}")
    for route in routes:
        floor_m = getattr(route, 'floor_minutes',        45)
        dmrt_offset = getattr(route, 'acceptable_offset_minutes', 30)
        mrt_on  = bool(getattr(route, 'mrt_enabled', False))
        mrt_val = getattr(route, 'mrt_minutes', None)
        if mrt_on and mrt_val is not None:
            cap_desc = f"hard MRT={float(mrt_val):.1f} min (AM and PM)"
        else:
            base_mrt = floor_m
            cap_desc = f"DMRT cap max({float(base_mrt):.1f}, direct+{float(dmrt_offset):.1f}) min"
        print(f"  {route.route_id}: {len(route.stops)} stops, {cap_desc}, capacity={route.bus.capacity}")
    
    print(f"\n{'='*80}\n")
