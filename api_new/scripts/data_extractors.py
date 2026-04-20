import copy
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _get_school(payload):
    school = payload.get("school") or {}
    if school.get("latitude") is not None and school.get("longitude") is not None:
        return {
            "name": school.get("name", "School"),
            "latitude": school.get("latitude"),
            "longitude": school.get("longitude"),
        }
    config_school = (payload.get("config") or {}).get("school") or {}
    if config_school.get("latitude") is not None and config_school.get("longitude") is not None:
        return {
            "name": config_school.get("name", "School"),
            "latitude": config_school.get("latitude"),
            "longitude": config_school.get("longitude"),
        }
    return {
        "name": "School",
        "latitude": None,
        "longitude": None,
    }


def _ensure_school_bookends(route, school):
    path = route.get("path", [])
    if not path:
        return
    school_lat = school.get("latitude")
    school_lon = school.get("longitude")
    if school_lat is None or school_lon is None:
        return
    if str(path[0].get("type", "pickup")).lower() != "school":
        path.insert(
            0,
            {
                "sequence": 0,
                "node_id": "school_start",
                "latitude": school_lat,
                "longitude": school_lon,
                "type": "school",
                "students_count": 0,
                "students": [],
            },
        )
    else:
        path[0]["type"] = "school"
        path[0]["latitude"] = school_lat
        path[0]["longitude"] = school_lon
        path[0]["node_id"] = path[0].get("node_id") or "school_start"

    if str(path[-1].get("type", "pickup")).lower() != "school":
        path.append(
            {
                "sequence": len(path),
                "node_id": "school_end",
                "latitude": school_lat,
                "longitude": school_lon,
                "type": "school",
                "students_count": 0,
                "students": [],
            }
        )
    else:
        path[-1]["type"] = "school"
        path[-1]["latitude"] = school_lat
        path[-1]["longitude"] = school_lon
        path[-1]["node_id"] = path[-1].get("node_id") or "school_end"

    for idx, stop in enumerate(path):
        stop["sequence"] = idx
        stop["students_count"] = len(stop.get("students", []))


def _normalize_student(student, stop):
    normalized = {
        "id": student.get("id"),
        "home_latitude": student.get("home_latitude"),
        "home_longitude": student.get("home_longitude"),
        "school_stage": student.get("school_stage") or student.get("stage"),
        "assignment": student.get("assignment", "permanent"),
        "valid_from": student.get("valid_from"),
        "valid_until": student.get("valid_until"),
        "walk_distance": student.get("walk_distance"),
        "stop_latitude": stop.get("latitude"),
        "stop_longitude": stop.get("longitude"),
        "stop_node_id": stop.get("node_id"),
    }
    return normalized


def _normalize_route(route, school, bus_lookup):
    route_id = route.get("route_id", route.get("id"))
    if route_id is None:
        raise ValueError("Route missing route_id/id.")
    path_in = route.get("path") or []
    path_out = []
    students_count = 0
    pickup_stops_count = 0
    for idx, stop in enumerate(path_in):
        stop_type = stop.get("type", "pickup")
        students = []
        for student in stop.get("students", []):
            student_out = _normalize_student(student, stop)
            students.append(student_out)
        students_count += len(students)
        if str(stop_type).lower() != "school":
            pickup_stops_count += 1
        path_out.append(
            {
                "sequence": idx,
                "node_id": stop.get("node_id", f"{route_id}_stop_{idx}"),
                "latitude": stop.get("latitude"),
                "longitude": stop.get("longitude"),
                "type": stop_type,
                "students_count": len(students),
                "students": students,
            }
        )

    bus_id = route.get("bus_id")
    bus_capacity = _safe_int((bus_lookup.get(str(bus_id)) or {}).get("capacity", 0), 0)

    route_out = {
        "route_id": str(route_id),
        "bus_id": bus_id,
        "total_distance_km": _safe_float(route.get("total_distance_km", 0.0), 0.0),
        "total_time_minutes": _safe_float(route.get("total_time_minutes", 0.0), 0.0),
        "detour_time_used_today": _safe_float(route.get("detour_time_used_today", 0.0), 0.0),
        "pickup_stops_count": _safe_int(route.get("pickup_stops_count", pickup_stops_count), pickup_stops_count),
        "students_count": _safe_int(route.get("students_count", students_count), students_count),
        "capacity_total": _safe_int(route.get("capacity_total", bus_capacity), bus_capacity),
        "capacity_used": _safe_int(route.get("capacity_used", students_count), students_count),
        "capacity_remaining": _safe_int(route.get("capacity_remaining", 0), 0),
        "occupancy_pct": _safe_float(route.get("occupancy_pct", 0.0), 0.0),
        "path": path_out,
    }
    if route_out["capacity_total"] <= 0 and route_out["capacity_used"] > 0:
        route_out["capacity_total"] = route_out["capacity_used"]
    route_out["capacity_remaining"] = max(0, route_out["capacity_total"] - route_out["capacity_used"])
    if route_out["capacity_total"] > 0:
        route_out["occupancy_pct"] = round((route_out["capacity_used"] / route_out["capacity_total"]) * 100.0, 1)

    _ensure_school_bookends(route_out, school)
    return route_out


def _normalize_routes_payload(payload):
    school = _get_school(payload)
    buses = copy.deepcopy(payload.get("buses") or [])
    bus_lookup = {str(bus.get("id")): bus for bus in buses if bus.get("id") is not None}
    routes_in = payload.get("routes") or []
    routes_out = [_normalize_route(route, school, bus_lookup) for route in routes_in]
    if not routes_out:
        raise ValueError("No routes with path were found in payload.")
    return {
        "meta": {
            "kind": "students_data",
            "version": 1,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_type": "routes_payload",
        },
        "school": school,
        "buses": buses,
        "routes": routes_out,
    }


def _extract_students_data_from_experiment(output_payload, snapshot_payload, mode_name):
    mode_payload = ((output_payload.get("modes") or {}).get(mode_name)) or {}
    routes_raw = mode_payload.get("routes") or []
    students_raw = mode_payload.get("students") or []
    if not routes_raw:
        raise ValueError(f"Mode '{mode_name}' has no routes in output.json.")

    has_coords = any(
        student.get("stop_latitude") is not None and student.get("stop_longitude") is not None for student in students_raw
    )
    if not has_coords:
        raise ValueError(
            "output.json for this run does not include stop coordinates per student. "
            "Provide api_new/inputs/students_data.json override or use a run with detailed stop coordinates."
        )

    school = _get_school(snapshot_payload)
    buses_count = _safe_int((snapshot_payload.get("buses") or {}).get("count", 0), 0)
    buses_capacity = _safe_int((snapshot_payload.get("buses") or {}).get("capacity", 0), 0)
    buses = []
    if buses_count > 0:
        buses = [{"id": f"BUS_{i}", "capacity": buses_capacity} for i in range(1, buses_count + 1)]

    route_stats = {}
    for route in routes_raw:
        route_id = route.get("route_id", route.get("id"))
        if route_id is None:
            continue
        route_stats[str(route_id)] = route

    students_by_route = defaultdict(list)
    for student in students_raw:
        route_id = student.get("route_id")
        if route_id is None:
            continue
        students_by_route[str(route_id)].append(student)

    routes = []
    for route_id, students in students_by_route.items():
        stops_map = defaultdict(list)
        for student in students:
            order = _safe_int(student.get("pickup_order", 10**6), 10**6)
            stop_lat = student.get("stop_latitude")
            stop_lon = student.get("stop_longitude")
            stop_node = student.get("stop_node_id", f"{route_id}_stop_{order}")
            key = (order, stop_node, stop_lat, stop_lon)
            normalized_student = {
                "id": student.get("id"),
                "home_latitude": student.get("home_latitude"),
                "home_longitude": student.get("home_longitude"),
                "school_stage": student.get("school_stage", student.get("stage")),
                "assignment": student.get("assignment", "permanent"),
                "valid_from": student.get("valid_from"),
                "valid_until": student.get("valid_until"),
                "walk_distance": student.get("walk_distance", student.get("walk_distance_m")),
                "stop_latitude": stop_lat,
                "stop_longitude": stop_lon,
                "stop_node_id": stop_node,
            }
            stops_map[key].append(normalized_student)

        ordered_stop_keys = sorted(stops_map.keys(), key=lambda item: item[0])
        path = [
            {
                "sequence": 0,
                "node_id": "school_start",
                "latitude": school.get("latitude"),
                "longitude": school.get("longitude"),
                "type": "school",
                "students_count": 0,
                "students": [],
            }
        ]
        for idx, (order, node_id, stop_lat, stop_lon) in enumerate(ordered_stop_keys, start=1):
            students_here = stops_map[(order, node_id, stop_lat, stop_lon)]
            path.append(
                {
                    "sequence": idx,
                    "node_id": node_id,
                    "latitude": stop_lat,
                    "longitude": stop_lon,
                    "type": "pickup",
                    "students_count": len(students_here),
                    "students": students_here,
                }
            )
        path.append(
            {
                "sequence": len(path),
                "node_id": "school_end",
                "latitude": school.get("latitude"),
                "longitude": school.get("longitude"),
                "type": "school",
                "students_count": 0,
                "students": [],
            }
        )

        stats = route_stats.get(str(route_id), {})
        students_count = sum(len(stop["students"]) for stop in path if stop["type"] != "school")
        capacity_total = buses_capacity
        route_out = {
            "route_id": str(route_id),
            "bus_id": stats.get("bus_id"),
            "total_distance_km": _safe_float(stats.get("total_distance_km", 0.0), 0.0),
            "total_time_minutes": _safe_float(stats.get("total_time_min", stats.get("total_time_minutes", 0.0)), 0.0),
            "detour_time_used_today": _safe_float(stats.get("detour_time_used_today", 0.0), 0.0),
            "pickup_stops_count": len(path) - 2,
            "students_count": students_count,
            "capacity_total": capacity_total,
            "capacity_used": students_count,
            "capacity_remaining": max(0, capacity_total - students_count),
            "occupancy_pct": round((students_count / capacity_total) * 100.0, 1) if capacity_total > 0 else 0.0,
            "path": path,
        }
        routes.append(route_out)

    if not routes:
        raise ValueError("Could not build any route paths from experiment output.")

    return {
        "meta": {
            "kind": "students_data",
            "version": 1,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_type": f"experiment_output:{mode_name}",
        },
        "school": school,
        "buses": buses,
        "routes": routes,
    }


def build_school_config(snapshot_payload):
    school = _get_school(snapshot_payload)
    buses = snapshot_payload.get("buses") or {}
    constraints = snapshot_payload.get("constraints") or {}
    return {
        "school": school,
        "buses": {
            "count": _safe_int(buses.get("count", 0), 0),
            "capacity": _safe_int(buses.get("capacity", 0), 0),
        },
        "constraints": {
            "floor_minutes": _safe_float(constraints.get("floor_minutes", 0.0), 0.0),
            "acceptable_offset_minutes": _safe_float(constraints.get("acceptable_offset_minutes", 0.0), 0.0),
            "daily_detour_budget_minutes": _safe_float(constraints.get("daily_detour_budget_minutes", 0.0), 0.0),
        },
        "stage_walk_limits": copy.deepcopy(snapshot_payload.get("stage_walk_limits") or {}),
    }


def validate_school_config(school_config):
    required = [
        ("buses", "count"),
        ("buses", "capacity"),
        ("constraints", "floor_minutes"),
        ("constraints", "acceptable_offset_minutes"),
        ("constraints", "daily_detour_budget_minutes"),
    ]
    for parent, child in required:
        if parent not in school_config or child not in school_config[parent]:
            raise ValueError(f"school_config missing required field: {parent}.{child}")
    if "stage_walk_limits" not in school_config:
        raise ValueError("school_config missing required field: stage_walk_limits")


def validate_students_data(students_data):
    if "routes" not in students_data or not isinstance(students_data["routes"], list):
        raise ValueError("students_data must contain routes list.")
    if not students_data["routes"]:
        raise ValueError("students_data.routes cannot be empty.")
    for route in students_data["routes"]:
        if "route_id" not in route:
            raise ValueError("Each route in students_data must contain route_id.")
        if "path" not in route or not isinstance(route["path"], list):
            raise ValueError(f"Route {route.get('route_id')} missing path list.")


def resolve_inputs_from_run(run_dir, mode_name="weakly_constrained"):
    run_path = Path(run_dir)
    snapshot_path = run_path / "snapshot_input.json"
    output_path = run_path / "output.json"
    base_routes_path = run_path / "base_routes.json"
    comparison_map_path = run_path / "comparison_map.html"

    if not snapshot_path.exists():
        raise FileNotFoundError(f"Missing run snapshot_input.json at {snapshot_path}")
    snapshot_payload = load_json(snapshot_path)
    school_config = build_school_config(snapshot_payload)

    if base_routes_path.exists():
        students_data = _normalize_routes_payload(load_json(base_routes_path))
        students_source = str(base_routes_path)
    elif output_path.exists():
        output_payload = load_json(output_path)
        if (output_payload.get("routes") or []) and any("path" in route for route in output_payload.get("routes", [])):
            students_data = _normalize_routes_payload(output_payload)
            students_source = str(output_path)
        else:
            students_data = _extract_students_data_from_experiment(output_payload, snapshot_payload, mode_name=mode_name)
            students_source = str(output_path)
    else:
        raise FileNotFoundError(f"Run folder has neither base_routes.json nor output.json: {run_path}")

    return {
        "school_config": school_config,
        "students_data": students_data,
        "comparison_map_html": str(comparison_map_path) if comparison_map_path.exists() else None,
        "school_config_source": str(snapshot_path),
        "students_source": students_source,
    }


def normalize_students_data_payload(payload):
    return _normalize_routes_payload(payload)

