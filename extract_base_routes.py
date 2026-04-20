import argparse
import copy
import json
import os
from collections import defaultdict
from datetime import datetime


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


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


def _is_valid_coord(lat, lon):
    return lat is not None and lon is not None


def _school_from_payload(payload):
    if not isinstance(payload, dict):
        return None
    s = payload.get("school") or {}
    lat = s.get("latitude")
    lon = s.get("longitude")
    if _is_valid_coord(lat, lon):
        return {
            "name": s.get("name", "School"),
            "latitude": lat,
            "longitude": lon,
        }

    cfg = payload.get("config") or {}
    s2 = cfg.get("school") or {}
    lat2 = s2.get("latitude")
    lon2 = s2.get("longitude")
    if _is_valid_coord(lat2, lon2):
        return {
            "name": s2.get("name", "School"),
            "latitude": lat2,
            "longitude": lon2,
        }
    return None


def _resolve_school(input_payload, input_path):
    direct = _school_from_payload(input_payload)
    if direct is not None:
        return direct

    candidates = []
    in_abs = os.path.abspath(input_path)
    in_dir = os.path.dirname(in_abs)
    candidates.append(os.path.join(in_dir, "snapshot_input.json"))
    candidates.append(os.path.join(in_dir, "input.json"))

    for c in candidates:
        if not os.path.exists(c):
            continue
        try:
            payload = _load_json(c)
        except Exception:
            continue
        s = _school_from_payload(payload)
        if s is not None:
            return s

    return {
        "name": "School",
        "latitude": None,
        "longitude": None,
    }


def _ensure_terminal_school_stop(route, school):
    if not _is_valid_coord(school.get("latitude"), school.get("longitude")):
        return
    path = route.get("path", [])
    if not isinstance(path, list) or len(path) == 0:
        return
    if str(path[-1].get("type", "pickup")).lower() == "school":
        return
    path.append({
        "sequence": len(path),
        "node_id": "school_end",
        "latitude": school.get("latitude"),
        "longitude": school.get("longitude"),
        "type": "school",
        "students_count": 0,
        "students": [],
    })


def _build_buses_lookup(buses):
    lookup = {}
    for bus in buses or []:
        bus_id = bus.get("id")
        if bus_id is not None:
            lookup[bus_id] = bus
    return lookup


def _normalize_unified_route(route, bus_lookup):
    bus_id = route.get("bus_id")
    bus = bus_lookup.get(bus_id, {})
    route_capacity_total = _safe_int(route.get("capacity_total", bus.get("capacity", 0)))

    path_out = []
    students_total = 0
    pickup_stops_count = 0
    students_index_entries = []

    for idx, stop in enumerate(route.get("path", [])):
        stop_students = []
        for student in stop.get("students", []):
            student_out = {
                "id": student.get("id"),
                "home_latitude": student.get("home_latitude"),
                "home_longitude": student.get("home_longitude"),
                "school_stage": student.get("school_stage"),
                "assignment": student.get("assignment", "permanent"),
                "valid_from": student.get("valid_from"),
                "valid_until": student.get("valid_until"),
                "walk_distance": student.get("walk_distance"),
                "stop_latitude": stop.get("latitude"),
                "stop_longitude": stop.get("longitude"),
                "stop_node_id": stop.get("node_id"),
            }
            stop_students.append(student_out)
            students_index_entries.append((student_out["id"], route.get("id"), idx, stop.get("node_id")))

        stop_students_count = _safe_int(stop.get("students_count", len(stop_students)))
        students_total += stop_students_count
        if str(stop.get("type", "pickup")).lower() != "school":
            pickup_stops_count += 1

        path_out.append({
            "sequence": idx,
            "node_id": stop.get("node_id"),
            "latitude": stop.get("latitude"),
            "longitude": stop.get("longitude"),
            "type": stop.get("type", "pickup"),
            "students_count": stop_students_count,
            "students": stop_students,
        })

    capacity_used = _safe_int(route.get("capacity_used", students_total))
    capacity_remaining = _safe_int(route.get("capacity_remaining", max(0, route_capacity_total - capacity_used)))

    route_out = {
        "route_id": route.get("id"),
        "bus_id": bus_id,
        "total_distance_km": _safe_float(route.get("total_distance_km", 0.0)),
        "total_time_minutes": _safe_float(route.get("total_time_minutes", 0.0)),
        "detour_time_used_today": _safe_float(route.get("detour_time_used_today", 0.0)),
        "pickup_stops_count": _safe_int(route.get("pickup_stops_count", pickup_stops_count)),
        "students_count": _safe_int(route.get("students_count", students_total)),
        "capacity_total": route_capacity_total,
        "capacity_used": capacity_used,
        "capacity_remaining": capacity_remaining,
        "occupancy_pct": _safe_float(route.get("occupancy_pct", (capacity_used / route_capacity_total * 100.0) if route_capacity_total > 0 else 0.0)),
        "path": path_out,
    }
    return route_out, students_index_entries


def _extract_from_unified(data, input_path):
    buses = data.get("buses", [])
    bus_lookup = _build_buses_lookup(buses)
    school = _resolve_school(data, input_path)

    routes_out = []
    students_index_entries = []
    for route in data.get("routes", []):
        if "path" not in route:
            continue
        normalized, entries = _normalize_unified_route(route, bus_lookup)
        _ensure_terminal_school_stop(normalized, school)
        routes_out.append(normalized)
        students_index_entries.extend(entries)

    if not routes_out:
        raise ValueError("Unified routes were not found in the input file.")

    students_index = {}
    for sid, route_id, stop_sequence, stop_node_id in students_index_entries:
        if sid is None:
            continue
        students_index[str(sid)] = {
            "route_id": route_id,
            "stop_sequence": stop_sequence,
            "stop_node_id": stop_node_id,
        }

    return {
        "meta": {
            "kind": "base_routes",
            "version": 1,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_file": input_path,
            "source_type": "unified_routes_output",
        },
        "school": school,
        "buses": copy.deepcopy(buses),
        "routes": routes_out,
        "students_index": students_index,
    }


def _extract_from_experiment(data, input_path, mode_name):
    modes = data.get("modes", {})
    mode_payload = modes.get(mode_name)
    if not isinstance(mode_payload, dict):
        raise ValueError(f"Mode '{mode_name}' not found in input.modes.")

    routes_raw = mode_payload.get("routes", [])
    students_raw = mode_payload.get("students", [])

    if not routes_raw:
        raise ValueError("Selected experiment mode has no routes.")

    any_stop_coords = any(s.get("stop_latitude") is not None and s.get("stop_longitude") is not None for s in students_raw)
    if not any_stop_coords:
        raise ValueError(
            "This experiment output does not contain stop coordinates per student yet. "
            "Re-run using updated exporter or extract from unified output that includes route path."
        )

    buses_count = _safe_int((data.get("config", {}).get("buses_count")), 0)
    buses_capacity = _safe_int((data.get("config", {}).get("buses_capacity")), 0)

    school = _resolve_school(data, input_path)

    buses = []
    if buses_count > 0:
        for i in range(1, buses_count + 1):
            buses.append({
                "id": f"BUS_{i}",
                "capacity": buses_capacity,
            })

    students_by_route = defaultdict(list)
    for s in students_raw:
        rid = s.get("route_id")
        if rid:
            students_by_route[str(rid)].append(s)

    routes_out = []
    students_index = {}

    for route_rec in routes_raw:
        route_id = route_rec.get("route_id")
        if route_id is None:
            continue

        route_students = students_by_route.get(str(route_id), [])
        grouped_by_order = defaultdict(list)
        for s in route_students:
            grouped_by_order[s.get("pickup_order")].append(s)

        path = []
        seq = 0

        pickup_orders = sorted(k for k in grouped_by_order.keys() if isinstance(k, int))
        for pickup_order in pickup_orders:
            student_group = grouped_by_order[pickup_order]
            rep = student_group[0]
            stop_lat = rep.get("stop_latitude")
            stop_lon = rep.get("stop_longitude")
            stop_node_id = rep.get("stop_node_id") or f"{route_id}_p{pickup_order}"
            students_out = []
            for s in student_group:
                student_out = {
                    "id": s.get("id"),
                    "home_latitude": s.get("home_latitude"),
                    "home_longitude": s.get("home_longitude"),
                    "school_stage": s.get("stage"),
                    "assignment": "permanent",
                    "valid_from": None,
                    "valid_until": None,
                    "walk_distance": s.get("walk_distance_m"),
                    "stop_latitude": stop_lat,
                    "stop_longitude": stop_lon,
                    "stop_node_id": stop_node_id,
                }
                students_out.append(student_out)
                sid = student_out["id"]
                if sid is not None:
                    students_index[str(sid)] = {
                        "route_id": route_id,
                        "stop_sequence": seq,
                        "stop_node_id": stop_node_id,
                    }

            path.append({
                "sequence": seq,
                "node_id": stop_node_id,
                "latitude": stop_lat,
                "longitude": stop_lon,
                "type": "pickup",
                "students_count": len(students_out),
                "students": students_out,
            })
            seq += 1

        if _is_valid_coord(school.get("latitude"), school.get("longitude")):
            path.append({
                "sequence": seq,
                "node_id": "school_end",
                "latitude": school.get("latitude"),
                "longitude": school.get("longitude"),
                "type": "school",
                "students_count": 0,
                "students": [],
            })

        cap_total = _safe_int(route_rec.get("capacity_total", buses_capacity))
        cap_used = _safe_int(route_rec.get("capacity_used", route_rec.get("students_count", len(route_students))))

        routes_out.append({
            "route_id": route_id,
            "bus_id": route_rec.get("bus_id"),
            "total_distance_km": _safe_float(route_rec.get("total_distance_km", 0.0)),
            "total_time_minutes": _safe_float(route_rec.get("total_time_min", 0.0)),
            "detour_time_used_today": _safe_float(route_rec.get("detour_time_used_today", 0.0)),
            "pickup_stops_count": _safe_int(route_rec.get("pickup_stops_count", len(pickup_orders))),
            "students_count": _safe_int(route_rec.get("students_count", len(route_students))),
            "capacity_total": cap_total,
            "capacity_used": cap_used,
            "capacity_remaining": _safe_int(route_rec.get("capacity_remaining", max(0, cap_total - cap_used))),
            "occupancy_pct": _safe_float(route_rec.get("occupancy_pct", (cap_used / cap_total * 100.0) if cap_total > 0 else 0.0)),
            "path": path,
        })

    return {
        "meta": {
            "kind": "base_routes",
            "version": 1,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_file": input_path,
            "source_type": f"experiment_modes.{mode_name}",
        },
        "school": school,
        "buses": buses,
        "routes": routes_out,
        "students_index": students_index,
    }


def extract_base_routes(input_path, output_path, mode_name):
    data = _load_json(input_path)

    is_unified = isinstance(data, dict) and isinstance(data.get("routes"), list) and any("path" in r for r in data.get("routes", []))
    if is_unified:
        payload = _extract_from_unified(data, input_path)
    else:
        payload = _extract_from_experiment(data, input_path, mode_name)

    _write_json(output_path, payload)
    print(f"Base routes saved to: {output_path}")
    print(f"Routes: {len(payload.get('routes', []))}, Indexed students: {len(payload.get('students_index', {}))}")


def main():
    parser = argparse.ArgumentParser(description="Extract operational base_routes.json from algorithm output.")
    parser.add_argument("--input", required=True, help="Path to source output.json")
    parser.add_argument("--output", required=True, help="Path to write base_routes.json")
    parser.add_argument(
        "--mode",
        default="weakly_constrained",
        help="Experiment mode to extract when input is a comparison output (default: weakly_constrained)",
    )
    args = parser.parse_args()

    extract_base_routes(args.input, args.output, args.mode)


if __name__ == "__main__":
    main()
