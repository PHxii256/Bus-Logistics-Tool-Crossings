import argparse
import copy
import json
import math
import time

import folium


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


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def _route_polyline(path):
    coords = []
    for stop in path:
        lat = stop.get("latitude")
        lon = stop.get("longitude")
        if lat is None or lon is None:
            continue
        coords.append((float(lat), float(lon)))
    return coords


def _estimate_path_distance_km(path):
    coords = _route_polyline(path)
    if len(coords) < 2:
        return 0.0
    total = 0.0
    for i in range(len(coords) - 1):
        total += _haversine_km(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1])
    return total


def _route_speed_km_per_min(route):
    total_distance = _safe_float(route.get("total_distance_km"), 0.0)
    total_time = _safe_float(route.get("total_time_minutes"), 0.0)
    if total_distance > 0 and total_time > 0:
        return total_distance / total_time

    path_distance = _estimate_path_distance_km(route.get("path", []))
    if path_distance > 0 and total_time > 0:
        return path_distance / total_time

    return 0.35


def _segment_delta(route, prev_stop, next_stop, new_lat, new_lon):
    p_lat, p_lon = prev_stop.get("latitude"), prev_stop.get("longitude")
    n_lat, n_lon = next_stop.get("latitude"), next_stop.get("longitude")
    if None in (p_lat, p_lon, n_lat, n_lon):
        return None, None

    old_dist = _haversine_km(p_lat, p_lon, n_lat, n_lon)
    new_dist = _haversine_km(p_lat, p_lon, new_lat, new_lon) + _haversine_km(new_lat, new_lon, n_lat, n_lon)
    delta_dist = new_dist - old_dist
    speed = _route_speed_km_per_min(route)
    delta_time = delta_dist / speed if speed > 0 else None
    return delta_dist, delta_time


def _route_student_count(route):
    total = 0
    for stop in route.get("path", []):
        total += len(stop.get("students", []))
    return total


def _recompute_route_stats(route):
    students_count = _route_student_count(route)
    pickup_stops_count = sum(1 for stop in route.get("path", []) if str(stop.get("type", "pickup")).lower() != "school")
    cap_total = int(route.get("capacity_total", 0) or 0)
    cap_used = students_count
    cap_remaining = max(0, cap_total - cap_used)
    route["students_count"] = students_count
    route["pickup_stops_count"] = pickup_stops_count
    route["capacity_used"] = cap_used
    route["capacity_remaining"] = cap_remaining
    route["occupancy_pct"] = round((cap_used / cap_total) * 100.0, 1) if cap_total > 0 else 0.0
    for idx, stop in enumerate(route.get("path", [])):
        stop["sequence"] = idx
        stop["students_count"] = len(stop.get("students", []))


def _extract_request(req):
    # New request shape
    if "student_id" in req and "new_location" in req:
        new_loc = req.get("new_location", {}) or {}
        return {
            "student_id": req.get("student_id"),
            "route_id": req.get("route_id"),
            "new_latitude": new_loc.get("latitude"),
            "new_longitude": new_loc.get("longitude"),
            "max_detour_minutes": _safe_float(req.get("max_detour_minutes", 5.0), 5.0),
            "change_type": req.get("change_type", "temporary"),
        }

    # Legacy nested shape fallback
    meta = req.get("meta", {}) if isinstance(req, dict) else {}
    new_loc = meta.get("new_location", {}) if isinstance(meta, dict) else {}
    constraints = meta.get("constraints", {}) if isinstance(meta, dict) else {}
    return {
        "student_id": meta.get("student_id"),
        "route_id": meta.get("route_id"),
        "new_latitude": new_loc.get("latitude"),
        "new_longitude": new_loc.get("longitude"),
        "max_detour_minutes": _safe_float(constraints.get("daily_detour_budget_minutes", 5.0), 5.0),
        "change_type": meta.get("change_type", "temporary"),
    }


def _find_and_remove_student(routes, student_id):
    for route in routes:
        path = route.get("path", [])
        for idx, stop in enumerate(path):
            students = stop.get("students", [])
            for s_idx, student in enumerate(students):
                if str(student.get("id")) == str(student_id):
                    removed = students.pop(s_idx)
                    if str(stop.get("type", "pickup")).lower() != "school" and len(students) == 0:
                        path.pop(idx)
                    _recompute_route_stats(route)
                    return route, removed
    return None, None


def _ordered_candidate_routes(routes, preferred_route_id):
    route_map = {str(r.get("route_id")): r for r in routes}
    ordered_ids = []

    if preferred_route_id is not None and str(preferred_route_id) in route_map:
        ordered_ids.append(str(preferred_route_id))

    for rid in sorted(route_map.keys()):
        if rid not in ordered_ids:
            ordered_ids.append(rid)

    return [route_map[rid] for rid in ordered_ids]


def _insert_candidate_stop(route, insert_index, student_record, new_lat, new_lon, change_type):
    new_stop_node_id = f"changed_{student_record.get('id')}"
    new_student = copy.deepcopy(student_record)
    new_student["home_latitude"] = new_lat
    new_student["home_longitude"] = new_lon
    new_student["assignment"] = change_type
    new_student["stop_latitude"] = new_lat
    new_student["stop_longitude"] = new_lon
    new_student["stop_node_id"] = new_stop_node_id
    new_student["walk_distance"] = 0.0

    new_stop = {
        "sequence": insert_index,
        "node_id": new_stop_node_id,
        "latitude": new_lat,
        "longitude": new_lon,
        "type": "pickup",
        "students_count": 1,
        "students": [new_student],
    }
    route.get("path", []).insert(insert_index, new_stop)
    _recompute_route_stats(route)


def _build_response_failure(student_id, reason, runtime_s, routes_considered, positions_checked):
    return {
        "success": False,
        "student_id": student_id,
        "reason": reason,
        "debug_stats": {
            "runtime_seconds": round(runtime_s, 4),
            "routes_considered": routes_considered,
            "candidate_positions_checked": positions_checked,
        },
    }


def _generate_updated_route_html(output_html, old_route, new_route, school, new_lat, new_lon):
    center = [new_lat, new_lon]
    if school.get("latitude") is not None and school.get("longitude") is not None:
        center = [school.get("latitude"), school.get("longitude")]

    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")

    old_coords = _route_polyline(old_route.get("path", []))
    new_coords = _route_polyline(new_route.get("path", []))

    if old_coords:
        folium.PolyLine(old_coords, color="#6c757d", weight=4, opacity=0.85, dash_array="8,6", tooltip="Original route").add_to(m)
    if new_coords:
        folium.PolyLine(new_coords, color="#0d6efd", weight=5, opacity=0.9, tooltip="Updated route").add_to(m)

    for stop in old_route.get("path", []):
        lat = stop.get("latitude")
        lon = stop.get("longitude")
        if lat is None or lon is None:
            continue
        if str(stop.get("type", "pickup")).lower() == "school":
            continue
        folium.CircleMarker(
            location=(lat, lon),
            radius=4,
            color="#6c757d",
            fill=True,
            fill_color="#6c757d",
            fill_opacity=0.7,
            tooltip="Original stop",
        ).add_to(m)

    for stop in new_route.get("path", []):
        lat = stop.get("latitude")
        lon = stop.get("longitude")
        if lat is None or lon is None:
            continue
        if str(stop.get("type", "pickup")).lower() == "school":
            continue
        folium.CircleMarker(
            location=(lat, lon),
            radius=5,
            color="#0d6efd",
            fill=True,
            fill_color="#0d6efd",
            fill_opacity=0.8,
            tooltip="Updated stop",
        ).add_to(m)

    if school.get("latitude") is not None and school.get("longitude") is not None:
        folium.Marker(
            location=(school.get("latitude"), school.get("longitude")),
            icon=folium.Icon(color="green", icon="graduation-cap", prefix="fa"),
            tooltip="School",
        ).add_to(m)

    folium.Marker(
        location=(new_lat, new_lon),
        icon=folium.Icon(color="orange", icon="star", prefix="fa"),
        tooltip="Inserted changed location",
        popup="Inserted changed location",
    ).add_to(m)

    m.save(output_html)


def process_request(base_routes_path, request_path, response_path, updated_route_html):
    t0 = time.time()

    base = _load_json(base_routes_path)
    request_payload = _extract_request(_load_json(request_path))

    student_id = request_payload.get("student_id")
    preferred_route_id = request_payload.get("route_id")
    new_lat = request_payload.get("new_latitude")
    new_lon = request_payload.get("new_longitude")
    max_detour = _safe_float(request_payload.get("max_detour_minutes", 5.0), 5.0)
    change_type = request_payload.get("change_type", "temporary")

    routes = copy.deepcopy(base.get("routes", []))
    school = copy.deepcopy(base.get("school", {}))

    if not student_id:
        response = _build_response_failure(None, "Missing student_id in request", time.time() - t0, 0, 0)
        _write_json(response_path, response)
        return response

    if new_lat is None or new_lon is None:
        response = _build_response_failure(student_id, "Missing new_location coordinates", time.time() - t0, 0, 0)
        _write_json(response_path, response)
        return response

    old_route, student_record = _find_and_remove_student(routes, student_id)
    if old_route is None or student_record is None:
        response = _build_response_failure(student_id, "Student was not found in base_routes", time.time() - t0, 0, 0)
        _write_json(response_path, response)
        return response

    old_route_snapshot = copy.deepcopy(old_route)

    routes_considered = 0
    positions_checked = 0
    best = None

    candidate_routes = _ordered_candidate_routes(routes, preferred_route_id)
    for route in candidate_routes:
        _recompute_route_stats(route)
        if _safe_float(route.get("capacity_remaining", 0), 0) < 1:
            continue

        path = route.get("path", [])
        if len(path) < 2:
            continue

        routes_considered += 1
        for insert_index in range(1, len(path)):
            prev_stop = path[insert_index - 1]
            next_stop = path[insert_index]
            delta_dist, delta_time = _segment_delta(route, prev_stop, next_stop, new_lat, new_lon)
            if delta_time is None:
                continue
            positions_checked += 1
            if delta_time > max_detour:
                continue

            if best is None or delta_time < best["delta_time"]:
                best = {
                    "route_id": route.get("route_id"),
                    "insert_index": insert_index,
                    "delta_time": delta_time,
                    "delta_distance": delta_dist,
                }

    runtime_s = time.time() - t0

    if best is None:
        response = _build_response_failure(
            student_id,
            f"No feasible insertion found under detour limit {max_detour:.2f} minutes",
            runtime_s,
            routes_considered,
            positions_checked,
        )
        _write_json(response_path, response)
        return response

    selected_route = next(r for r in routes if str(r.get("route_id")) == str(best["route_id"]))
    _insert_candidate_stop(selected_route, best["insert_index"], student_record, new_lat, new_lon, change_type)

    selected_route["total_distance_km"] = round(_safe_float(selected_route.get("total_distance_km"), 0.0) + best["delta_distance"], 4)
    selected_route["total_time_minutes"] = round(_safe_float(selected_route.get("total_time_minutes"), 0.0) + best["delta_time"], 4)
    if str(change_type).lower() == "temporary":
        selected_route["detour_time_used_today"] = round(
            _safe_float(selected_route.get("detour_time_used_today", 0.0), 0.0) + max(0.0, best["delta_time"]),
            4,
        )
    _recompute_route_stats(selected_route)

    response = {
        "success": True,
        "student_id": student_id,
        "requested_route_id": preferred_route_id,
        "assigned_route_id": selected_route.get("route_id"),
        "detour_minutes_added": round(best["delta_time"], 4),
        "updated_route": selected_route,
        "debug_stats": {
            "runtime_seconds": round(time.time() - t0, 4),
            "routes_considered": routes_considered,
            "candidate_positions_checked": positions_checked,
        },
    }

    _write_json(response_path, response)

    _generate_updated_route_html(
        output_html=updated_route_html,
        old_route=old_route_snapshot,
        new_route=selected_route,
        school=school,
        new_lat=new_lat,
        new_lon=new_lon,
    )

    return response


def main():
    parser = argparse.ArgumentParser(description="Process a change-location request using base_routes.json.")
    parser.add_argument("--base-routes", required=True, help="Path to base_routes.json")
    parser.add_argument("--request", required=True, help="Path to change_location_request.json")
    parser.add_argument("--response", default="response.json", help="Path to write response JSON")
    parser.add_argument("--updated-route-html", default="updated_route.html", help="Path to write updated route map HTML")
    args = parser.parse_args()

    response = process_request(
        base_routes_path=args.base_routes,
        request_path=args.request,
        response_path=args.response,
        updated_route_html=args.updated_route_html,
    )

    print(f"Request processed: success={response.get('success')}")
    print(f"Response saved to: {args.response}")
    if response.get("success"):
        print(f"Updated route HTML saved to: {args.updated_route_html}")


if __name__ == "__main__":
    main()
