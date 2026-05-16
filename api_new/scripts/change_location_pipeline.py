import copy
import math
import time
import sys
from pathlib import Path

from api_new.scripts.map_tools import generate_before_map, generate_updated_route_map
from api_new.scripts.route_reoptimizer import reoptimize_route, _load_or_create_graph


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _haversine_km(lat1, lon1, lat2, lon2):
    radius_km = 6371.0
    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))
    d_lat = math.radians(float(lat2) - float(lat1))
    d_lon = math.radians(float(lon2) - float(lon1))
    a = math.sin(d_lat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(d_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_km * c


def _route_student_count(route):
    return sum(len(stop.get("students", [])) for stop in route.get("path", []))


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


def _ensure_school_bookends(route, school):
    school_lat = school.get("latitude")
    school_lon = school.get("longitude")
    if school_lat is None or school_lon is None:
        return
    path = route.get("path", [])
    if not path:
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

    _recompute_route_stats(route)


def _extract_request(request_payload):
    if "student_id" in request_payload and "new_location" in request_payload:
        new_location = request_payload.get("new_location") or {}
        return {
            "student_id": request_payload.get("student_id"),
            "route_id": request_payload.get("route_id"),
            "new_latitude": new_location.get("latitude"),
            "new_longitude": new_location.get("longitude"),
            "max_detour_minutes": _safe_float(request_payload.get("max_detour_minutes", 5.0), 5.0),
            "change_type": request_payload.get("change_type", "temporary"),
        }
    raise ValueError("Unsupported request payload shape. Expected student_id + new_location.")


def _route_speed_km_per_min(route):
    total_distance = _safe_float(route.get("total_distance_km"), 0.0)
    total_time = _safe_float(route.get("total_time_minutes"), 0.0)
    if total_distance > 0 and total_time > 0:
        return total_distance / total_time
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


def _direct_speed_km_per_min(route):
    route_speed = _route_speed_km_per_min(route)
    return max(0.5, route_speed) if route_speed > 0 else 0.5


def _estimate_direct_time_min(student, school, route):
    direct = student.get("direct_potential_min")
    if direct is not None:
        try:
            direct_val = float(direct)
        except (TypeError, ValueError):
            direct_val = None
        if direct_val is not None and direct_val > 0:
            return direct_val

    home_lat = student.get("home_latitude")
    home_lon = student.get("home_longitude")
    if home_lat is None or home_lon is None:
        home_lat = student.get("stop_latitude")
        home_lon = student.get("stop_longitude")
    school_lat = school.get("latitude")
    school_lon = school.get("longitude")
    if None in (home_lat, home_lon, school_lat, school_lon):
        return None

    dist_km = _haversine_km(home_lat, home_lon, school_lat, school_lon)
    speed = _direct_speed_km_per_min(route)
    if speed <= 0:
        return None
    return dist_km / speed


def _build_time_to_school(path, route_speed):
    if not path or route_speed <= 0:
        return None
    times = [0.0 for _ in path]
    accum = 0.0
    for idx in range(len(path) - 2, -1, -1):
        a = path[idx]
        b = path[idx + 1]
        a_lat, a_lon = a.get("latitude"), a.get("longitude")
        b_lat, b_lon = b.get("latitude"), b.get("longitude")
        if None in (a_lat, a_lon, b_lat, b_lon):
            return None
        segment_km = _haversine_km(a_lat, a_lon, b_lat, b_lon)
        accum += segment_km / route_speed
        times[idx] = accum
    return times


def _check_route_ride_time_caps(route, school, constraints):
    floor_min = _safe_float((constraints or {}).get("floor_minutes", 0.0), 0.0)
    offset_min = _safe_float((constraints or {}).get("acceptable_offset_minutes", 0.0), 0.0)
    route_speed = _route_speed_km_per_min(route)
    if route_speed <= 0:
        return {
            "ok": False,
            "reason": "invalid_route_speed",
            "violations": [],
        }

    path = route.get("path", [])
    time_to_school = _build_time_to_school(path, route_speed)
    if time_to_school is None:
        return {
            "ok": False,
            "reason": "missing_route_coordinates",
            "violations": [],
        }

    violations = []
    for stop_idx, stop in enumerate(path):
        if str(stop.get("type", "pickup")).lower() == "school":
            continue
        ride_time = time_to_school[stop_idx]
        for student in stop.get("students", []):
            direct_time = _estimate_direct_time_min(student, school, route)
            if direct_time is None:
                violations.append(
                    {
                        "student_id": student.get("id"),
                        "ride_time_min": round(ride_time, 2),
                        "cap_min": None,
                        "direct_time_min": None,
                        "reason": "missing_direct_time",
                    }
                )
                continue
            cap = max(floor_min, direct_time + offset_min)
            if ride_time > cap:
                violations.append(
                    {
                        "student_id": student.get("id"),
                        "ride_time_min": round(ride_time, 2),
                        "cap_min": round(cap, 2),
                        "direct_time_min": round(direct_time, 2),
                    }
                )

    return {
        "ok": len(violations) == 0,
        "violations": violations,
        "route_speed_km_per_min": round(route_speed, 4),
    }


def _find_and_remove_student(routes, student_id):
    for route in routes:
        path = route.get("path", [])
        for stop_idx, stop in enumerate(path):
            students = stop.get("students", [])
            for student_idx, student in enumerate(students):
                if str(student.get("id")) == str(student_id):
                    removed = students.pop(student_idx)
                    if str(stop.get("type", "pickup")).lower() != "school" and len(students) == 0:
                        path.pop(stop_idx)
                    _recompute_route_stats(route)
                    return route, removed
    return None, None


def _find_student_profile(school_students_data, student_id):
    candidates = school_students_data
    if isinstance(candidates, dict):
        candidates = candidates.get("students") or []
    for student in candidates or []:
        if str(student.get("id")) == str(student_id):
            return copy.deepcopy(student)
    return None


def _ordered_candidate_routes(routes, preferred_route_id):
    route_map = {str(route.get("route_id")): route for route in routes}
    ordered_ids = []
    if preferred_route_id is not None and str(preferred_route_id) in route_map:
        ordered_ids.append(str(preferred_route_id))
    for route_id in sorted(route_map.keys()):
        if route_id not in ordered_ids:
            ordered_ids.append(route_id)
    return [route_map[route_id] for route_id in ordered_ids]


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


def _capture_non_requesting_index(routes, requested_student_id):
    student_to_route = {}
    all_ids = set()
    for route in routes:
        route_id = str(route.get("route_id"))
        for stop in route.get("path", []):
            for student in stop.get("students", []):
                student_id = str(student.get("id"))
                all_ids.add(student_id)
                if student_id == str(requested_student_id):
                    continue
                if student_id in student_to_route and student_to_route[student_id] != route_id:
                    raise ValueError(f"Student {student_id} appears on multiple routes in baseline data.")
                student_to_route[student_id] = route_id
    return {
        "all_ids": all_ids,
        "non_requesting_route": student_to_route,
    }


def _verify_non_requesting_integrity(before_routes, after_routes, requested_student_id):
    try:
        before = _capture_non_requesting_index(before_routes, requested_student_id)
        after = _capture_non_requesting_index(after_routes, requested_student_id)
    except ValueError as exc:
        return {
            "ok": False,
            "reason": str(exc),
        }

    before_non_requesting_ids = set(before["non_requesting_route"].keys())
    after_non_requesting_ids = set(after["non_requesting_route"].keys())
    if before_non_requesting_ids != after_non_requesting_ids:
        missing = sorted(before_non_requesting_ids - after_non_requesting_ids)
        added = sorted(after_non_requesting_ids - before_non_requesting_ids)
        return {
            "ok": False,
            "reason": "Non-requesting student IDs changed.",
            "missing_non_requesting_students": missing,
            "added_non_requesting_students": added,
        }

    moved = []
    for student_id, route_id_before in before["non_requesting_route"].items():
        route_id_after = after["non_requesting_route"].get(student_id)
        if route_id_after != route_id_before:
            moved.append({"student_id": student_id, "from_route": route_id_before, "to_route": route_id_after})
    if moved:
        return {
            "ok": False,
            "reason": "Non-requesting students changed routes.",
            "moved_non_requesting_students": moved,
        }

    return {
        "ok": True,
        "before_total_students": len(before["all_ids"]),
        "after_total_students": len(after["all_ids"]),
        "checked_non_requesting_students": len(before_non_requesting_ids),
    }


def _build_failure(student_id, reason, runtime_seconds, routes_considered, positions_checked, invariant=None):
    response = {
        "success": False,
        "student_id": student_id,
        "reason": reason,
        "debug_stats": {
            "runtime_seconds": round(runtime_seconds, 4),
            "routes_considered": routes_considered,
            "candidate_positions_checked": positions_checked,
        },
    }
    if invariant is not None:
        response["debug_stats"]["invariant"] = invariant
    return response


def process_change_location(
    *,
    school_config,
    students_data,
    school_students_data=None,
    change_location_request,
    before_map_path,
    after_map_path,
    comparison_map_source=None,
    skip_map=False,
):
    start_time = time.time()
    request = _extract_request(change_location_request)

    student_id = request.get("student_id")
    preferred_route_id = request.get("route_id")
    new_lat = request.get("new_latitude")
    new_lon = request.get("new_longitude")
    max_detour = _safe_float(request.get("max_detour_minutes", 5.0), 5.0)
    change_type = request.get("change_type", "temporary")

    if not student_id:
        return _build_failure(None, "Missing student_id in request", time.time() - start_time, 0, 0)
    if new_lat is None or new_lon is None:
        return _build_failure(student_id, "Missing new_location coordinates", time.time() - start_time, 0, 0)

    school = copy.deepcopy(school_config.get("school") or students_data.get("school") or {})
    routes = copy.deepcopy(students_data.get("routes") or [])

    for route in routes:
        _ensure_school_bookends(route, school)
        _recompute_route_stats(route)

    before_routes_snapshot = copy.deepcopy(routes)
    if not skip_map:
        before_map_mode = generate_before_map(
            output_html=before_map_path,
            comparison_map_html=comparison_map_source,
            routes=before_routes_snapshot,
            school=school,
            new_lat=new_lat,
            new_lon=new_lon,
            student_id=student_id,
        )
    else:
        before_map_mode = "skipped"

    old_route_snapshot = None
    for route in before_routes_snapshot:
        for stop in route.get("path", []):
            if any(str(student.get("id")) == str(student_id) for student in stop.get("students", [])):
                old_route_snapshot = copy.deepcopy(route)
                break
        if old_route_snapshot is not None:
            break

    old_route, student_record = _find_and_remove_student(routes, student_id)
    if student_record is None:
        student_record = _find_student_profile(school_students_data, student_id)
    if student_record is None:
        return _build_failure(student_id, "Student was not found in students_data", time.time() - start_time, 0, 0)

    routes_considered = 0
    positions_checked = 0
    candidates = []
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
            candidates.append(
                {
                    "route_id": route.get("route_id"),
                    "insert_index": insert_index,
                    "delta_time": delta_time,
                    "delta_distance": delta_dist,
                }
            )

    if not candidates:
        response = _build_failure(
            student_id,
            f"No feasible insertion found under detour limit {max_detour:.2f} minutes",
            time.time() - start_time,
            routes_considered,
            positions_checked,
        )
        response["before_map"] = before_map_path
        response["before_map_mode"] = before_map_mode
        return response

    candidates.sort(key=lambda item: item["delta_time"])

    selected_route = None
    selected_route_before = None
    best = None
    ride_time_check = None
    constraints = school_config.get("constraints") or {}
    for candidate in candidates:
        route = next(r for r in routes if str(r.get("route_id")) == str(candidate["route_id"]))
        route_before = copy.deepcopy(route)
        temp_route = copy.deepcopy(route)
        _insert_candidate_stop(temp_route, candidate["insert_index"], student_record, new_lat, new_lon, change_type)
        temp_route["total_distance_km"] = round(_safe_float(temp_route.get("total_distance_km"), 0.0) + candidate["delta_distance"], 4)
        temp_route["total_time_minutes"] = round(_safe_float(temp_route.get("total_time_minutes"), 0.0) + candidate["delta_time"], 4)
        if str(change_type).lower() == "temporary":
            temp_route["detour_time_used_today"] = round(
                _safe_float(temp_route.get("detour_time_used_today", 0.0), 0.0) + max(0.0, candidate["delta_time"]),
                4,
            )
        _recompute_route_stats(temp_route)

        ride_time_check = _check_route_ride_time_caps(temp_route, school, constraints)
        if ride_time_check.get("ok"):
            selected_route = route
            selected_route_before = route_before
            best = candidate
            break

    if selected_route is None or best is None:
        response = _build_failure(
            student_id,
            "No insertion satisfied ride-time caps for existing students.",
            time.time() - start_time,
            routes_considered,
            positions_checked,
            invariant=ride_time_check,
        )
        response["before_map"] = before_map_path
        response["before_map_mode"] = before_map_mode
        return response

    _insert_candidate_stop(selected_route, best["insert_index"], student_record, new_lat, new_lon, change_type)
    selected_route["total_distance_km"] = round(_safe_float(selected_route.get("total_distance_km"), 0.0) + best["delta_distance"], 4)
    selected_route["total_time_minutes"] = round(_safe_float(selected_route.get("total_time_minutes"), 0.0) + best["delta_time"], 4)
    if str(change_type).lower() == "temporary":
        selected_route["detour_time_used_today"] = round(
            _safe_float(selected_route.get("detour_time_used_today", 0.0), 0.0) + max(0.0, best["delta_time"]),
            4,
        )
    _recompute_route_stats(selected_route)

    invariant = _verify_non_requesting_integrity(before_routes_snapshot, routes, student_id)
    if not invariant.get("ok"):
        response = _build_failure(
            student_id,
            "Invariant violation: non-requesting students changed unexpectedly.",
            time.time() - start_time,
            routes_considered,
            positions_checked,
            invariant=invariant,
        )
        response["before_map"] = before_map_path
        response["before_map_mode"] = before_map_mode
        return response

    route_for_map_before = selected_route_before
    if old_route_snapshot is not None and str(selected_route_before.get("route_id")) == str(old_route_snapshot.get("route_id")):
        route_for_map_before = old_route_snapshot

    # Attempt full ALNS re-optimization only when map generation is enabled
    optimized_path_coords = None
    if not skip_map:
        try:
            print("[Pipeline] Attempting ALNS re-optimization...")

            # DEBUG: Log what students are in the route before optimization
            route_students_debug = []
            for stop in selected_route.get("path", []):
                if stop.get("type", "pickup").lower() != "school":
                    for s in stop.get("students", []):
                        route_students_debug.append((s.get("id"), s.get("home_latitude"), s.get("home_longitude")))
            print(f"[Pipeline] selected_route R{selected_route.get('route_id')} has {len(route_students_debug)} students before re-opt")
            s093_before = [s for s in route_students_debug if s[0] == "S093"]
            if s093_before:
                print(f"[Pipeline]   S093: {s093_before[0]}")

            # Compute bounding box from route stops
            all_lats = [school.get("latitude")]
            all_lons = [school.get("longitude")]
            for stop in selected_route.get("path", []):
                if stop.get("latitude") is not None and stop.get("longitude") is not None:
                    all_lats.append(float(stop["latitude"]))
                    all_lons.append(float(stop["longitude"]))

            if len(all_lats) < 2:
                raise ValueError("Not enough coordinates to compute bounding box")

            bbox = [min(all_lats), min(all_lons), max(all_lats), max(all_lons)]
            print(f"[Pipeline] Route bbox: {bbox}")

            # Load road network
            graph = _load_or_create_graph(bbox)

            # Run ALNS re-optimizer
            print("[Pipeline] Calling reoptimize_route...")
            result = reoptimize_route(selected_route, school, graph, iterations=100, time_budget_seconds=60)
            print(f"[Pipeline] reoptimize_route returned: {type(result)}")

            # Trace optimized waypoints through OSRM to get road geometry
            from api_new.scripts.map_tools import _osrm_segment_geometry
            optimized_waypoints = result.get("optimized_path", [])
            print(f"[Pipeline] Got {len(optimized_waypoints)} optimized waypoints")

            if len(optimized_waypoints) >= 2:
                merged_coords = []
                for idx in range(len(optimized_waypoints) - 1):
                    a_lat, a_lon = optimized_waypoints[idx]
                    b_lat, b_lon = optimized_waypoints[idx + 1]
                    segment = _osrm_segment_geometry(a_lat, a_lon, b_lat, b_lon)
                    if not segment:
                        segment = [(a_lat, a_lon), (b_lat, b_lon)]
                    if not merged_coords:
                        merged_coords.extend(segment)
                    else:
                        merged_coords.extend(segment[1:])
                optimized_path_coords = merged_coords
                print(f"[Pipeline] ALNS re-optimization succeeded: {len(optimized_waypoints)} waypoints -> {len(optimized_path_coords)} road coords")
        except Exception as e:
            import traceback
            print(f"[Pipeline] ALNS re-optimization failed: {e}")
            print(traceback.format_exc())
            optimized_path_coords = None

    if not skip_map:
        generate_updated_route_map(
            output_html=after_map_path,
            old_route=route_for_map_before,
            new_route=selected_route,
            school=school,
            new_lat=new_lat,
            new_lon=new_lon,
            student_id=student_id,
            use_osrm=True,
            optimized_path_coords=optimized_path_coords,
        )
    else:
        # Skip writing an updated map when requested (fast mode)
        pass

    response = {
        "success": True,
        "student_id": student_id,
        "requested_route_id": preferred_route_id,
        "assigned_route_id": selected_route.get("route_id"),
        "detour_minutes_added": round(best["delta_time"], 4),
        "updated_route": selected_route,
        "updated_routes": routes,
        "debug_stats": {
            "runtime_seconds": round(time.time() - start_time, 4),
            "routes_considered": routes_considered,
            "candidate_positions_checked": positions_checked,
            "before_map_mode": before_map_mode,
            "invariant": invariant,
            "ride_time_check": ride_time_check,
        },
        "before_map": before_map_path,
        "after_map": after_map_path,
    }
    return response

