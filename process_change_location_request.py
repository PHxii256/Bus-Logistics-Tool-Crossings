import argparse
import copy
import json
import math
import os
import time
import urllib.parse
import urllib.request

import folium
from folium import plugins


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


def _is_valid_coord(lat, lon):
    return lat is not None and lon is not None


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


def _route_polyline_on_roads(path, osrm_base_url="http://localhost:5000"):
    """Build a road-following polyline by querying OSRM route geometry per segment."""
    stops = _route_polyline(path)
    if len(stops) < 2:
        return stops

    merged = []
    for i in range(len(stops) - 1):
        a_lat, a_lon = stops[i]
        b_lat, b_lon = stops[i + 1]
        segment = _osrm_segment_geometry(a_lat, a_lon, b_lat, b_lon, osrm_base_url=osrm_base_url)
        if not segment:
            segment = [(a_lat, a_lon), (b_lat, b_lon)]

        if not merged:
            merged.extend(segment)
        else:
            merged.extend(segment[1:])

    return merged


def _osrm_segment_geometry(lat1, lon1, lat2, lon2, osrm_base_url="http://localhost:5000"):
    """Return (lat, lon) geometry for one road segment from OSRM, or None on failure."""
    coords = f"{float(lon1)},{float(lat1)};{float(lon2)},{float(lat2)}"
    base = str(osrm_base_url).rstrip("/")
    path = f"/route/v1/driving/{coords}"
    query = urllib.parse.urlencode({
        "overview": "full",
        "geometries": "geojson",
        "steps": "false",
    })
    url = f"{base}{path}?{query}"

    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None

    if payload.get("code") != "Ok":
        return None

    routes = payload.get("routes") or []
    if not routes:
        return None

    geom = (routes[0].get("geometry") or {}).get("coordinates") or []
    if len(geom) < 2:
        return None

    out = []
    for lon, lat in geom:
        out.append((float(lat), float(lon)))
    return out


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


def _resolve_school_from_base(base_payload, base_routes_path):
    direct = _school_from_payload(base_payload)
    if direct is not None:
        return direct

    candidates = []
    base_abs = os.path.abspath(base_routes_path)
    base_dir = os.path.dirname(base_abs)

    meta = base_payload.get("meta") or {}
    source_file = meta.get("source_file")
    source_abs = None
    if source_file:
        source_abs = source_file if os.path.isabs(source_file) else os.path.abspath(os.path.join(base_dir, source_file))
        source_dir = os.path.dirname(source_abs)
        candidates.append(os.path.join(source_dir, "snapshot_input.json"))
        candidates.append(os.path.join(source_dir, "input.json"))
        candidates.append(source_abs)

    candidates.append(os.path.join(base_dir, "snapshot_input.json"))
    candidates.append(os.path.join(base_dir, "input.json"))

    for c in candidates:
        if not c or not os.path.exists(c):
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


def _ensure_school_bookends(route, school):
    if not _is_valid_coord(school.get("latitude"), school.get("longitude")):
        return

    path = route.get("path", [])
    if not isinstance(path, list) or len(path) == 0:
        return

    if str(path[0].get("type", "pickup")).lower() != "school":
        path.insert(0, {
            "sequence": 0,
            "node_id": "school_start",
            "latitude": school.get("latitude"),
            "longitude": school.get("longitude"),
            "type": "school",
            "students_count": 0,
            "students": [],
        })
    else:
        path[0]["node_id"] = path[0].get("node_id") or "school_start"
        path[0]["latitude"] = school.get("latitude")
        path[0]["longitude"] = school.get("longitude")
        path[0]["type"] = "school"

    if str(path[-1].get("type", "pickup")).lower() != "school":
        path.append({
            "sequence": len(path),
            "node_id": "school_end",
            "latitude": school.get("latitude"),
            "longitude": school.get("longitude"),
            "type": "school",
            "students_count": 0,
            "students": [],
        })
    else:
        path[-1]["node_id"] = path[-1].get("node_id") or "school_end"
        path[-1]["latitude"] = school.get("latitude")
        path[-1]["longitude"] = school.get("longitude")
        path[-1]["type"] = "school"


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


def _generate_updated_route_html(output_html, old_route, new_route, school, new_lat, new_lon, student_id=None):
    center = [new_lat, new_lon]
    if school.get("latitude") is not None and school.get("longitude") is not None:
        center = [school.get("latitude"), school.get("longitude")]

    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")

    old_route_display = copy.deepcopy(old_route)
    new_route_display = copy.deepcopy(new_route)
    _ensure_school_bookends(old_route_display, school)
    _ensure_school_bookends(new_route_display, school)
    _recompute_route_stats(old_route_display)
    _recompute_route_stats(new_route_display)

    old_route_id = str(old_route_display.get("route_id") or "N/A")
    new_route_id = str(new_route_display.get("route_id") or "N/A")
    student_label = str(student_id) if student_id is not None else "Unknown"

    old_coords = _route_polyline_on_roads(old_route_display.get("path", []))
    new_coords = _route_polyline_on_roads(new_route_display.get("path", []))

    if old_coords:
        folium.PolyLine(
            old_coords,
            color="#6c757d",
            weight=4,
            opacity=0.8,
            dash_array="8,6",
            tooltip=f"Original route {old_route_id} (school to school)",
        ).add_to(m)
    if new_coords:
        updated = folium.PolyLine(
            new_coords,
            color="#2E7D32",
            weight=5,
            opacity=0.9,
            tooltip=f"Updated route {new_route_id} (school to school)",
        )
        updated.add_to(m)
        plugins.PolyLineTextPath(
            updated,
            "          \u27A4          ",
            repeat=True,
            offset=6,
            attributes={"fill": "#2E7D32", "font-weight": "bold", "font-size": "24"},
        ).add_to(m)

    for stop in old_route_display.get("path", []):
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

    for stop in new_route_display.get("path", []):
        lat = stop.get("latitude")
        lon = stop.get("longitude")
        if lat is None or lon is None:
            continue
        if str(stop.get("type", "pickup")).lower() == "school":
            continue
        folium.CircleMarker(
            location=(lat, lon),
            radius=5,
            color="#2E7D32",
            fill=True,
            fill_color="#2E7D32",
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
        tooltip=f"Changed student location ({student_label}) on route {new_route_id}",
        popup=(
            f"<b>Student location change</b><br>"
            f"Student: {student_label}<br>"
            f"Route used: {new_route_id}<br>"
            f"Lat: {float(new_lat):.7f}<br>"
            f"Lon: {float(new_lon):.7f}"
        ),
    ).add_to(m)

    route_info_html = (
        "<div style='position: fixed; top: 10px; left: 50px; z-index: 9999; "
        "background: white; border: 1px solid #d1d5db; border-radius: 6px; "
        "padding: 6px 10px; font: 12px/1.3 Arial, sans-serif; "
        "box-shadow: 0 1px 4px rgba(0,0,0,.25);'>"
        f"<b>Student:</b> {student_label}<br>"
        f"<b>Route used:</b> {new_route_id}<br>"
        "<b>Path:</b> School → Stops → School"
        "</div>"
    )
    m.get_root().html.add_child(folium.Element(route_info_html))

    if new_coords:
        m.fit_bounds(new_coords)

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
    school = _resolve_school_from_base(base, base_routes_path)

    for route in routes:
        _ensure_school_bookends(route, school)
        _recompute_route_stats(route)

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
        student_id=student_id,
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
