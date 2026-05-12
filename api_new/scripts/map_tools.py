import copy
import json
import os
import re
import urllib.parse
import urllib.request

import folium
from folium import plugins


def _route_icon_class(route_color):
    safe = str(route_color or "").lstrip("#").lower()
    return f"route-{safe}" if safe else "blue"


def _ensure_route_icon_css(map_obj, route_color):
    if map_obj is None or not route_color:
        return
    used = getattr(map_obj, "_route_icon_colors", None)
    if used is None:
        used = set()
        map_obj._route_icon_colors = used
    if route_color in used:
        return
    class_name = _route_icon_class(route_color)
    css = (
        "<style>"
        f".awesome-marker-icon-{class_name} "
        f"{{ background-color: {route_color}; border-color: {route_color}; }}"
        "</style>"
    )
    map_obj.get_root().header.add_child(folium.Element(css))
    used.add(route_color)


def inject_star_pin(input_html_path, output_html_path, latitude, longitude):
    input_abs = os.path.abspath(input_html_path)
    output_abs = os.path.abspath(output_html_path)
    if input_abs == output_abs:
        raise ValueError("Output HTML must be different from source HTML.")

    with open(input_html_path, "r", encoding="utf-8") as handle:
        html = handle.read()

    match = re.search(r"var\s+(map_[A-Za-z0-9_]+)\s*=\s*L\.map\(", html)
    if not match:
        raise ValueError("Could not find Folium map variable in provided HTML.")

    map_var = match.group(1)
    injection_js = f"""
// Injected star pin marker.
(function() {{
    var mapRef = {map_var};
    if (!mapRef) return;
    if (mapRef.__injectedStarPinDone) return;
    mapRef.__injectedStarPinDone = true;

    var starMarker = L.marker([{latitude}, {longitude}], {{
        zIndexOffset: 10000,
        riseOnHover: true
    }}).addTo(mapRef);

    if (L.AwesomeMarkers && typeof L.AwesomeMarkers.icon === 'function') {{
        var starIcon = L.AwesomeMarkers.icon({{
            markerColor: 'orange',
            iconColor: 'white',
            icon: 'star',
            prefix: 'fa'
        }});
        starMarker.setIcon(starIcon);
    }}

    starMarker
        .bindPopup('<b>Student location change</b><br>Lat: {latitude:.7f}<br>Lon: {longitude:.7f}')
        .bindTooltip('Injected star pin', {{sticky: true}})
        .openPopup();

    L.circleMarker([{latitude}, {longitude}], {{
        radius: 9,
        color: '#f59e0b',
        weight: 2,
        fill: false,
        opacity: 0.95
    }}).addTo(mapRef);
}})();
"""

    script_head, script_sep, script_tail = html.rpartition("</script>")
    if script_sep:
        html = script_head + "\n" + injection_js + "\n</script>" + script_tail
    else:
        html += "\n<script>\n" + injection_js + "\n</script>\n"

    with open(output_html_path, "w", encoding="utf-8") as handle:
        handle.write(html)


def _route_polyline(path):
    coords = []
    for stop in path:
        lat = stop.get("latitude")
        lon = stop.get("longitude")
        if lat is None or lon is None:
            continue
        coords.append((float(lat), float(lon)))
    return coords


def _route_waypoints(path, use_student_homes=False):
    coords = []
    for stop in path:
        if str(stop.get("type", "pickup")).lower() == "school":
            lat = stop.get("latitude")
            lon = stop.get("longitude")
            if lat is None or lon is None:
                continue
            point = (float(lat), float(lon))
            if not coords or coords[-1] != point:
                coords.append(point)
            continue

        if use_student_homes:
            for student in stop.get("students", []):
                lat = student.get("home_latitude")
                lon = student.get("home_longitude")
                if lat is None or lon is None:
                    lat = stop.get("latitude")
                    lon = stop.get("longitude")
                if lat is None or lon is None:
                    continue
                point = (float(lat), float(lon))
                if not coords or coords[-1] != point:
                    coords.append(point)
            continue

        lat = stop.get("latitude")
        lon = stop.get("longitude")
        if lat is None or lon is None:
            continue
        point = (float(lat), float(lon))
        if not coords or coords[-1] != point:
            coords.append(point)
    return coords


def _osrm_segment_geometry(lat1, lon1, lat2, lon2, osrm_base_url="http://localhost:5000"):
    coords = f"{float(lon1)},{float(lat1)};{float(lon2)},{float(lat2)}"
    base = str(osrm_base_url).rstrip("/")
    path = f"/route/v1/driving/{coords}"
    query = urllib.parse.urlencode(
        {
            "overview": "full",
            "geometries": "geojson",
            "steps": "false",
        }
    )
    url = f"{base}{path}?{query}"

    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
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
    return [(float(lat), float(lon)) for lon, lat in geom]


def _route_polyline_on_roads(path, use_osrm=False, osrm_base_url="http://localhost:5000", use_student_homes=False):
    stops = _route_waypoints(path, use_student_homes=use_student_homes)
    if len(stops) < 2:
        return stops
    if not use_osrm:
        return stops

    merged = []
    for idx in range(len(stops) - 1):
        a_lat, a_lon = stops[idx]
        b_lat, b_lon = stops[idx + 1]
        segment = _osrm_segment_geometry(a_lat, a_lon, b_lat, b_lon, osrm_base_url=osrm_base_url)
        if not segment:
            segment = [(a_lat, a_lon), (b_lat, b_lon)]
        if not merged:
            merged.extend(segment)
        else:
            merged.extend(segment[1:])
    return merged


def _draw_school_marker(map_obj, school):
    school_lat = school.get("latitude")
    school_lon = school.get("longitude")
    if school_lat is None or school_lon is None:
        return
    folium.Marker(
        location=(school_lat, school_lon),
        icon=folium.Icon(color="green", icon="graduation-cap", prefix="fa"),
        tooltip="School",
    ).add_to(map_obj)


def _draw_student_home_markers(map_obj, route, route_label, route_color="#1D4ED8", home_color="orange"):
    seen = set()
    _ensure_route_icon_css(map_obj, route_color)
    home_icon = folium.Icon(
        color=home_color,
        icon="home",
        prefix="fa",
        icon_color="white",
    )
    route_students = route.get("students") or []
    if route_students:
        for student in route_students:
            student_id = str(student.get("id") or "Unknown")
            home_lat = student.get("home_latitude")
            home_lon = student.get("home_longitude")
            if home_lat is None or home_lon is None:
                continue
            home_key = (student_id, float(home_lat), float(home_lon))
            if home_key in seen:
                continue
            seen.add(home_key)
            folium.Marker(
                location=(float(home_lat), float(home_lon)),
                icon=home_icon,
                tooltip=f"Home of {student_id}",
                popup=(
                    f"<b>Student home</b><br>"
                    f"Student: {student_id}<br>"
                    f"Route: {route_label}<br>"
                    f"Home: {float(home_lat):.6f}, {float(home_lon):.6f}"
                ),
            ).add_to(map_obj)
        return

    for stop in route.get("path", []):
        if str(stop.get("type", "pickup")).lower() == "school":
            continue
        for student in stop.get("students", []):
            student_id = str(student.get("id") or "Unknown")
            home_lat = student.get("home_latitude")
            home_lon = student.get("home_longitude")
            if home_lat is None or home_lon is None:
                continue
            home_key = (student_id, float(home_lat), float(home_lon))
            if home_key in seen:
                continue
            seen.add(home_key)
            folium.Marker(
                location=(float(home_lat), float(home_lon)),
                icon=home_icon,
                tooltip=f"Home of {student_id}",
                popup=(
                    f"<b>Student home</b><br>"
                    f"Student: {student_id}<br>"
                    f"Route: {route_label}<br>"
                    f"Home: {float(home_lat):.6f}, {float(home_lon):.6f}"
                ),
            ).add_to(map_obj)


def build_before_map_from_routes(output_html, routes, school, new_lat, new_lon, student_id):
    center = [float(new_lat), float(new_lon)]
    if school.get("latitude") is not None and school.get("longitude") is not None:
        center = [float(school["latitude"]), float(school["longitude"])]
    map_obj = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")

    for route in routes:
        route_id = route.get("route_id", "N/A")
        coords = _route_polyline(route.get("path", []))
        if coords:
            folium.PolyLine(
                coords,
                color="#1D4ED8",
                weight=3,
                opacity=0.7,
                tooltip=f"Route {route_id} before request",
            ).add_to(map_obj)
        for stop in route.get("path", []):
            if str(stop.get("type", "pickup")).lower() == "school":
                continue
            lat = stop.get("latitude")
            lon = stop.get("longitude")
            if lat is None or lon is None:
                continue
            count = len(stop.get("students", []))
            folium.CircleMarker(
                location=(float(lat), float(lon)),
                radius=4,
                color="#1D4ED8",
                fill=True,
                fill_color="#1D4ED8",
                fill_opacity=0.7,
                tooltip=f"Route {route_id} stop ({count} students)",
            ).add_to(map_obj)

    _draw_school_marker(map_obj, school)

    folium.Marker(
        location=(float(new_lat), float(new_lon)),
        icon=folium.Icon(color="orange", icon="star", prefix="fa"),
        tooltip=f"Requested new location for {student_id}",
    ).add_to(map_obj)

    map_obj.save(output_html)


def generate_before_map(output_html, comparison_map_html, routes, school, new_lat, new_lon, student_id):
    if comparison_map_html and os.path.exists(comparison_map_html):
        try:
            inject_star_pin(
                input_html_path=comparison_map_html,
                output_html_path=output_html,
                latitude=float(new_lat),
                longitude=float(new_lon),
            )
            return "comparison_map_with_star"
        except Exception:
            pass

    build_before_map_from_routes(
        output_html=output_html,
        routes=routes,
        school=school,
        new_lat=new_lat,
        new_lon=new_lon,
        student_id=student_id,
    )
    return "generated_routes_before_map"


def _draw_walking_paths(map_obj, route, osrm_base_url="http://localhost:5000"):
    """Draw walking paths (home → stop) as green dotted lines for all students in the route."""
    seen_paths = set()
    
    for stop in route.get("path", []):
        if str(stop.get("type", "pickup")).lower() == "school":
            continue
        
        stop_lat = float(stop.get("latitude", 0))
        stop_lon = float(stop.get("longitude", 0))
        
        for student in stop.get("students", []):
            home_lat = student.get("home_latitude")
            home_lon = student.get("home_longitude")
            student_id = str(student.get("id", "unknown"))
            
            if home_lat is None or home_lon is None:
                continue
            
            home_lat = float(home_lat)
            home_lon = float(home_lon)
            
            # Avoid drawing duplicate paths
            path_key = (round(home_lat, 6), round(home_lon, 6), round(stop_lat, 6), round(stop_lon, 6))
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)
            
            # Get walking path from home to stop via OSRM
            walk_coords = _osrm_segment_geometry(home_lat, home_lon, stop_lat, stop_lon, osrm_base_url=osrm_base_url)
            if not walk_coords:
                # Fallback: straight line if OSRM fails
                walk_coords = [(home_lat, home_lon), (stop_lat, stop_lon)]
            
            # Draw green dotted walking line
            folium.PolyLine(
                walk_coords,
                color="#22C55E",
                weight=2,
                opacity=0.6,
                dash_array="5,5",
                tooltip=f"Walking path: {student_id} to stop",
            ).add_to(map_obj)


def generate_updated_route_map(
    output_html,
    old_route,
    new_route,
    school,
    new_lat,
    new_lon,
    student_id=None,
    use_osrm=False,
    optimized_path_coords=None,
):
    center = [float(new_lat), float(new_lon)]
    if school.get("latitude") is not None and school.get("longitude") is not None:
        center = [float(school["latitude"]), float(school["longitude"])]
    map_obj = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")

    old_route_display = copy.deepcopy(old_route)
    new_route_display = copy.deepcopy(new_route)

    old_route_id = str(old_route_display.get("route_id") or "N/A")
    new_route_id = str(new_route_display.get("route_id") or "N/A")
    student_label = str(student_id) if student_id is not None else "Unknown"

    old_coords = _route_polyline_on_roads(
        old_route_display.get("path", []),
        use_osrm=use_osrm,
        use_student_homes=True,
    )
    
    # Use optimized path if provided, otherwise compute from new route
    if optimized_path_coords:
        new_coords = optimized_path_coords
        geometry_source = "ALNS-optimized + OSRM road geometry"
    else:
        new_coords = _route_polyline_on_roads(
            new_route_display.get("path", []),
            use_osrm=use_osrm,
            use_student_homes=True,
        )
        route_stop_count = len(_route_waypoints(new_route_display.get("path", []), use_student_homes=True))
        geometry_source = "OSRM road geometry" if use_osrm and len(new_coords) > route_stop_count else "home-sequence fallback"

    # Only draw the previous (dashed) route when we don't have an optimized OSRM-traced geometry
    if old_coords and not optimized_path_coords:
        folium.PolyLine(
            old_coords,
            color="#6C757D",
            weight=4,
            opacity=0.8,
            dash_array="8,6",
            tooltip=f"Previous route {old_route_id}",
        ).add_to(map_obj)

    if new_coords:
        updated_polyline = folium.PolyLine(
            new_coords,
            color="#2E7D32",
            weight=5,
            opacity=0.9,
            tooltip=f"Updated route {new_route_id}",
        )
        updated_polyline.add_to(map_obj)
        plugins.PolyLineTextPath(
            updated_polyline,
            "          ->          ",
            repeat=True,
            offset=6,
            attributes={"fill": "#2E7D32", "font-weight": "bold", "font-size": "18"},
        ).add_to(map_obj)

    route_info_html = (
        "<div style='position: fixed; top: 10px; left: 50px; z-index: 9999; "
        "background: white; border: 1px solid #d1d5db; border-radius: 6px; "
        "padding: 6px 10px; font: 12px/1.3 Arial, sans-serif; "
        "box-shadow: 0 1px 4px rgba(0,0,0,.25);'>"
        f"<b>Student:</b> {student_label}<br>"
        f"<b>Route used:</b> {new_route_id}<br>"
        f"<b>Route geometry:</b> {geometry_source}<br>"
        "<b>Path:</b> School → Homes → School"
        "</div>"
    )
    map_obj.get_root().html.add_child(folium.Element(route_info_html))

    for stop in old_route_display.get("path", []):
        if str(stop.get("type", "pickup")).lower() == "school":
            continue
        lat = stop.get("latitude")
        lon = stop.get("longitude")
        if lat is None or lon is None:
            continue
        folium.CircleMarker(
            location=(float(lat), float(lon)),
            radius=4,
            color="#6C757D",
            fill=True,
            fill_color="#6C757D",
            fill_opacity=0.7,
            tooltip="Previous stop",
        ).add_to(map_obj)

    for stop in new_route_display.get("path", []):
        if str(stop.get("type", "pickup")).lower() == "school":
            continue
        lat = stop.get("latitude")
        lon = stop.get("longitude")
        if lat is None or lon is None:
            continue
        folium.CircleMarker(
            location=(float(lat), float(lon)),
            radius=5,
            color="#2E7D32",
            fill=True,
            fill_color="#2E7D32",
            fill_opacity=0.8,
            tooltip="Updated stop",
        ).add_to(map_obj)

    _draw_student_home_markers(
        map_obj,
        new_route_display,
        new_route_id,
        route_color="#2E7D32",
    )

    _draw_school_marker(map_obj, school)

    folium.Marker(
        location=(float(new_lat), float(new_lon)),
        icon=folium.Icon(color="orange", icon="star", prefix="fa"),
        tooltip=f"Changed location ({student_label})",
        popup=(
            f"<b>Student location change</b><br>"
            f"Student: {student_label}<br>"
            f"Route used: {new_route_id}<br>"
            f"Lat: {float(new_lat):.7f}<br>"
            f"Lon: {float(new_lon):.7f}"
        ),
    ).add_to(map_obj)

    # Draw walking paths from homes to stops (green dotted lines)
    _draw_walking_paths(map_obj, new_route_display)

    if new_coords:
        map_obj.fit_bounds(new_coords)

    map_obj.save(output_html)

