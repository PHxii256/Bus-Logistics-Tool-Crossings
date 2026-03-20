"""
Visualization module for routes, students, and bus stops
Creates an enhanced Folium map showing:
- Road network (colored by safety)
- Bus routes (sequences of stops)
- Student home locations
- Bus stop boarding points
- Walking paths from student homes to stops
"""

import folium
from folium import plugins
import networkx as nx
import osmnx as ox
import math
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from detour_engine import (
    calculate_student_ride_time, 
    calculate_route_path_and_stats,
    calculate_route_time_from_matrix,
    find_shortest_path_with_turns,
    walk_distance_on_roads,
    walk_path_on_roads,
    calculate_walk_penalty,
    compute_direct_time,
    compute_student_tmax,
    _MATRIX_CACHE,
)


def get_address_from_coords(lat, lon):
    """Reverse geocode coordinates to an address string."""
    try:
        geolocator = Nominatim(user_agent="bus_logistics_tool")
        location = geolocator.reverse((lat, lon), language='en')
        return location.address if location else "Address not found"
    except Exception as e:
        print(f"Geocoding error: {e}")
        return "Address lookup error"


def _compute_pm_ride_time(route, stop, G):
    """Afternoon (school→home) ride time for a student at `stop`.

    PM route = school + reversed interior stops + school.
    The student is dropped at their stop; ride = sum of legs from school
    to that stop's position in the reversed sequence.
    """
    interior = [s for s in route.stops if s.stop_type != 'school']
    afternoon = [route.stops[0]] + interior[::-1] + [route.stops[-1]]
    target_idx = next((i for i, s in enumerate(afternoon) if s is stop), -1)
    if target_idx <= 0:
        return 0.0
    total = 0.0
    for i in range(target_idx):
        u = afternoon[i].node_id
        v = afternoon[i + 1].node_id
        t = _MATRIX_CACHE.get((u, v), None)
        if t is None:
            _, t = find_shortest_path_with_turns(G, u, v)
        if not math.isfinite(t):
            return float('inf')
        total += t
    return total


def _dir_cap_html(label, ride, direct, cap, k):
    """Build a compact per-direction ride-cap HTML block."""
    if direct is None or direct <= 0 or not math.isfinite(ride):
        ride_str = f"{ride:.1f}" if math.isfinite(ride) else "∞"
        return (
            f'<div style="margin:3px 0; font-size:11px;">'
            f'  <b>{label}:</b> ride&nbsp;<b>{ride_str}&nbsp;min</b> — direct N/A'
            f'</div>'
        )
    ratio = ride / direct if direct > 0 else 0
    ratio_color = 'green' if ratio <= 1.5 else ('darkorange' if ratio <= k else 'red')
    cap_safe = cap if math.isfinite(cap) else 999
    usage_pct = min(100, int(ride / cap_safe * 100)) if cap_safe > 0 else 0
    over = ride > cap
    status = '✖ over cap' if over else '✔ ok'
    status_color = 'red' if over else 'green'
    return (
        f'<div style="margin:3px 0; font-size:11px;border-left:3px solid {ratio_color};padding-left:4px;">'
        f'  <b>{label}:</b> '
        f'  ride <b style="color:{ratio_color};">{ride:.1f}</b> / cap <b>{cap:.1f}</b> min'
        f'  &nbsp;<span style="color:{ratio_color};">({ratio:.2f}×)</span>'
        f'  <span style="color:{status_color}; float:right;">{status}</span><br>'
        f'  direct {direct:.1f} min'
        f'  <div style="background:#eee;border-radius:3px;height:5px;margin-top:2px;">'
        f'    <div style="background:{ratio_color};width:{usage_pct}%;height:5px;border-radius:3px;"></div>'
        f'  </div>'
        f'</div>'
    )


def _add_crossing_usage_markers(feature_group, solution, drive_graph):
    """Add synthetic crossing usage markers to a feature group.

    Args:
        feature_group: Folium FeatureGroup to add markers to
        solution: ServiceSolution object
        drive_graph: Drive graph for path extraction
    """
    from detour_engine import (
        get_crossing_usage_from_solution,
        _get_crossing_geometry_and_midpoint,
        _get_walk_graph,
    )

    # Get walk graph (with synthetic crossings)
    try:
        walk_graph = _get_walk_graph(drive_graph)
    except Exception as e:
        print(f"Warning: Could not get walk graph for crossing visualization: {e}")
        return

    # Get crossing usage data
    try:
        crossing_usage = get_crossing_usage_from_solution(solution, drive_graph, walk_graph)
    except Exception as e:
        print(f"Warning: Could not extract crossing usage: {e}")
        return

    if not crossing_usage:
        print("No crossing usage found in solution")
        return

    # Draw each crossing
    for (u, v), usage_data in crossing_usage.items():
        try:
            # Get crossing geometry and midpoint
            geom_data = _get_crossing_geometry_and_midpoint(walk_graph, u, v)

            if not geom_data.get("coords"):
                continue

            # Draw white halo
            folium.PolyLine(
                locations=geom_data["coords"],
                color="#ffffff",
                weight=8,
                opacity=0.85,
                popup=None,
                tooltip=None
            ).add_to(feature_group)

            # Draw magenta core
            folium.PolyLine(
                locations=geom_data["coords"],
                color="#c2185b",
                weight=5,
                opacity=0.95,
                popup=None,
                tooltip=None
            ).add_to(feature_group)

            # Build popup HTML
            student_ids_str = ", ".join(usage_data.get("students", [])[:10])
            if len(usage_data.get("students", [])) > 10:
                student_ids_str += f", ... +{len(usage_data['students']) - 10} more"

            home_count = len(usage_data.get("homes", []))
            popup_html = f"""
            <div style="width: 300px; font-size: 12px;">
                <b>Synthetic Crossing</b><br>
                <hr style="margin: 5px 0;">
                <b>Length:</b> {geom_data.get('length_m', 0):.1f} m<br>
                <b>Location:</b> {geom_data.get('lat', 0):.5f}, {geom_data.get('lon', 0):.5f}<br>
                <hr style="margin: 5px 0;">
                <b>Students using crossing:</b> {student_ids_str}<br>
                <b>Number of students:</b> {usage_data.get('count', 0)}<br>
                <b>Number of homes:</b> {home_count}<br>
                <hr style="margin: 5px 0;">
                <b>Impact:</b> {usage_data.get('count', 0)} student(s) can reach stops on opposite side of dual carriageway
            </div>
            """

            # Add marker at midpoint
            folium.CircleMarker(
                location=(geom_data.get("lat", 0), geom_data.get("lon", 0)),
                radius=8,
                color="#4a148c",
                fill=True,
                fillColor="#ffeb3b",
                fillOpacity=0.95,
                weight=2,
                popup=folium.Popup(popup_html, max_width=400),
                tooltip=f"Crossing ({geom_data.get('length_m', 0):.0f}m) - {usage_data.get('count', 0)} students from {home_count} homes"
            ).add_to(feature_group)

        except Exception as e:
            print(f"Warning: Could not render crossing ({u}, {v}): {e}")
            continue


def create_route_map(G, routes, students_to_routes=None, all_students=None, school_coords=None, output_file='route_map.html', solution=None, show_crossing_usage=True):
    """Create a comprehensive map showing routes, stops, and students.

    Args:
        G: NetworkX road network graph with 'travel_time' and 'is_safe_to_cross' attributes
        routes: List of Route objects
        students_to_routes: Dict mapping student id to route (for finding assignments)
        all_students: List of all Student objects (to identify unserved ones)
        school_coords: Dict with school location coordinates
        output_file: Output HTML file name
        solution: Optional ServiceSolution object for crossing usage visualization
        show_crossing_usage: Whether to show synthetic crossing usage overlay (requires solution)

    Returns:
        Folium Map object
    """
    
    # Calculate map center
    lats = [G.nodes[node]['y'] for node in G.nodes()]
    lons = [G.nodes[node]['x'] for node in G.nodes()]
    center_lat = (min(lats) + max(lats)) / 2
    center_lon = (min(lons) + max(lons)) / 2
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles='OpenStreetMap'
    )
    
    # Color palette for routes
    route_colors = ['blue', 'red', 'green', 'purple', 'orange', 'darkblue', 'darkred', 'darkgreen']
    
    # Add school location if provided
    if school_coords:
        folium.Marker(
            location=(school_coords.get('latitude', school_coords.get('lat')), 
                     school_coords.get('longitude', school_coords.get('lon'))),
            popup="<b>SCHOOL</b>",
            tooltip="School Location",
            icon=folium.Icon(color='darkgreen', icon='graduation-cap', prefix='fa')
        ).add_to(m)
    
    # Track all students and their stops for visualization
    student_stop_map = {}  # student_id -> stop object
    
    # Add routes, stops, and build student mapping
    for route_idx, route in enumerate(routes):
        route_color = route_colors[route_idx % len(route_colors)]
        
        # Get school coordinates for comparison
        school_lat = school_coords.get('latitude', school_coords.get('lat')) if school_coords else None
        school_lon = school_coords.get('longitude', school_coords.get('lon')) if school_coords else None
        
        # Draw route path connecting stops along actual road network
        if len(route.stops) > 1:
            # Calculate the ENTIRE route path to ensure turns across stops are handled correctly
            # Use travel_time weight to match the engine's choice of path
            full_path_nodes, _ = calculate_route_path_and_stats(G, route.stops, weight='travel_time')
            
            if full_path_nodes:
                # Convert node IDs to coordinates using edge geometries for precise road following
                coord_offset = 0.00003 * (route_idx - 0.5)
                path_coords = []
                
                for i in range(len(full_path_nodes) - 1):
                    u, v = full_path_nodes[i], full_path_nodes[i+1]
                    # Get edge data (handle multigraph keys)
                    data = G.get_edge_data(u, v)
                    if not data: continue
                    
                    # Pick the first available edge (standard for routing)
                    edge_data = data[0] if 0 in data else list(data.values())[0]
                    
                    if 'geometry' in edge_data:
                        # Extract all intermediate points from the road geometry
                        # This prevents the line from "cutting through" buildings
                        for lon, lat in edge_data['geometry'].coords:
                            path_coords.append((lat + coord_offset, lon + coord_offset))
                    else:
                        # Fallback for straight roads
                        path_coords.append((G.nodes[u]['y'] + coord_offset, G.nodes[u]['x'] + coord_offset))
                
                # Add the final node
                last_node = full_path_nodes[-1]
                path_coords.append((G.nodes[last_node]['y'] + coord_offset, G.nodes[last_node]['x'] + coord_offset))
                
                # Draw the full path on the map
                path_polyline = folium.PolyLine(
                    path_coords,
                    color=route_color,
                    weight=5,
                    opacity=0.8,
                    popup=folium.Popup(f"Route {route.route_id} complete path", max_width=450),
                    dash_array='1, 0'
                ).add_to(m)
                
                # Add directional arrows
                folium.plugins.PolyLineTextPath(
                    path_polyline,
                    '          \u27A4          ',
                    repeat=True,
                    offset=6,
                    attributes={'fill': route_color, 'font-weight': 'bold', 'font-size': '24'}
                ).add_to(m)
            else:
                # Fallback to straight lines if path calculation fails
                straight_coords = [(s.coords[0], s.coords[1]) for s in route.stops]
                folium.PolyLine(
                    straight_coords,
                    color='gray',
                    weight=2,
                    opacity=0.5,
                    dash_array='5, 5',
                    popup=f"Route {route.route_id}: Path calculation failed (Check for disconnected road network)"
                ).add_to(m)
        
        # Add stop markers
        for stop_idx, stop in enumerate(route.stops):
            student_count = len(stop.students)
            stop_type = stop.stop_type.capitalize()  # "School" or "Pickup"
            
            # Create student list for popup
            student_list_html = ""
            if stop.students:
                student_list_html = f"<br><b>Stop Type:</b> {stop_type}"
                student_list_html += "<br><b>Students:</b><ul style='margin: 0; padding-left: 15px;'>"
                for s in stop.students:
                    student_list_html += f"<li>{s.id} ({s.school_stage.name})</li>"
                student_list_html += "</ul>"
            else:
                student_list_html = f"<br><b>Stop Type:</b> {stop_type}"
            
            # Check if this is a school stop (start or end)
            is_school_start = (stop_idx == 0)
            is_school_end = (stop_idx == len(route.stops) - 1)
            is_school_stop = False
            
            if school_coords:
                school_lat = school_coords.get('latitude', school_coords.get('lat'))
                school_lon = school_coords.get('longitude', school_coords.get('lon'))
                # Check if stop coords match school coords (within small tolerance)
                if (abs(stop.coords[0] - school_lat) < 0.0005 and 
                    abs(stop.coords[1] - school_lon) < 0.0005):
                    is_school_stop = True
            
            if is_school_stop:
                label = "SCHOOL (Start)" if is_school_start else "SCHOOL (End)"
                if is_school_start and is_school_end: label = "SCHOOL (Start/End)"
                
                # Special styling for school stops: larger marker with label
                folium.CircleMarker(
                    location=(stop.coords[0], stop.coords[1]),
                    radius=12,
                    popup=folium.Popup(f"<b>{label}: {route.route_id}</b><br>Coords: {stop.coords[0]:.6f}, {stop.coords[1]:.6f}<br>Count: {student_count}{student_list_html}", max_width=200),
                    tooltip=f"{route.route_id} {label}",

                    color='darkgreen',
                    fill=True,
                    fillColor='lightgreen',
                    fillOpacity=0.9,
                    weight=3
                ).add_to(m)
                
                # Add a label showing student count at school stop
                folium.Marker(
                    location=(stop.coords[0], stop.coords[1]),
                    popup=folium.Popup(f"<b>{route.route_id}: SCHOOL</b><br>Coords: {stop.coords[0]:.6f}, {stop.coords[1]:.6f}<br>{student_count} students{student_list_html}", max_width=200),
                    icon=folium.Icon(
                        color='green',
                        icon_color='white',
                        prefix='fa',
                        icon='users'
                    )
                ).add_to(m)
            else:
                # Regular stop marker
                folium.CircleMarker(
                    location=(stop.coords[0], stop.coords[1]),
                    radius=8,
                    popup=folium.Popup(f"<b>Stop: {route.route_id}-{stop_idx}</b><br>Coords: {stop.coords[0]:.6f}, {stop.coords[1]:.6f}<br>Count: {student_count}{student_list_html}", max_width=200),
                    tooltip=f"{route.route_id} Stop {stop_idx} ({student_count} students)",

                    color=route_color,
                    fill=True,
                    fillColor=route_color,
                    fillOpacity=0.8,
                    weight=2
                ).add_to(m)
            
            # Record students at this stop
            for student in stop.students:
                student_stop_map[student.id] = (stop, route, route_color)
    
    # Add student home locations and walking paths to stops
    for student_id, (stop, route, route_color) in student_stop_map.items():
        # Find the student object directly from their assigned stop
        student_obj = next((s for s in stop.students if s.id == student_id), None)
        
        if student_obj:
            # Get address from Nominatim
            print(f"Geocoding address for student {student_obj.id}...")
            address_name = get_address_from_coords(student_obj.coords[0], student_obj.coords[1])
            
            # Calculate AM ride time (home-stop → school) via matrix-cache sum
            ride_time_am = 0.0
            stop_idx = -1
            for i, s in enumerate(route.stops):
                if s == stop:
                    stop_idx = i
                    break
            if stop_idx != -1:
                ride_time_am = calculate_route_time_from_matrix(route.stops[stop_idx:], G)
                if ride_time_am >= 9999:
                    ride_time_am = float('inf')

            # Calculate PM ride time (school → home-stop, reversed route)
            ride_time_pm = _compute_pm_ride_time(route, stop, G)

            # Direct times: AM = home→school, PM = school→home (may differ on one-way roads)
            student_node = ox.nearest_nodes(G, student_obj.coords[1], student_obj.coords[0])
            school_node  = route.stops[-1].node_id

            direct_am = None  # home → school
            direct_pm = None  # school → home
            try:
                _, direct_am = find_shortest_path_with_turns(G, student_node, school_node, weight='travel_time')
                if not math.isfinite(direct_am):
                    direct_am = None
            except Exception:
                pass
            try:
                _, direct_pm = find_shortest_path_with_turns(G, school_node, student_node, weight='travel_time')
                if not math.isfinite(direct_pm):
                    direct_pm = None
            except Exception:
                pass

            # Per-student caps (use each direction's own direct time)
            k           = getattr(route, 'ride_time_multiplier', 2.5)
            floor_min   = getattr(route, 'floor_minutes',        45)
            ceiling_min = getattr(route, 'ceiling_minutes',      60)

            def _cap(d):
                if d is None or d <= 0:
                    return float('inf')
                return max(floor_min, min(k * d, d + ceiling_min))

            cap_am = _cap(direct_am)
            cap_pm = _cap(direct_pm)

            ride_cap_html = (
                _dir_cap_html('🟠 AM home→school', ride_time_am, direct_am, cap_am, k) +
                _dir_cap_html('🟦 PM school→home', ride_time_pm, direct_pm, cap_pm, k)
            )

            # Calculate walking distance along roads (undirected - ignores one-way)
            # Pedestrians walk on sidewalks/roads without U-turn or direction rules
            student_nearest_node = ox.nearest_nodes(G, student_obj.coords[1], student_obj.coords[0])
            walk_dist_m = walk_distance_on_roads(G, student_nearest_node, stop.node_id)
            if walk_dist_m == float('inf'):
                walk_dist_m = 0.0  # Fallback if graph disconnected
            
            # Calculate walk penalty for popup display
            walk_penalty, _, walk_over_limit = calculate_walk_penalty(
                student_obj, stop.node_id, G
            )

            walk_warning_html = ""
            if student_obj.walk_radius > 0:
                if walk_over_limit and walk_penalty < float('inf'):
                    walk_info = f"{walk_dist_m:.0f}m / {student_obj.walk_radius}m"
                    walk_warning_html = f'<br><b style="color:orange;">Walking beyond limit (+{walk_penalty:.1f} min penalty)</b>'
                elif walk_penalty == float('inf'):
                    walk_info = f"{walk_dist_m:.0f}m / {student_obj.walk_radius}m"
                    walk_warning_html = f'<br><b style="color:red;">Beyond absolute walk maximum!</b>'
                else:
                    walk_info = f"{walk_dist_m:.0f}m / {student_obj.walk_radius}m"
            else:
                walk_info = f"{walk_dist_m:.0f}m (Door-to-Door Required)"
                if walk_dist_m > 20:  # 20m tolerance for snapping to nearest road
                    walk_warning_html = f'<br><b style="color:red;">House far from stop ({walk_dist_m:.0f}m)</b>'

            # Temporary students get a distinct red/orange marker so they stand out
            is_temporary = getattr(student_obj, 'assignment', 'permanent') == 'temporary'
            home_icon = folium.Icon(
                color='orange' if is_temporary else 'blue',
                icon='home',
                prefix='fa'
            )
            assignment_label = (
                f'<br><b style="color:darkorange;">Temporary: {student_obj.valid_from} → {student_obj.valid_until}</b>'
                if is_temporary else ''
            )

            # Marker for student home
            folium.Marker(
                location=student_obj.coords,
                popup=folium.Popup(f"""
                <div style="width: 280px; font-size:12px;">
                    <b>Student: {student_obj.id}</b>{assignment_label}<br>
                    Stage: {student_obj.school_stage.name}<br>
                    Home: {student_obj.coords[0]:.6f}, {student_obj.coords[1]:.6f}<br>
                    Address: {address_name}<br>
                    <div style="margin-top:5px; border-top:1px solid #ccc; padding-top:5px;">
                        <b>Assigned Route:</b> [ {route.route_id} ]<br>
                        {ride_cap_html}
                        <div style="margin-top:4px; font-size:11px;">
                            Walk to Stop: {walk_info} {walk_warning_html}
                        </div>
                    </div>
                </div>
                """, max_width=500),
                tooltip=f"{'[TEMP] ' if is_temporary else ''}Home: {student_obj.id} ({student_obj.coords[0]:.5f}, {student_obj.coords[1]:.5f})",
                icon=home_icon
            ).add_to(m)
            
            # Draw walking path along roads (undirected shortest path)
            if walk_dist_m > 5:  # Only draw if meaningful distance
                walk_line_color = 'orange' if walk_over_limit else route_color
                
                # Get road-following walk path
                walk_path_nodes = walk_path_on_roads(G, student_nearest_node, stop.node_id)
                
                if walk_path_nodes and len(walk_path_nodes) >= 2:
                    # Build coordinates following road geometry
                    walk_coords = []
                    for wi in range(len(walk_path_nodes) - 1):
                        u, v = walk_path_nodes[wi], walk_path_nodes[wi + 1]
                        # Try both edge directions (undirected path may reference either)
                        data = G.get_edge_data(u, v) or G.get_edge_data(v, u)
                        if data:
                            edge_data = data[0] if 0 in data else list(data.values())[0]
                            if 'geometry' in edge_data:
                                geom_coords = list(edge_data['geometry'].coords)
                                # Ensure correct direction
                                node_u_x = G.nodes[u]['x'] if u in G.nodes else None
                                if node_u_x is not None and len(geom_coords) > 1:
                                    first_x = geom_coords[0][0]
                                    if abs(first_x - node_u_x) > 0.0001:
                                        geom_coords = list(reversed(geom_coords))
                                for lon, lat in geom_coords:
                                    walk_coords.append((lat, lon))
                            else:
                                walk_coords.append((G.nodes[u]['y'], G.nodes[u]['x']))
                        else:
                            walk_coords.append((G.nodes[u]['y'], G.nodes[u]['x']))
                    # Add final node
                    last_node = walk_path_nodes[-1]
                    walk_coords.append((G.nodes[last_node]['y'], G.nodes[last_node]['x']))
                    
                    folium.PolyLine(
                        locations=walk_coords,
                        color=walk_line_color,
                        weight=2,
                        opacity=0.6,
                        dash_array='8, 6',
                        popup=folium.Popup(
                            f"<div style='width:250px;'>"
                            f"<b>Walking Path</b><br>"
                            f"<b>Student:</b> {student_obj.id} ({student_obj.school_stage.name})<br>"
                            f"<b>Distance:</b> {walk_dist_m:.0f}m (along roads)<br>"
                            f"<b>Walk Limit:</b> {student_obj.walk_radius}m"
                            f"{'<br><b>Penalty:</b> +' + f'{walk_penalty:.1f} min' if walk_over_limit else ''}"
                            f"</div>",
                            max_width=300
                        )
                    ).add_to(m)
                else:
                    # Fallback: straight dashed line if no path found
                    folium.PolyLine(
                        locations=[student_obj.coords, stop.coords],
                        color=walk_line_color,
                        weight=2,
                        opacity=0.4,
                        dash_array='8, 6'
                    ).add_to(m)

    # Add synthetic crossing usage visualization
    if solution and show_crossing_usage:
        crossing_fg = folium.FeatureGroup(name="Synthetic Crossing Usage", show=True)
        _add_crossing_usage_markers(crossing_fg, solution, G)
        crossing_fg.add_to(m)

        # Enable LayerControl if not already present
        if not hasattr(m, '_has_layer_control'):
            folium.LayerControl().add_to(m)
            m._has_layer_control = True

    # Add unserved students (failed assignments)
    if all_students:
        for student in all_students:
            if not student.is_served:
                print(f"Geocoding address for unassigned student {student.id}...")
                address_name = get_address_from_coords(student.coords[0], student.coords[1])
                failure_msg = getattr(student, 'failure_reason', 'No valid insertion found')

                folium.Marker(
                    location=student.coords,
                    popup=folium.Popup(f"""
                    <div style="width: 300px;">
                        <b style="color:red;">UNASSIGNED: {student.id}</b><br>
                        Stage: {student.school_stage.name}<br>
                        Address: {address_name}<br>
                        <div style="margin-top:5px; border-top:1px solid #ccc; padding-top:5px;">
                            <b>Reason for Failure:</b><br>
                            <span style="color:darkred;">{failure_msg}</span>
                        </div>
                    </div>
                    """, max_width=400),
                    tooltip=f"UNASSIGNED: {student.id}",
                    icon=folium.Icon(color='red', icon='times', prefix='fa')
                ).add_to(m)

    # Calculate total served students and total pool
    total_assigned = 0
    total_pool = len(all_students) if all_students else 0
    
    if all_students:
        total_assigned = sum(1 for s in all_students if s.is_served)
    else:
        # Fallback: sum of students in displayed routes
        total_assigned = sum(sum(len(stop.students) for stop in r.stops) for r in routes)
    
    # Add legend and statistics
    stats_html = ""
    for route_idx, route in enumerate(routes):
        route_color = route_colors[route_idx % len(route_colors)]
        student_count = sum(len(stop.students) for stop in route.stops)
        
        # Calculate student ride time (1st pickup to school)
        ride_time = calculate_student_ride_time(route, G)
        
        stats_html += f"""
        <div style="margin-top: 10px; border-top: 1px solid #ddd; padding-top: 5px;">
            <p style="margin: 0; font-weight: bold; color: {route_color};">Route {route.route_id}</p>
            <p style="margin: 0; font-size: 12px;">
                Stops: {len(route.stops) - 2} | Students: {student_count} / {total_pool}<br>
                Distance: {route.total_distance:.2f} km<br>
                Max Student Ride Time: {ride_time:.1f} min<br>
            </p>
        </div>
        """

    unassigned_count = total_pool - total_assigned
    unassigned_text = f"<br><span style='color:red; font-size:11px;'>({unassigned_count} unassigned)</span>" if unassigned_count > 0 else ""

    legend_html = f'''
    <div style="position: fixed; 
                bottom: 20px; right: 20px; width: 320px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px; border-radius: 5px;">
        <p style="margin: 0; font-weight: bold;">Map Legend</p>
        <div style="font-size: 12px; border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-bottom: 5px;">
            Total Students: {total_pool} | Served: {total_assigned} {unassigned_text}
        </div>
        <p style="margin: 5px 0;">
            <span style="color: darkgreen; font-weight: bold;">●</span> School Location (Start/End)
        </p>
        <p style="margin: 5px 0;">
            <span style="color: blue;">●</span> Student Home (permanent)
        </p>
        <p style="margin: 5px 0;">
            <span style="color: orange;">●</span> Student Home (temporary change)
        </p>
        <p style="margin: 5px 0;">
            <span style="color: black;">●</span> Unassigned Student (Failed)
        </p>
        <p style="margin: 10px 0 5px 0; font-weight: bold;">Route Statistics</p>
        {stats_html}
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    m.save(output_file)
    print(f"Route map saved as '{output_file}'")
    
    return m
