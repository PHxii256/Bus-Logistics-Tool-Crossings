#!/usr/bin/env python3
"""
Quick bbox preview - draws bounding box on map without downloading road network.

Usage:
    python preview_bbox.py
    
Output:
    bbox_preview.html
"""

import folium

# Read bbox from run_algorithm.py or override here
# FORMAT: [min_lat, min_lon, max_lat, max_lon]
# FORMAT: [min_lat, min_lon, max_lat, max_lon]
# South-West Corner: Al Abadiah Al Bahriya (29.925630, 31.229084)
# North-East Corner: International Park (30.051972, 31.338611)
_DEFAULT_BBOX = [29.925630, 31.229084, 30.051972, 31.338611]
_DEFAULT_BBOX = [31.229084, 29.925630, 31.33660186220229, 30.048847047121367]

# School location (Victory College School)
SCHOOL = {"name": "Victory College School", "lat": 29.964406, "lon": 31.270319}

# International Park (Nasr City) - for reference
INTL_PARK = {"name": "International Park", "lat": 30.0475, "lon": 31.3375}


def draw_bbox_preview(bbox, school, landmarks=None, output="bbox_preview.html"):
    """Draw bbox rectangle on Folium map with school marker."""
    
    # Calculate center for initial view
    center_lat = (bbox[0] + bbox[2]) / 2
    center_lon = (bbox[1] + bbox[3]) / 2
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap")
    
    # Draw bbox rectangle
    # bbox format: [min_lat, min_lon, max_lat, max_lon]
    # Folium PolyLine needs: [[lat, lon], [lat, lon], ...]
    bbox_coords = [
        [bbox[2], bbox[1]],  # top-left: (max_lat, min_lon)
        [bbox[2], bbox[3]],  # top-right: (max_lat, max_lon)
        [bbox[0], bbox[3]],  # bottom-right: (min_lat, max_lon)
        [bbox[0], bbox[1]],  # bottom-left: (min_lat, min_lon)
        [bbox[2], bbox[1]],  # close the loop
    ]
    
    folium.PolyLine(
        bbox_coords,
        color="#3498db",
        weight=4,
        opacity=0.8,
        dash_array="10,5",
        tooltip=f"Bbox: N={bbox[2]:.4f}, S={bbox[0]:.4f}, E={bbox[3]:.4f}, W={bbox[1]:.4f}",
    ).add_to(m)
    
    # Add school marker
    folium.Marker(
        location=[school["lat"], school["lon"]],
        popup=f"<b>{school['name']}</b>",
        tooltip=school['name'],
        icon=folium.Icon(color="green", icon="graduation-cap", prefix="fa"),
    ).add_to(m)
    
    # Add landmark markers if provided
    if landmarks:
        for landmark in landmarks:
            folium.Marker(
                location=[landmark["lat"], landmark["lon"]],
                popup=f"<b>{landmark['name']}</b>",
                tooltip=landmark['name'],
                icon=folium.Icon(color="red", icon="info-sign"),
            ).add_to(m)
    
    # Add corner markers for reference
    corners = [
        ("NW", bbox[2], bbox[1]),
        ("NE", bbox[2], bbox[3]),
        ("SE", bbox[0], bbox[3]),
        ("SW", bbox[0], bbox[1]),
    ]
    
    for label, lat, lon in corners:
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            popup=f"<b>{label} Corner</b><br>Lat: {lat:.6f}<br>Lon: {lon:.6f}",
            tooltip=f"{label} ({lat:.4f}, {lon:.4f})",
            color="#3498db",
            fill=True,
            fillColor="#3498db",
            fillOpacity=0.7,
        ).add_to(m)
    
    # Save map
    m.save(output)
    print(f"✓ Bbox preview saved to: {output}")
    print(f"  Bbox: N={bbox[2]:.6f}, S={bbox[0]:.6f}, E={bbox[3]:.6f}, W={bbox[1]:.6f}")
    print(f"  Center: {center_lat:.6f}, {center_lon:.6f}")
    print(f"  School: {school['lat']:.6f}, {school['lon']:.6f}")


if __name__ == "__main__":
    draw_bbox_preview(
        _DEFAULT_BBOX,
        SCHOOL,
        landmarks=[INTL_PARK],
        output="bbox_preview.html",
    )
