import argparse
import os
import re


def inject_star_pin(input_html_path, output_html_path, latitude, longitude):
    input_abs = os.path.abspath(input_html_path)
    output_abs = os.path.abspath(output_html_path)
    if input_abs == output_abs:
        raise ValueError("Output HTML must be different from source HTML to avoid modifying the source map.")

    with open(input_html_path, "r", encoding="utf-8") as f:
        html = f.read()

    match = re.search(r"var\s+(map_[A-Za-z0-9_]+)\s*=\s*L\.map\(", html)
    if not match:
        raise ValueError("Could not find Folium map variable in the provided comparison map HTML.")

    map_var = match.group(1)

    injection_js = f"""
// Injected star pin marker (same style used by process_change_location_request.py).
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

    var currentZoom = (typeof mapRef.getZoom === 'function') ? mapRef.getZoom() : 13;
    var targetZoom = Math.max(currentZoom, 15);
    if (typeof mapRef.setView === 'function') {{
        mapRef.setView([{latitude}, {longitude}], targetZoom, {{animate: false}});
    }}
}})();
"""

    script_head, script_sep, script_tail = html.rpartition("</script>")
    if script_sep:
        html = script_head + "\n" + injection_js + "\n</script>" + script_tail
    else:
        html += "\n<script>\n" + injection_js + "\n</script>\n"

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(
        description="Inject a star pin into an existing comparison_map.html and write a new output HTML file."
    )
    parser.add_argument("--comparison-map-html", required=True, help="Path to existing comparison_map.html")
    parser.add_argument("--latitude", required=True, type=float, help="Latitude for inserted marker")
    parser.add_argument("--longitude", required=True, type=float, help="Longitude for inserted marker")
    parser.add_argument("--output", default="map_insertion_attempt.html", help="Output HTML path")
    args = parser.parse_args()

    inject_star_pin(
        input_html_path=args.comparison_map_html,
        output_html_path=args.output,
        latitude=args.latitude,
        longitude=args.longitude,
    )
    print(f"Injected map saved to: {args.output}")


if __name__ == "__main__":
    main()
