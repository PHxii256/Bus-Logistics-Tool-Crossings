import argparse
import re


def inject_star_pin(input_html_path, output_html_path, latitude, longitude):
    with open(input_html_path, "r", encoding="utf-8") as f:
        html = f.read()

    match = re.search(r"var\s+(map_[A-Za-z0-9_]+)\s*=\s*L\.map\(", html)
    if not match:
        raise ValueError("Could not find Folium map variable in the provided comparison map HTML.")

    map_var = match.group(1)

    injection = f"""
<script>
(function() {{
  var mapRef = {map_var};
  if (!mapRef) return;

  var starIcon = L.divIcon({{
    className: 'insertion-star-pin',
    html: '<div style="font-size:24px; line-height:24px; color:#f59e0b; text-shadow:0 0 2px #222;">★</div>',
    iconSize: [24, 24],
    iconAnchor: [12, 12],
    popupAnchor: [0, -12]
  }});

  L.marker([{latitude}, {longitude}], {{icon: starIcon}})
    .addTo(mapRef)
    .bindPopup('<b>Change-location insertion attempt</b><br>Lat: {latitude:.7f}<br>Lon: {longitude:.7f}')
    .bindTooltip('Insertion attempt location');
}})();
</script>
"""

    if "</body>" in html:
        html = html.replace("</body>", injection + "\n</body>", 1)
    else:
        html += injection

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description="Inject a star pin into an existing comparison_map.html.")
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
