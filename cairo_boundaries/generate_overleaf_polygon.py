#!/usr/bin/env python3
"""Extract Cairo boundary coordinates from shapefile and build an Overleaf-ready plot.

Outputs in the same folder:
- cairo_boundary_points.csv
- cairo_boundary_polygon_overleaf.tex
- cairo_boundary_summary.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import shapefile


def main() -> None:
    folder = Path(__file__).resolve().parent
    shp_path = folder / "Cairo boundries.shp"

    reader = shapefile.Reader(str(shp_path))
    if len(reader) == 0:
        raise RuntimeError("No records found in shapefile")

    shape = reader.shapes()[0]
    points = shape.points
    if not points:
        raise RuntimeError("No points found in shape geometry")

    # Ensure polygon is explicitly closed for plotting.
    if points[0] != points[-1]:
        points = points + [points[0]]

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    csv_path = folder / "cairo_boundary_points.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lon", "lat"])
        for lon, lat in points:
            writer.writerow([f"{lon:.12f}", f"{lat:.12f}"])

    summary = {
        "num_vertices": len(points),
        "bbox": {
            "min_lon": min(xs),
            "max_lon": max(xs),
            "min_lat": min(ys),
            "max_lat": max(ys),
        },
        "source": str(shp_path.name),
    }
    summary_path = folder / "cairo_boundary_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    tex_path = folder / "cairo_boundary_polygon_overleaf.tex"
    tex_content = r"""\documentclass{article}
\usepackage[margin=1in]{geometry}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}

\title{Cairo Boundary Polygon}
\date{}

\begin{document}
\maketitle

\begin{center}
\begin{tikzpicture}
\begin{axis}[
    axis equal image,
    xlabel={Longitude},
    ylabel={Latitude},
    grid=both,
    title={Cairo Boundary (from shapefile)},
    width=0.92\textwidth,
    height=0.75\textwidth,
]
    % Filled polygon
    \addplot[
        draw=black,
        line width=1pt,
        fill=blue!20,
        fill opacity=0.35,
    ] table[col sep=comma, x=lon, y=lat] {cairo_boundary_points.csv} -- cycle;

    % Optional vertex markers
    \addplot[
        only marks,
        mark=*,
        mark size=0.5pt,
        color=red,
    ] table[col sep=comma, x=lon, y=lat] {cairo_boundary_points.csv};
\end{axis}
\end{tikzpicture}
\end{center}

\end{document}
"""
    tex_path.write_text(tex_content, encoding="utf-8")

    coords_inline = "\n".join([f"        ({lon:.12f}, {lat:.12f})" for lon, lat in points])
    tex_inline_path = folder / "cairo_boundary_polygon_overleaf_inline.tex"
    tex_inline = (
        "\\documentclass{article}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\usepackage{pgfplots}\n"
        "\\pgfplotsset{compat=1.18}\n\n"
        "\\title{Cairo Boundary Polygon (Inline Coordinates)}\n"
        "\\date{}\n\n"
        "\\begin{document}\n"
        "\\maketitle\n\n"
        "\\begin{center}\n"
        "\\begin{tikzpicture}\n"
        "\\begin{axis}[\n"
        "    axis equal image,\n"
        "    xlabel={Longitude},\n"
        "    ylabel={Latitude},\n"
        "    grid=both,\n"
        "    title={Cairo Boundary (from shapefile)},\n"
        "    width=0.92\\textwidth,\n"
        "    height=0.75\\textwidth,\n"
        "]\n"
        "    \\addplot[draw=black, line width=1pt, fill=blue!20, fill opacity=0.35] coordinates {\n"
        f"{coords_inline}\n"
        "    } -- cycle;\n"
        "\\end{axis}\n"
        "\\end{tikzpicture}\n"
        "\\end{center}\n\n"
        "\\end{document}\n"
    )
    tex_inline_path.write_text(tex_inline, encoding="utf-8")

    # Leaflet HTML viewer with embedded coordinates.
    leaflet_path = folder / "cairo_boundary_leaflet.html"
    coords_js = ",\n".join([f"      [{lat:.12f}, {lon:.12f}]" for lon, lat in points])
    leaflet_html = (
        "<!doctype html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\" />\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n"
        "  <title>Cairo Boundary Leaflet Viewer</title>\n"
        "  <link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\"\n"
        "        integrity=\"sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=\" crossorigin=\"\" />\n"
        "  <style>\n"
        "    html, body { height: 100%; margin: 0; }\n"
        "    #map { height: 100%; width: 100%; }\n"
        "    .info {\n"
        "      position: absolute;\n"
        "      top: 10px;\n"
        "      right: 10px;\n"
        "      z-index: 1000;\n"
        "      background: rgba(255,255,255,0.95);\n"
        "      padding: 8px 10px;\n"
        "      border-radius: 6px;\n"
        "      font: 13px/1.4 sans-serif;\n"
        "      box-shadow: 0 2px 10px rgba(0,0,0,0.15);\n"
        "    }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <div id=\"map\"></div>\n"
        "  <div class=\"info\">\n"
        f"    <div><b>Vertices:</b> {len(points)}</div>\n"
        f"    <div><b>Lon:</b> {min(xs):.6f} to {max(xs):.6f}</div>\n"
        f"    <div><b>Lat:</b> {min(ys):.6f} to {max(ys):.6f}</div>\n"
        "  </div>\n"
        "\n"
        "  <script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"\n"
        "          integrity=\"sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=\" crossorigin=\"\"></script>\n"
        "  <script>\n"
        "    const polygonLatLngs = [\n"
        f"{coords_js}\n"
        "    ];\n"
        "\n"
        "    const map = L.map('map');\n"
        "    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {\n"
        "      maxZoom: 19,\n"
        "      attribution: '&copy; OpenStreetMap contributors'\n"
        "    }).addTo(map);\n"
        "\n"
        "    const polygon = L.polygon(polygonLatLngs, {\n"
        "      color: '#0d47a1',\n"
        "      weight: 2,\n"
        "      fillColor: '#42a5f5',\n"
        "      fillOpacity: 0.25\n"
        "    }).addTo(map);\n"
        "\n"
        "    const bounds = polygon.getBounds();\n"
        "    map.fitBounds(bounds, { padding: [20, 20] });\n"
        "\n"
        "    const center = bounds.getCenter();\n"
        "    L.marker(center).addTo(map).bindPopup('Cairo Boundary Center').openPopup();\n"
        "  </script>\n"
        "</body>\n"
        "</html>\n"
    )
    leaflet_path.write_text(leaflet_html, encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {tex_path}")
    print(f"Wrote: {tex_inline_path}")
    print(f"Wrote: {leaflet_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
