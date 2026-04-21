#!/usr/bin/env python3
"""
Remove side panels from Folium comparison-map HTML files.

Panels removed:
1) Bottom-right fixed white stats box
2) Top-right route toggles/custom layer control script

Usage:
  python strip_comparison_side_panels.py --input path/to/comparison_map.html
  python strip_comparison_side_panels.py --input path/to/comparison_map.html --output path/to/cleaned.html
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


def _find_matching_div_end(html: str, start_div_idx: int) -> int:
    """Return end index (exclusive) of the div that starts at start_div_idx."""
    token_re = re.compile(r"<div\b|</div>", re.IGNORECASE)
    depth = 0
    for match in token_re.finditer(html, start_div_idx):
        token = match.group(0).lower()
        if token.startswith("<div") and not token.startswith("</"):
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return match.end()
    return -1


def _remove_bottom_right_panel(html: str) -> tuple[str, bool]:
    marker = "position:fixed; bottom:15px; right:15px;"
    idx = html.find(marker)
    if idx < 0:
        return html, False

    start = html.rfind("<div", 0, idx)
    if start < 0:
        return html, False

    end = _find_matching_div_end(html, start)
    if end < 0:
        return html, False

    return html[:start] + html[end:], True


def _remove_custom_toggle_control(html: str) -> tuple[str, int]:
    patterns = [
        # Custom "Route Views" control injected by this project.
        re.compile(
            r"\s*window\.addEventListener\('load', function\(\) \{.*?new CustomCtrl\(\)\.addTo\(m\);\s*\}\);\s*",
            re.DOTALL,
        ),
        # Generic Folium/Leaflet layer control attachment.
        re.compile(
            r"\s*L\.control\.layers\(.*?\)\.addTo\(.*?\);\s*",
            re.DOTALL,
        ),
    ]

    removed = 0
    current = html
    for pattern in patterns:
        current, count = pattern.subn("\n", current)
        removed += count
    return current, removed


def strip_side_panels(html: str) -> tuple[str, dict]:
    cleaned, removed_bottom = _remove_bottom_right_panel(html)
    cleaned, toggle_removed_count = _remove_custom_toggle_control(cleaned)

    stats = {
        "bottom_right_panel_removed": int(removed_bottom),
        "toggle_blocks_removed": int(toggle_removed_count),
    }
    return cleaned, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove side panels from comparison map HTML.")
    parser.add_argument("--input", required=True, help="Path to input HTML.")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write cleaned HTML. If omitted, input file is overwritten.",
    )
    args = parser.parse_args()

    input_path = pathlib.Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        return 2

    output_path = pathlib.Path(args.output).expanduser().resolve() if args.output else input_path

    html = input_path.read_text(encoding="utf-8")
    cleaned, stats = strip_side_panels(html)
    output_path.write_text(cleaned, encoding="utf-8")

    print(f"[OK] Cleaned HTML written to: {output_path}")
    print(f"  bottom-right panel removed: {stats['bottom_right_panel_removed']}")
    print(f"  toggle control blocks removed: {stats['toggle_blocks_removed']}")

    if stats["bottom_right_panel_removed"] == 0 and stats["toggle_blocks_removed"] == 0:
        print("[WARN] No known side-panel patterns were found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
