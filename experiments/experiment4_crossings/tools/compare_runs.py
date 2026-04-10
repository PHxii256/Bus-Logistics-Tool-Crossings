#!/usr/bin/env python3
"""Compare two experiment output.json files and generate compact reports.

Outputs:
- ASCII table (.txt)
- Markdown report (.md)

Only modes that are present and not skipped in BOTH runs are shown.

python experiments/experiment4_crossings/tools/compare_runs.py <runA/output.json> <runB/output.json>
python experiments/experiment4_crossings/tools/compare_runs.py <runA/output.json> <runB/output.json> --no-md
python experiments/experiment4_crossings/tools/compare_runs.py <runA/output.json> <runB/output.json> --no-txt
"""

import argparse
import datetime as dt
import json
import math
import os
import re
from typing import Dict, List, Tuple, Any


MODE_LABELS = {
    "strictly_constrained": "Strictly Constrained",
    "weakly_constrained": "Weakly Constrained",
    "door_to_door": "Door-to-Door",
}

# Keep this compact: user-requested metrics plus a small set of useful extras.
METRICS: List[Tuple[str, str]] = [
    ("routes_created", "Routes"),
    ("students_served", "Students Served"),
    ("students_unserved", "Students Unserved"),
    ("avg_route_time_min", "Avg Route Time (min)"),
    ("avg_ride_time_min", "Avg Ride Time (min)"),
    ("max_ride_time_min", "Max Ride Time (min)"),
    ("objective_value", "Objective Value"),
    ("total_route_time_min", "Total Route Time (min)"),
    ("total_route_dist_km", "Total Route Distance (km)"),
    ("alns_runtime_seconds", "ALNS Runtime (s)"),
]

SNAPSHOT_SECTIONS: List[str] = [
    "annulus",
    "debug",
    "algorithm",
    "walk_graph",
    "constraints",
    "buses",
    "stage_distribution",
    "n_students",
    "seed",
    "stage_walk_limits",
]


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _fmt_value(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if abs(v) >= 1000:
            return f"{v:,.2f}"
        return f"{v:.2f}"
    return str(v)


def _fmt_delta(a: Any, b: Any) -> Tuple[str, str]:
    if not (_is_number(a) and _is_number(b)):
        return "-", "-"
    delta = float(b) - float(a)
    delta_s = _fmt_value(delta)
    if abs(float(a)) < 1e-12:
        pct_s = "-"
    else:
        pct_s = f"{(delta / float(a)) * 100.0:+.2f}%"
    return delta_s, pct_s


def _safe_label(path: str, fallback: str) -> str:
    name = os.path.basename(os.path.dirname(path)) or os.path.basename(path)
    if not name:
        return fallback
    return name


def _sanitize_filename_part(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def _load_snapshot_for_output(output_json_path: str) -> Dict[str, Any] | None:
    run_dir = os.path.dirname(output_json_path)
    snapshot_path = os.path.join(run_dir, "snapshot_input.json")
    if not os.path.exists(snapshot_path):
        return None
    return _load_json(snapshot_path)


def _ordered_union_keys(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    for k in a.keys():
        if k not in keys:
            keys.append(k)
    for k in b.keys():
        if k not in keys:
            keys.append(k)
    return keys


def _canon(v: Any) -> str:
    return json.dumps(v, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)


def _fmt_compact(v: Any, max_len: int = 96) -> str:
    if v is None:
        return "-"
    if isinstance(v, (int, float, bool)):
        return _fmt_value(v)
    if isinstance(v, str):
        s = v
    else:
        s = _canon(v)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _build_snapshot_table_rows(snapshot_a: Dict[str, Any], snapshot_b: Dict[str, Any]) -> List[List[str]]:
    rows: List[List[str]] = []
    for section in SNAPSHOT_SECTIONS:
        rows.append([f"----- {section} -----", "", "", ""])
        va = snapshot_a.get(section) if isinstance(snapshot_a, dict) else None
        vb = snapshot_b.get(section) if isinstance(snapshot_b, dict) else None

        if isinstance(va, dict) or isinstance(vb, dict):
            da = va if isinstance(va, dict) else {}
            db = vb if isinstance(vb, dict) else {}
            for k in _ordered_union_keys(da, db):
                a_val = da.get(k)
                b_val = db.get(k)
                changed = "YES" if _canon(a_val) != _canon(b_val) else "NO"
                rows.append([f"{section}.{k}", _fmt_compact(a_val), _fmt_compact(b_val), changed])
        else:
            changed = "YES" if _canon(va) != _canon(vb) else "NO"
            rows.append([section, _fmt_compact(va), _fmt_compact(vb), changed])
    return rows


def _active_modes(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    modes = payload.get("modes", {}) if isinstance(payload, dict) else {}
    out = {}
    for key, entry in modes.items():
        if isinstance(entry, dict) and not entry.get("skipped", False):
            out[key] = entry
    return out


def _build_mode_table_rows(mode_a: Dict[str, Any], mode_b: Dict[str, Any]) -> List[List[str]]:
    rows: List[List[str]] = []
    for metric_key, metric_label in METRICS:
        a = mode_a.get(metric_key)
        b = mode_b.get(metric_key)
        d, p = _fmt_delta(a, b)
        rows.append([metric_label, _fmt_value(a), _fmt_value(b), d, p])
    return rows


def _render_ascii_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: List[str]) -> str:
        return "| " + " | ".join(cells[i].ljust(widths[i]) for i in range(len(cells))) + " |"

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"

    lines = [sep, fmt_row(headers), sep]
    for row in rows:
        lines.append(fmt_row(row))
    lines.append(sep)
    return "\n".join(lines)


def _render_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def compare_runs(path_a: str, path_b: str, label_a: str, label_b: str) -> Tuple[str, str]:
    payload_a = _load_json(path_a)
    payload_b = _load_json(path_b)

    modes_a = _active_modes(payload_a)
    modes_b = _active_modes(payload_b)

    # Show only modes active in both runs; skipped modes are omitted entirely.
    common_modes = [m for m in MODE_LABELS.keys() if m in modes_a and m in modes_b]

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    headers = ["Metric", label_a, label_b, "Delta", "Delta %"]

    txt_parts: List[str] = []
    md_parts: List[str] = []

    txt_parts.append("RUN COMPARISON REPORT")
    txt_parts.append(f"Generated: {now}")
    txt_parts.append(f"Run A: {path_a}")
    txt_parts.append(f"Run B: {path_b}")
    txt_parts.append("")

    md_parts.append("# Run Comparison Report")
    md_parts.append(f"Generated: {now}")
    md_parts.append("")
    md_parts.append(f"- Run A: {path_a}")
    md_parts.append(f"- Run B: {path_b}")
    md_parts.append("")

    if not common_modes:
        msg = "No common active modes found (all overlapping modes are skipped in at least one run)."
        txt_parts.append(msg)
        txt_parts.append("")
        md_parts.append(msg)
        md_parts.append("")
    else:
        for mode_key in common_modes:
            mode_name = MODE_LABELS.get(mode_key, mode_key)
            rows = _build_mode_table_rows(modes_a[mode_key], modes_b[mode_key])

            txt_parts.append(f"MODE: {mode_name}")
            txt_parts.append(_render_ascii_table(headers, rows))
            txt_parts.append("")

            md_parts.append(f"## {mode_name}")
            md_parts.append(_render_markdown_table(headers, rows))
            md_parts.append("")

    snapshot_a = _load_snapshot_for_output(path_a)
    snapshot_b = _load_snapshot_for_output(path_b)

    txt_parts.append("SNAPSHOT INPUT COMPARISON")
    md_parts.append("## Snapshot Input Comparison")

    if snapshot_a is None or snapshot_b is None:
        missing: List[str] = []
        if snapshot_a is None:
            missing.append(f"missing snapshot_input.json beside: {path_a}")
        if snapshot_b is None:
            missing.append(f"missing snapshot_input.json beside: {path_b}")
        msg = "; ".join(missing)
        txt_parts.append(msg)
        md_parts.append(msg)
        txt_parts.append("")
        md_parts.append("")
    else:
        cfg_headers = ["Config Field", label_a, label_b, "Changed"]
        cfg_rows = _build_snapshot_table_rows(snapshot_a, snapshot_b)
        txt_parts.append(_render_ascii_table(cfg_headers, cfg_rows))
        txt_parts.append("")
        md_parts.append(_render_markdown_table(cfg_headers, cfg_rows))
        md_parts.append("")

    return "\n".join(txt_parts).strip() + "\n", "\n".join(md_parts).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two experiment output.json runs")
    parser.add_argument("run_a", help="Path to first output.json")
    parser.add_argument("run_b", help="Path to second output.json")
    parser.add_argument("--label-a", default=None, help="Column label for run A")
    parser.add_argument("--label-b", default=None, help="Column label for run B")
    parser.add_argument(
        "--out-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory for generated reports",
    )
    parser.add_argument(
        "--name-prefix",
        default="run_compare",
        help="Output filename prefix",
    )
    parser.add_argument(
        "--no-txt",
        action="store_true",
        help="Disable writing the ASCII .txt report",
    )
    parser.add_argument(
        "--no-md",
        action="store_true",
        help="Disable writing the Markdown .md report",
    )
    args = parser.parse_args()

    run_a = os.path.abspath(args.run_a)
    run_b = os.path.abspath(args.run_b)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    label_a = args.label_a or _safe_label(run_a, "Run A")
    label_b = args.label_b or _safe_label(run_b, "Run B")

    txt, md = compare_runs(run_a, run_b, label_a, label_b)

    write_txt = not args.no_txt
    write_md = not args.no_md
    if not write_txt and not write_md:
        raise ValueError("Both outputs are disabled. Enable at least one of TXT or MD output.")

    ts = dt.datetime.now().strftime("%m%d-%H%M")
    base = f"{_sanitize_filename_part(args.name_prefix)}_{_sanitize_filename_part(label_a)}_vs_{_sanitize_filename_part(label_b)}_{ts}"
    txt_path = os.path.join(out_dir, base + ".txt")
    md_path = os.path.join(out_dir, base + ".md")

    if write_txt:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt)
        print(f"TXT report: {txt_path}")
    else:
        print("TXT report: disabled")

    if write_md:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"MD report : {md_path}")
    else:
        print("MD report : disabled")


if __name__ == "__main__":
    main()
