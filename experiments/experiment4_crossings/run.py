"""
experiments/experiment4_crossings/run.py
========================================
Thin launcher for Experiment 4 – Crossings (walk network + synthetic crossings).

Each unique input config produces a deterministic 8-char subfolder so runs
with different parameters are never overwritten. The comparison map and the
exact input used are saved together in the subfolder.

Usage
-----
    python experiments/experiment4_crossings/run.py
    python experiments/experiment4_crossings/run.py --input experiments/experiment4_crossings/input.json
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from io import StringIO

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR, os.pardir, os.pardir))
_COMPARISON_DIR = os.path.join(_ROOT, "experiments", "comparison")

for _p in (_ROOT, _COMPARISON_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _strip_comments(obj):
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items() if k != "_comment"}
    if isinstance(obj, list):
        return [_strip_comments(v) for v in obj]
    return obj


def _input_hash(input_path: str) -> str:
    """Generate hash from input config, excluding mrt_enabled from constraints."""
    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)
    
    # Remove mrt_enabled from constraints before hashing
    hash_data = _strip_comments(raw)
    if "constraints" in hash_data and "mrt_enabled" in hash_data["constraints"]:
        hash_data = json.loads(json.dumps(hash_data))  # deep copy
        del hash_data["constraints"]["mrt_enabled"]
    
    canonical = json.dumps(hash_data, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(canonical.encode()).hexdigest()[:8]


def _get_mrt_status(input_path: str) -> str:
    """Return 'mrt_enabled' or 'mrt_disabled' based on input config."""
    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)
    constraints_cfg = raw.get("constraints", {})
    enabled = constraints_cfg.get("mrt_enabled", False)
    return "mrt" if enabled else "dmrt"


def _write_no_sidepanels_html(source_html: str, target_html: str):
    """Create a no-sidepanels copy of a generated comparison map HTML."""
    from strip_comparison_side_panels import strip_side_panels

    with open(source_html, encoding="utf-8") as f:
        raw_html = f.read()
    cleaned_html, stats = strip_side_panels(raw_html)
    with open(target_html, "w", encoding="utf-8") as f:
        f.write(cleaned_html)
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 4 – Crossings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        default=os.path.join(_DIR, "input.json"),
        help="Path to an input.json config file.",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        sys.exit(f"[ERROR] config file not found: {input_path}")

    h = _input_hash(input_path)
    mrt_status = _get_mrt_status(input_path)
    timestamp = datetime.now().strftime("%m%d-%H%M")  # Removed YY
    base_dir = os.path.join(_DIR, f"{h}_{mrt_status}")
    run_dir = f"{base_dir}_1_{timestamp}"
    
    i = 1
    while os.path.exists(run_dir):
        i += 1
        run_dir = f"{base_dir}_{i}_{timestamp}"
    os.makedirs(run_dir)



    dest_input = os.path.join(run_dir, "snapshot_input.json")
    output_path = os.path.join(run_dir, "comparison_map.html")
    output_no_sidepanels_path = os.path.join(run_dir, "comparison_map_no_sidepanels.html")
    log_path = os.path.join(run_dir, "terminal_log.txt")

    if os.path.abspath(input_path) != os.path.abspath(dest_input):
        shutil.copy2(input_path, dest_input)

    # Capture terminal output
    log_buffer = StringIO()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    class TeeOutput:
        def __init__(self, *outputs):
            self.outputs = outputs
        def write(self, data):
            for output in self.outputs:
                output.write(data)
        def flush(self):
            for output in self.outputs:
                output.flush()
    
    sys.stdout = TeeOutput(original_stdout, log_buffer)
    sys.stderr = TeeOutput(original_stderr, log_buffer)

    try:
        print("Experiment 4 – Crossings")
        print(f"  Config hash  : {h}")
        print(f"  MRT Status   : {mrt_status.replace('_', ' ')}")
        print(f"  Run folder   : {run_dir}")
        print(f"  Input config : {input_path}")
        print(f"  Output map   : {output_path}")
        print()

        from run_comparison import run as _run_comparison
        _run_comparison(
            input_path=dest_input,
            output_path=output_path,
            run_modes=("B",),
            visible_modes=("B",),
        )

        if os.path.isfile(output_path):
            try:
                panel_stats = _write_no_sidepanels_html(output_path, output_no_sidepanels_path)
                print(f"  Output map (no side panels): {output_no_sidepanels_path}")
                print(
                    "  Panels removed: "
                    f"bottom_right={panel_stats.get('bottom_right_panel_removed', 0)}, "
                    f"toggles={panel_stats.get('toggle_blocks_removed', 0)}"
                )
            except Exception as exc:
                print(f"[WARN] Could not generate no-sidepanels HTML: {exc}")
        else:
            print(f"[WARN] comparison_map.html was not found at: {output_path}")
    finally:
        # Restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
        # Get complete log content
        log_content = log_buffer.getvalue()
        
        # Split debug logs to separate file
        debug_log_path = os.path.join(run_dir, "debug.txt")
        terminal_lines = []
        debug_lines = []
        
        for line in log_content.splitlines(keepends=True):
            if '[DEBUG]' in line:
                debug_lines.append(line)
            else:
                terminal_lines.append(line)
        
        # Write non-debug lines to terminal_log.txt
        with open(log_path, 'w', encoding='utf-8') as f:
            f.writelines(terminal_lines)
        
        # Write debug lines to debug.txt
        if debug_lines:
            with open(debug_log_path, 'w', encoding='utf-8') as f:
                f.writelines(debug_lines)
        
        print(f"\n[Log saved to: {log_path}]")
        if debug_lines:
            print(f"[Debug log saved to: {debug_log_path}] ({len(debug_lines)} lines)")


if __name__ == "__main__":
    main()
