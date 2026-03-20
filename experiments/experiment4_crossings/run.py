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
    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)
    canonical = json.dumps(
        _strip_comments(raw), sort_keys=True, separators=(",", ":")
    )
    return hashlib.md5(canonical.encode()).hexdigest()[:8]


def _get_crossings_status(input_path: str) -> str:
    """Return 'crossings_enabled' or 'crossings_disabled' based on input config."""
    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)
    synth_cfg = raw.get("synthetic_crossings", {})
    enabled = synth_cfg.get("enabled", False)
    return "crossings_enabled" if enabled else "crossings_disabled"


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
    crossings_status = _get_crossings_status(input_path)
    run_dir = os.path.join(_DIR, f"{h}_{crossings_status}")
    os.makedirs(run_dir, exist_ok=True)

    dest_input = os.path.join(run_dir, "input.json")
    output_path = os.path.join(run_dir, "comparison_map.html")

    if os.path.abspath(input_path) != os.path.abspath(dest_input):
        shutil.copy2(input_path, dest_input)

    print("Experiment 4 – Crossings")
    print(f"  Config hash  : {h}")
    print(f"  Crossings    : {crossings_status.replace('_', ' ')}")
    print(f"  Run folder   : {run_dir}")
    print(f"  Input config : {input_path}")
    print(f"  Output map   : {output_path}")
    print()

    from run_comparison import run as _run_comparison
    _run_comparison(
        input_path=dest_input,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
