"""
experiments/experiment3_safety_vs_efficency/run.py
===================================================
Thin launcher for Experiment 3 – Safety vs Efficiency.

Each unique input config produces a deterministic 8-char subfolder so runs
with different parameters are never overwritten.  The comparison map *and*
the exact input used are saved together in the subfolder.

Usage
-----
    # Run with the default input.json in this folder
    python experiments/experiment3_safety_vs_efficency/run.py

    # Run with a custom input
    python experiments/experiment3_safety_vs_efficency/run.py --input path/to/input.json
    
    or just run:
    python experiments/experiment3_safety_vs_efficency/run.py --input experiments\experiment3_safety_vs_efficency\input.json
    
    Linux: 
    python experiments/experiment3_safety_vs_efficency/run.py --input experiments/experiment3_safety_vs_efficency/input.json


Output structure
----------------
    experiments/experiment3_safety_vs_efficency/
        input.json              ← default / template config
        run.py                  ← this file
        a1b2c3d4/               ← hash of the config used
            input.json          ← copy of the config for reproducibility
            comparison_map.html ← interactive Folium map
            output.json         ← metrics for this run
        e5f6a7b8/               ← another run with different settings
            input.json
            comparison_map.html
            output.json
"""

import argparse
import hashlib
import json
import os
import shutil
import sys

# ── ensure repo root is importable ────────────────────────────────────────────
_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR, os.pardir, os.pardir))
_COMPARISON_DIR = os.path.join(_ROOT, "experiments", "comparison")

for _p in (_ROOT, _COMPARISON_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── helpers ───────────────────────────────────────────────────────────────────

def _strip_comments(obj):
    """Recursively remove ``_comment`` keys so comment-only edits
    don't produce a new experiment hash."""
    if isinstance(obj, dict):
        return {k: _strip_comments(v) for k, v in obj.items()
                if k != "_comment"}
    if isinstance(obj, list):
        return [_strip_comments(v) for v in obj]
    return obj


def _input_hash(input_path: str) -> str:
    """Return an 8-char hex digest of the canonical (comment-stripped) input."""
    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)
    canonical = json.dumps(
        _strip_comments(raw), sort_keys=True, separators=(",", ":")
    )
    return hashlib.md5(canonical.encode()).hexdigest()[:8]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3 – Safety vs Efficiency comparison runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        default=os.path.join(_DIR, "input.json"),
        help=(
            "Path to an input.json config file.  "
            "Defaults to input.json in this folder."
        ),
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        sys.exit(f"[ERROR] config file not found: {input_path}")

    # ── compute hash → create run subfolder ───────────────────────────────────
    h       = _input_hash(input_path)
    run_dir = os.path.join(_DIR, h)
    os.makedirs(run_dir, exist_ok=True)

    dest_input   = os.path.join(run_dir, "input.json")
    output_path = os.path.join(run_dir, "comparison_map.html")

    # copy input into the run folder (idempotent; skip if already the same file)
    if os.path.abspath(input_path) != os.path.abspath(dest_input):
        shutil.copy2(input_path, dest_input)

    print("Experiment 3 – Safety vs Efficiency")
    print(f"  Config hash  : {h}")
    print(f"  Run folder   : {run_dir}")
    print(f"  Input config : {input_path}")
    print(f"  Output map   : {output_path}")
    print()

    # ── delegate to the comparison engine ─────────────────────────────────────
    from run_comparison import run as _run_comparison
    _run_comparison(
        input_path=dest_input,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
