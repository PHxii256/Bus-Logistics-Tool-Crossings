"""
run_benchmark.py — Graph-free ALNS benchmark runner for Lewis .bus instances.

Runs the ALNS on all (or selected) .bus files in the busprobs/ directory
using the bus d-matrix directly — no OSMnx graph download required.

Results are saved to benchmark_results.json and printed as a summary table.

Usage
─────
    python run_benchmark.py                        # all 10 instances
    python run_benchmark.py Porthcawl Bridgend      # specific instances
    python run_benchmark.py --iterations 300        # override iteration count
    python run_benchmark.py --unconstrained         # disable ride-time caps

Objective (same as normal ALNS mode):
    (students_served × 10000) - sum(route_times_minutes)

The maximisation objective means 100 % coverage is always preferred; among
fully-covering solutions the one with shorter total route time wins.
"""

import os
import sys
import json
import time
import math
import argparse

# ── path fix: core modules live one level up ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, '..'))
sys.path.insert(0, _SCRIPT_DIR)  # for bus_loader

# ---------------------------------------------------------------------------
# Import ALNS stack
# ---------------------------------------------------------------------------
from bus_loader import load_bus_file
from solution_state import ServiceSolution
from alns_engine import ALNSEngine, _student_candidate_cache
import detour_engine as _eng

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
BUS_PROBS_DIR   = os.path.join(os.path.dirname(__file__), 'busprobs')
RESULTS_FILE    = os.path.join(_SCRIPT_DIR, 'results', 'benchmark_results.json')

# Bus capacity used in the Lewis paper (UK instances)
DEFAULT_BUS_CAPACITY = 70

# Ride-time constraints (matching default system settings)
_CONSTRAINED   = dict(ride_time_multiplier=2.5, floor_minutes=45, ceiling_minutes=60)
_UNCONSTRAINED = dict(ride_time_multiplier=999, floor_minutes=9999, ceiling_minutes=9999)


# ---------------------------------------------------------------------------
# Per-instance run helper
# ---------------------------------------------------------------------------

def _run_instance(bus_file: str,
                  iterations: int,
                  constrained: bool,
                  bus_capacity: int) -> dict:
    """Load one .bus file, run ALNS (graph-free), return result dict."""

    constraints = _CONSTRAINED if constrained else _UNCONSTRAINED

    # 1. Parse .bus file and pre-populate engine caches
    inst = load_bus_file(
        bus_file,
        bus_capacity     = bus_capacity,
        alns_constraints = constraints,
    )

    # 2. Clear per-iteration caches that must not bleed across instances
    _student_candidate_cache.clear()
    _eng._WALK_GRAPH        = None
    _eng._WALK_DIST_CACHE.clear()
    _eng._STUDENT_NODE_CACHE.clear()
    _eng._path_cache.clear()

    print(f"\n{'='*70}")
    print(f"  Instance : {inst.instance_name}")
    print(f"  Students : {inst.n_total_pupils}")
    print(f"  Stops    : {inst.n_stops - 1}  (excl. school)")
    print(f"  Buses    : {len(inst.buses)} × cap {bus_capacity}")
    print(f"  Mode     : {'CONSTRAINED' if constrained else 'UNCONSTRAINED'}")
    print(f"  Iters    : {iterations}")
    print(f"{'='*70}")

    # 3. Build initial ServiceSolution (graph=None for benchmark mode)
    initial_sol = ServiceSolution(inst.students, inst.routes, graph=None)

    # 4. Run ALNS
    t0 = time.time()
    optimizer = ALNSEngine(initial_sol, iterations=iterations)
    best_sol  = optimizer.run()
    elapsed   = time.time() - t0

    # 5. Collect metrics
    served      = sum(1 for s in best_sol.students if s.is_served)
    total       = len(best_sol.students)
    served_pct  = 100.0 * served / total if total > 0 else 0.0
    active_routes = [r for r in best_sol.routes if r.get_student_count() > 0]
    total_dist_km = sum(
        sum(
            _eng._MATRIX_CACHE_LENGTH.get((r.stops[i].node_id, r.stops[i+1].node_id), 0) / 1000.0
            for i in range(len(r.stops) - 1)
        )
        for r in active_routes
    )
    total_time_min = sum(r.total_time for r in active_routes)
    objective      = best_sol.calculate_objective()
    stops_used = len({
        stop.node_id
        for route in active_routes
        for stop in route.stops
        if stop.stop_type != 'school'
    })

    print(f"\n  OK {inst.instance_name}: {served}/{total} served "
          f"({served_pct:.1f}%), {len(active_routes)} routes, "
          f"{total_dist_km:.1f} km, obj={objective:.0f}, {elapsed:.1f}s")

    return {
        "instance":            inst.instance_name,
        "constrained":         constrained,
        "n_pupils":            total,
        "n_stops_candidate":   inst.n_stops - 1,
        "students_served":     served,
        "students_unserved":   total - served,
        "serve_pct":           round(served_pct, 2),
        "active_routes":       len(active_routes),
        "stops_used":          stops_used,
        "total_dist_km":       round(total_dist_km, 2),
        "total_time_min":      round(total_time_min, 2),
        "objective":           round(objective, 2),
        "runtime_seconds":     round(elapsed, 2),
        "alns_iterations":     iterations,
        "bus_capacity":        bus_capacity,
    }


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

def _print_table(results: list[dict]):
    hdr = (f"{'Instance':<18} {'N':<6} {'Served':<8} {'%':<7} "
           f"{'Routes':<8} {'Dist km':<10} {'Obj':>12} {'Time s':>8}")
    sep = "-" * len(hdr)
    print(f"\n{sep}\nBENCHMARK SUMMARY\n{sep}")
    print(hdr)
    print(sep)
    for r in results:
        print(f"{r['instance']:<18} {r['n_pupils']:<6} "
              f"{r['students_served']:<8} {r['serve_pct']:<7.1f} "
              f"{r['active_routes']:<8} {r['total_dist_km']:<10.1f} "
              f"{r['objective']:>12.0f} {r['runtime_seconds']:>8.1f}")
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ALNS benchmark runner for .bus instances")
    parser.add_argument('instances', nargs='*',
                        help="Instance name(s) without extension (default: all)")
    parser.add_argument('--iterations', type=int, default=200,
                        help="ALNS iterations per instance (default: 200)")
    parser.add_argument('--capacity', type=int, default=DEFAULT_BUS_CAPACITY,
                        help=f"Bus seat capacity (default: {DEFAULT_BUS_CAPACITY})")
    parser.add_argument('--unconstrained', action='store_true',
                        help="Disable ride-time safety caps (unconstrained mode)")
    parser.add_argument('--output', default=RESULTS_FILE,
                        help=f"Output JSON path (default: {RESULTS_FILE})")
    args = parser.parse_args()

    # Discover .bus files
    all_files = sorted(
        f for f in os.listdir(BUS_PROBS_DIR) if f.endswith('.bus')
    )
    if args.instances:
        selected = {n.lower() for n in args.instances}
        all_files = [f for f in all_files
                     if os.path.splitext(f)[0].lower() in selected]
        if not all_files:
            print(f"ERROR: No .bus files matched: {args.instances}")
            sys.exit(1)

    print(f"Running {len(all_files)} instance(s) with {args.iterations} iterations each.")
    print(f"Mode: {'UNCONSTRAINED' if args.unconstrained else 'CONSTRAINED'}")
    print(f"Bus capacity: {args.capacity}")

    results = []
    failed  = []

    for fname in all_files:
        full_path = os.path.join(BUS_PROBS_DIR, fname)
        try:
            result = _run_instance(
                bus_file    = full_path,
                iterations  = args.iterations,
                constrained = not args.unconstrained,
                bus_capacity= args.capacity,
            )
            results.append(result)
            
            # Save progress after each instance
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
                
        except Exception as exc:
            import traceback
            print(f"\n  ERROR on {fname}: {exc}")
            traceback.print_exc()
            failed.append(fname)

    # Print table
    _print_table(results)

    if failed:
        print(f"\nFailed instances: {failed}")

    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to '{args.output}'")


if __name__ == '__main__':
    main()
