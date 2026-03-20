"""
experiment_runner.py  In-process runner for Experiment 1 and Experiment 2.

Downloads the Cairo road network ONCE, then runs every seed/condition directly
in the same process (no subprocesses).  Module-level caches are cleared between
runs so different seeds do not interfere.

Experiment 1  (Safety vs Unconstrained):
  5 seeds x 40 students x 2 modes (constrained / unconstrained)
  -> experiment1_results.csv

Experiment 2  (Scalability):
  n_students in {20, 40, 60, 80, 100} x 1 seed (constrained)
  -> experiment2_results.csv
"""

import os, sys, json, time, copy, csv

# ── path fix: core modules live one level up ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, '..'))

import detour_engine as _eng
import alns_engine as _alns

from run_algorithm  import setup_graph, precompute_matrix
from data_loader    import load_json, load_mode1_input, print_input_summary, serialize_routes
from solution_state import ServiceSolution
from alns_engine    import ALNSEngine

# ── subfolders for experiment data ──
_EXP1_DIR = os.path.join(_SCRIPT_DIR, 'experiment1_safety_vs_unconstrained')
_EXP2_DIR = os.path.join(_SCRIPT_DIR, 'experiment2_scalability')

# Enable fast snap (nearest_node) for experiments on large city graphs.
# This avoids the ~4s ox.nearest_edges call per student on Cairo's 609K node graph.
# Comparisons between constrained/unconstrained remain valid.
_eng._FAST_SNAP_MODE = True


def _prebuild_ball_tree(G):
    """Pre-build our BallTree so the first real snap call is instant.
    OSMnx rebuilds BallTree + converts all nodes to GeoDataFrame on every call.
    We bypass that by building sklearn BallTree once here and caching it in
    detour_engine._BALL_TREE.
    """
    print("  Pre-building BallTree spatial index (one-time)...", end='', flush=True)
    t0 = time.time()
    _eng._get_or_build_ball_tree(G)  # populates module-level cache
    print(f" done in {time.time()-t0:.1f}s")


def _reset_run_caches(full=False):
    """Clear per-run caches between ALNS runs.
    
    full=False (default): only clear ALNS candidate cache (safe to reuse
      spatial/distance data when re-running the same student dataset).
    full=True: clear everything including spatial caches (use between
      different input files with different student locations).
    """
    _alns._student_candidate_cache.clear()
    if full:
        _eng._MATRIX_CACHE.clear()
        _eng._MATRIX_CACHE_LENGTH.clear()
        _eng._path_cache.clear()
        _eng._WALK_DIST_CACHE.clear()
        _eng._STUDENT_NODE_CACHE.clear()
        _eng._WALK_GRAPH = None


def _run_alns(data, G, label, iterations, precomputed=False):
    """Run one ALNS.  If precomputed=True, skip matrix precompute (reuse existing cache)."""
    _reset_run_caches(full=False)
    students, buses, routes, school_coords, constraints, algo_config = load_mode1_input(data, G)
    iters = iterations if iterations else algo_config.get('iterations', 60)
    if not precomputed:
        precompute_matrix(students, routes, G)
    initial_sol = ServiceSolution(students, routes, G)
    optimizer   = ALNSEngine(initial_sol, iterations=iters)
    t0       = time.time()
    best_sol = optimizer.run()
    elapsed  = time.time() - t0
    from detour_engine import calculate_route_time_from_matrix, calculate_route_distance_from_matrix
    for r in best_sol.routes:
        t = calculate_route_time_from_matrix(r.stops, G)
        r.total_time     = t if t is not None else 0.0
        d = calculate_route_distance_from_matrix(r.stops, G)
        r.total_distance = d if d is not None else 0.0
    served  = sum(1 for s in best_sol.students if s.is_served)
    total   = len(best_sol.students)
    active  = [r for r in best_sol.routes if r.get_student_count() > 0]
    total_time = sum(r.total_time     for r in active)
    total_dist = sum(r.total_distance for r in active)
    objective  = best_sol.calculate_objective()
    print(f"  [{label}] {served}/{total} served, time={total_time:.1f}min, dist={total_dist:.1f}km, obj={objective:.0f}, {elapsed:.1f}s")
    return {
        "label": label, "served": served, "total": total,
        "serve_pct": round(100*served/total, 2) if total else 0,
        "total_time_min": round(total_time, 2), "total_dist_km": round(total_dist, 2),
        "objective": round(objective, 2), "active_routes": len(active),
        "runtime_seconds": round(elapsed, 2), "iterations": iters,
    }


def _apply_unconstrained(data):
    d = copy.deepcopy(data)
    if 'data' in d and 'students' in d['data']:
        for s in d['data']['students']:
            s['walk_radius_override'] = 400
    d['input'].setdefault('constraints', {}).update({
        "ride_time_multiplier": 999, "floor_minutes": 999, "ceiling_minutes": 999})
    return d


def _write_csv(rows, path):
    if not rows:
        print(f"  No rows for {path}"); return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f"  Saved: {path}")


def experiment_1(G, iterations=60):
    print("\n" + "="*60 + "\nEXPERIMENT 1: Safety-Constrained vs Unconstrained\n" + "="*60)
    results = []
    for seed in [42, 43, 44, 45, 46]:
        fname = f"experiment1_40st_s{seed}.json"
        fpath = os.path.join(_EXP1_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  SKIP: {fpath}"); continue
        data = load_json(fpath)
        # Full cache reset between different input files (different student locations)
        _reset_run_caches(full=True)
        print(f"\nSeed {seed} - CONSTRAINED  (with matrix precompute)")
        res_c = _run_alns(data, G, f"s{seed}_C", iterations, precomputed=False)
        res_c.update({"seed": seed, "mode": "constrained", "input_file": fname})
        # Unconstrained reuses the same student locations -> skip recompute
        print(f"Seed {seed} - UNCONSTRAINED (reusing matrix cache)")
        res_u = _run_alns(_apply_unconstrained(data), G, f"s{seed}_U", iterations, precomputed=True)
        res_u.update({"seed": seed, "mode": "unconstrained", "input_file": fname})
        results.extend([res_c, res_u])
    _write_csv(results, os.path.join(_EXP1_DIR, "experiment1_results.csv"))
    if results:
        print("\n--- Experiment 1 Summary ---")
        for mode in ('constrained', 'unconstrained'):
            rows = [r for r in results if r['mode'] == mode]
            if not rows: continue
            print(f"  {mode:14s}  served={sum(r['serve_pct'] for r in rows)/len(rows):.1f}%  "
                  f"time={sum(r['total_time_min'] for r in rows)/len(rows):.1f}min  "
                  f"runtime={sum(r['runtime_seconds'] for r in rows)/len(rows):.1f}s")
    return results


def experiment_2(G, iterations=60):
    print("\n" + "="*60 + "\nEXPERIMENT 2: Scalability\n" + "="*60)
    n_files = {20: os.path.join(_EXP2_DIR, "scalability_20st.json"),
               40: os.path.join(_EXP1_DIR, "experiment1_40st_s42.json"),
               60: os.path.join(_EXP2_DIR, "scalability_60st.json"),
               80: os.path.join(_EXP2_DIR, "scalability_80st.json"),
               100: os.path.join(_EXP2_DIR, "scalability_100st.json")}
    results = []
    for n, fpath in sorted(n_files.items()):
        if not os.path.exists(fpath):
            print(f"  SKIP: {fpath}"); continue
        data = load_json(fpath)
        print(f"\nn={n} - {fpath}")
        res = _run_alns(data, G, f"n{n}", iterations)
        res.update({"n_students": n, "input_file": fpath})
        results.append(res)
    _write_csv(results, os.path.join(_EXP2_DIR, "experiment2_results.csv"))
    if results:
        print("\n--- Experiment 2 Summary ---")
        print(f"  {'n':>6}  {'served%':>8}  {'time min':>9}  {'runtime s':>10}")
        for r in results:
            print(f"  {r['n_students']:>6}  {r['serve_pct']:>8.1f}  {r['total_time_min']:>9.1f}  {r['runtime_seconds']:>10.1f}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Experiments 1 and 2")
    parser.add_argument('--iterations', type=int, default=60)
    parser.add_argument('--exp', choices=['1', '2', 'all'], default='all')
    parser.add_argument('--input', default=os.path.join(_EXP1_DIR, 'experiment1_40st_s42.json'))
    args = parser.parse_args()

    print(f"Loading graph from '{args.input}'...")
    input = load_json(args.input)['input']
    G = setup_graph(input, unconstrained=False)
    _prebuild_ball_tree(G)   # Build BallTree once; all snaps use cached tree (<1ms each)

    if args.exp in ('1', 'all'):
        experiment_1(G, iterations=args.iterations)
    if args.exp in ('2', 'all'):
        experiment_2(G, iterations=args.iterations)

    print("\nAll experiments complete.")
