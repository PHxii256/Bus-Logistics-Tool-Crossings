"""
compare_lewis.py — Compare our benchmark results against Lewis et al. (2022) Table 3.

Our total_time_min uses the SAME Bing Maps d-matrix travel times from .bus files,
so the driving-time component is directly comparable. The only difference is that
Lewis includes dwell time: 15s + 5s × students_boarding per stop per route.

We estimate dwell: 15s × stops_used + 5s × students_served (each student boards once).
"""

import json, os, sys

_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_DIR, 'benchmark_1_results.json')

with open(RESULTS) as f:
    results = json.load(f)

# Lewis Table 3: best total journey time (seconds) across all variants, and |R|
lewis = {
    'Porthcawl':    {'routes': 1,  'best_s': 1116,  'students': 66},
    'Cardiff':      {'routes': 2,  'best_s': 2389,  'students': 156},
    'Suffolk':      {'routes': 3,  'best_s': 6668,  'students': 209},
    'MiltonKeynes': {'routes': 4,  'best_s': 4820,  'students': 274},
    'Edinburgh-2':  {'routes': 4,  'best_s': 5428,  'students': 320},
    'Bridgend':     {'routes': 6,  'best_s': 8208,  'students': 381},
    'Canberra':     {'routes': 7,  'best_s': 8214,  'students': 499},
    'Edinburgh-1':  {'routes': 9,  'best_s': 13272, 'students': 680},
    'Adelaide':     {'routes': 8,  'best_s': 10295, 'students': 565},
    'Brisbane':     {'routes': 10, 'best_s': 15631, 'students': 757},
}

print(f"{'Instance':<14} {'S':>4} {'L|R|':>5} {'O|R|':>5} {'Match':>6}  "
      f"{'OurDrive(s)':>11} {'Est.Dwell':>9} {'OurTotal(s)':>11} {'Lewis(s)':>8} {'Gap%':>7}")
print("-" * 95)

route_matches = 0
total_instances = 0

for r in results:
    name = r['instance']
    if name not in lewis:
        continue
    lw = lewis[name]
    total_instances += 1

    our_routes   = r['active_routes']
    our_time_sec = r['total_time_min'] * 60.0  # Bing Maps driving time only
    stops_used   = r['stops_used']
    served       = r['students_served']

    # Estimate dwell: 15s fixed per stop + 5s per boarding student
    est_dwell_s = stops_used * 15 + served * 5
    our_total_s = our_time_sec + est_dwell_s

    lewis_best  = lw['best_s']
    gap_pct     = (our_total_s - lewis_best) / lewis_best * 100.0

    match = "YES" if our_routes == lw['routes'] else f"{our_routes}v{lw['routes']}"
    if our_routes == lw['routes']:
        route_matches += 1

    print(f"{name:<14} {lw['students']:>4} {lw['routes']:>5} {our_routes:>5} {match:>6}  "
          f"{our_time_sec:>11.0f} {est_dwell_s:>9} {our_total_s:>11.0f} {lewis_best:>8} {gap_pct:>+7.1f}%")

print("-" * 95)
print(f"Route count match: {route_matches}/{total_instances}")
print()
print("Notes:")
print("  - OurDrive(s): sum of route times from Bing Maps d-matrix (same source as Lewis)")
print("  - Est.Dwell:   15s × stops_used + 5s × students_served (Lewis uses d1=15, d2=5)")
print("  - Lewis(s):    Best reported total journey time from Sciortino et al. Table 3")
print("  - Gap%:        (OurTotal - Lewis) / Lewis × 100  (positive = ours worse)")
print()
print("Caveats:")
print("  - Our system uses homogeneous fleet (cap=70); Lewis uses heterogeneous fleet")
print("  - Dwell estimate assumes each student boards once; multistops add extra dwell")
print("  - Lewis ran 25×4 variants with 5-min time limit each; we ran ALNS 50-200 iters")
print("  - Our ALNS uses different operators (destroy/repair vs local search)")
