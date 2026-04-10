#!/usr/bin/env python3
"""Quick test to validate the new student metrics in output.json"""
import json
import sys

if len(sys.argv) < 2:
    print("Usage: python test_new_metrics.py <output_dir>")
    print("Example: python test_new_metrics.py experiments/experiment4_crossings/8027b921_dmrt_7.5km")
    sys.exit(1)

output_dir = sys.argv[1]

with open(f"{output_dir}/output.json") as f:
    output = json.load(f)

# Check door_to_door mode
dd = output['modes']['door_to_door']
students = dd['students']

print("=" * 80)
print("NEW METRICS VALIDATION")
print("=" * 80)
print(f"\nTotal students in door_to_door mode: {len(students)}")

# Check first few students
print("\nFirst 3 students:")
for s in students[:3]:
    print(f"\n{s['id']}:")
    print(f"  stage:              {s.get('stage')}")
    print(f"  route_id:           {s.get('route_id', 'MISSING')}")
    print(f"  pickup_order:       {s.get('pickup_order', 'MISSING')}")
    print(f"  ride_time_min:      {s.get('ride_time_min')}")
    print(f"  ride_distance_km:   {s.get('ride_distance_km', 'MISSING')}")
    print(f"  direct_potential_min: {s.get('direct_potential_min')}")
    print(f"  direct_distance_km:   {s.get('direct_distance_km', 'MISSING')}")
    print(f"  walk_distance_m:    {s.get('walk_distance_m')}")

# Validate all students have new fields
print("\n" + "=" * 80)
print("FIELD COVERAGE")
print("=" * 80)

fields_to_check = ['route_id', 'pickup_order', 'ride_distance_km', 'direct_distance_km']
for field in fields_to_check:
    has_field = sum(1 for s in students if field in s)
    not_null = sum(1 for s in students if s.get(field) is not None)
    print(f"{field:25s}: {has_field}/{len(students)} have field, {not_null}/{len(students)} non-null")

# Validate ride_distance >= direct_distance
print("\n" + "=" * 80)
print("DISTANCE VALIDATION (ride_distance should be >= direct_distance)")
print("=" * 80)

violations = []
for s in students:
    ride_dist = s.get('ride_distance_km')
    direct_dist = s.get('direct_distance_km')
    if ride_dist is not None and direct_dist is not None:
        if ride_dist < direct_dist:
            violations.append({
                'id': s['id'],
                'ride': ride_dist,
                'direct': direct_dist,
                'diff': ride_dist - direct_dist
            })

if violations:
    print(f"\n⚠️  WARNING: {len(violations)} students have ride_distance < direct_distance:")
    for v in violations[:5]:
        print(f"  {v['id']}: ride={v['ride']:.2f} < direct={v['direct']:.2f} (diff: {v['diff']:.2f})")
else:
    print("✓ All students have ride_distance >= direct_distance (as expected)")

# Check pickup_order sequence per route
print("\n" + "=" * 80)
print("PICKUP ORDER VALIDATION (should be 1, 2, 3... per route)")
print("=" * 80)

routes = {}
for s in students:
    route_id = s.get('route_id')
    pickup_order = s.get('pickup_order')
    if route_id and pickup_order:
        if route_id not in routes:
            routes[route_id] = []
        routes[route_id].append(pickup_order)

print(f"\nFound {len(routes)} routes")
for route_id, orders in sorted(routes.items())[:5]:
    orders.sort()
    expected = list(range(1, len(orders) + 1))
    match = "✓" if orders == expected else "✗"
    print(f"  {route_id}: {orders[:10]}{'...' if len(orders) > 10 else ''} {match}")

print("\n" + "=" * 80)
print("✓ Validation complete!")
print("=" * 80)
