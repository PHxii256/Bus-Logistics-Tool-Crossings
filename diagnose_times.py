#!/usr/bin/env python3
"""
Quick diagnostic: trace S072's time calculations from the experiment output
"""
import json
import sys
from math import radians, cos, sin, asin, sqrt

def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in kilometers"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371.0 * c

def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_times.py <output_dir>")
        print("Example: python diagnose_times.py experiments/experiment4_crossings/8027b921_dmrt_7.5km")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    
    # Load files
    with open(f"{output_dir}/output.json") as f:
        output = json.load(f)
    with open(f"{output_dir}/snapshot_input.json") as f:
        snapshot = json.load(f)
    
    # Get school location
    school_lat = snapshot['school']['latitude']
    school_lon = snapshot['school']['longitude']
    
    # Get door_to_door mode
    dd = output['modes']['door_to_door']
    students = dd['students']
    
    print("=" * 80)
    print("RIDE TIME DIAGNOSTIC")
    print("=" * 80)
    print(f"\nSchool: ({school_lat:.6f}, {school_lon:.6f})")
    print(f"\nTotal students: {len(students)}")
    
    # Show examples with various metrics
    examples = [
        ("S072", "HIGH"),  # User's example
        ("S104", "KG"),    # From user's earlier message (was null)
    ]
    
    print("\n" + "=" * 80)
    print("EXAMPLE STUDENTS")
    print("=" * 80)
    
    for student_id, stage in examples:
        matches = [s for s in students if s['id'] == student_id]
        if not matches:
            print(f"\n{student_id} not found in door_to_door mode")
            continue
            
        s = matches[0]
        ride_time = s['ride_time_min']
        direct_time = s['direct_potential_min']
        walk_dist = s['walk_distance_m']
        
        print(f"\n{student_id} ({stage}):")
        print(f"  ride_time_min:        {ride_time}")
        print(f"  direct_potential_min: {direct_time}")
        print(f"  walk_distance_m:      {walk_dist}")
        
        if direct_time and direct_time > 0:
            # If we assume straight-line distance would take direct_time at some speed,
            # we can estimate the distance and compare to reality
            # But we don't have student home coords here...
            print(f"\n  IMPLIED METRICS (assuming typical Cairo driving):")
            for speed_kph in [20, 30, 40, 50]:
                dist_km = (direct_time / 60.0) * speed_kph
                print(f"    At {speed_kph} km/h: {dist_km:.2f} km distance")
    
    # Show distribution stats
    print("\n" + "=" * 80)
    print("OVERALL DISTRIBUTION")
    print("=" * 80)
    
    ride_times = [s['ride_time_min'] for s in students if s['ride_time_min'] is not None]
    direct_times = [s['direct_potential_min'] for s in students if s['direct_potential_min'] is not None]
    
    if ride_times:
        print(f"\nRide Times (n={len(ride_times)}):")
        print(f"  Min:    {min(ride_times):.2f} min")
        print(f"  Max:    {max(ride_times):.2f} min")
        print(f"  Mean:   {sum(ride_times)/len(ride_times):.2f} min")
        print(f"  Median: {sorted(ride_times)[len(ride_times)//2]:.2f} min")
    
    if direct_times:
        print(f"\nDirect Potential Times (n={len(direct_times)}):")
        print(f"  Min:    {min(direct_times):.2f} min")
        print(f"  Max:    {max(direct_times):.2f} min")
        print(f"  Mean:   {sum(direct_times)/len(direct_times):.2f} min")
        print(f"  Median: {sorted(direct_times)[len(direct_times)//2]:.2f} min")
    
    # Check nulls
    null_rides = len([s for s in students if s['ride_time_min'] is None])
    null_directs = len([s for s in students if s['direct_potential_min'] is None])
    
    print(f"\nNull Values:")
    print(f"  ride_time_min null:        {null_rides}/{len(students)} ({100*null_rides/len(students):.1f}%)")
    print(f"  direct_potential_min null: {null_directs}/{len(students)} ({100*null_directs/len(students):.1f}%)")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION NOTES")
    print("=" * 80)
    print("""
ride_time_min:
  - Time student spends on bus from their pickup stop to school
  - Includes all intermediate stops and detours
  - Should be >= direct_potential_min

direct_potential_min:
  - Direct driving time from student's HOME to school
  - Represents optimal time if bus went straight there
  - Calculated from OSRM distance matrix (real road network)
  
If times seem too small:
  1. Check OSRM speed configuration (road_speeds_config.json)
  2. Verify OSRM is using correct road network (Egypt OSM data)
  3. Check if units are correct (should be minutes, not seconds)
  4. Verify student placement (are they really far from school?)
    """)

if __name__ == "__main__":
    main()
