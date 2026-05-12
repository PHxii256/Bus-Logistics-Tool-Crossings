"""
Generate route_response.json from output.json

Route response schema:
- routes[]: each route has route_id and students list (with routed student fields)
- unserved_students[]: students not served (id, stage, physically_mentally_disabled, home_latitude, home_longitude)
"""

import json
from pathlib import Path


def generate_route_response(output_json_path: str, output_response_path: str = None):
    """
    Generate route_response.json from an output.json file.
    
    Args:
        output_json_path: path to output.json (from latest run or specific run)
        output_response_path: where to write route_response.json. If None, writes to api_new/outputs/route_response.json
    
    Returns:
        dict: the generated route_response
    """
    output_path = Path(output_json_path)
    if not output_path.exists():
        raise FileNotFoundError(f"output.json not found: {output_json_path}")
    
    with output_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract mode data
    mode = data.get('modes', {}).get('weakly_constrained', {})
    routes = mode.get('routes', [])
    students = mode.get('students', [])
    unserved = mode.get('unserved_students', [])
    
    # Fields to keep for routed students
    routed_keep_keys = [
        'id', 'stage', 'physically_mentally_disabled', 'route_id',
        'stop_node_id', 'stop_latitude', 'stop_longitude',
        'home_latitude', 'home_longitude', 'pickup_order',
        'ride_time_min', 'ride_distance_km',
        'direct_potential_min', 'direct_distance_km'
    ]
    
    # Fields to keep for unserved students
    unserved_keep_keys = [
        'id', 'stage', 'physically_mentally_disabled',
        'home_latitude', 'home_longitude'
    ]
    
    # Group routed students by route_id
    by_route = {}
    for s in students:
        rid = s.get('route_id')
        if not rid:
            continue
        filtered = {k: s.get(k) for k in routed_keep_keys}
        by_route.setdefault(rid, []).append(filtered)
    
    # Keep route order from routes list, then append any extra route_ids
    ordered_ids = [r.get('route_id') for r in routes if r.get('route_id')]
    for rid in sorted(by_route):
        if rid not in ordered_ids:
            ordered_ids.append(rid)
    
    # Build response
    response = {
        'routes': [],
        'unserved_students': []
    }
    
    # Add routed students per route
    for rid in ordered_ids:
        studs = by_route.get(rid, [])
        studs.sort(key=lambda x: (x.get('pickup_order') is None, x.get('pickup_order'), x.get('id')))
        response['routes'].append({
            'route_id': rid,
            'students': studs
        })
    
    # Add unserved students
    for u in unserved:
        filtered = {k: u.get(k) for k in unserved_keep_keys}
        response['unserved_students'].append(filtered)
    
    # Determine output path
    if output_response_path is None:
        output_response_path = Path(__file__).parent.parent / 'outputs' / 'route_response.json'
    else:
        output_response_path = Path(output_response_path)
    
    # Write response
    output_response_path.parent.mkdir(parents=True, exist_ok=True)
    with output_response_path.open('w', encoding='utf-8') as f:
        json.dump(response, f, ensure_ascii=False, indent=2)
    
    return response


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python generate_route_response.py <output.json> [output_response.json]")
        sys.exit(1)
    
    out_path = sys.argv[1]
    resp_path = sys.argv[2] if len(sys.argv) > 2 else None
    result = generate_route_response(out_path, resp_path)
    print(f"Generated route_response.json")
    print(f"  Routes: {len(result['routes'])}")
    print(f"  Routed students: {sum(len(r['students']) for r in result['routes'])}")
    print(f"  Unserved students: {len(result['unserved_students'])}")
