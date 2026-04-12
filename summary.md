# Change Location Re-Support Summary

## Overview

This update adds a standalone pipeline to support student location-change requests without re-running the full optimization.

The new flow is:

1. Extract a compact operational file from algorithm output (`base_routes.json`).
2. Process one change-location request (`change_location_request.json`) against that base.
3. Return a response (`response.json`) with success/failure and debug stats.
4. Generate maps for operations visibility:
   - `updated_route.html` (before vs after route, with star marker)
   - `map_insertion_attempt.html` (star pin injected into existing comparison map)

The behavior is insertion-only:

- No other students are moved between routes.
- Routes with no remaining capacity are skipped.
- If `route_id` is provided in request, it is treated as a soft hint (tried first, then fallback to others).

Default max detour is 5 minutes if not provided.

---

## What Was Changed

### New scripts

- `extract_base_routes.py`
  - Converts full algorithm output into operational `base_routes.json`.
  - Supports:
    - Unified route output (`routes[].path[]` present)
    - Experiment output under `modes.<mode>` (for example `weakly_constrained`) when stop coordinates are available per student.

- `process_change_location_request.py`
  - Reads `base_routes.json` and a request JSON.
  - Finds the student, removes old assignment, evaluates insertion positions, enforces detour limit, and writes:
    - `response.json`
    - `updated_route.html` on success

- `inject_star_pin_into_comparison_map.py`
  - Adds a star marker to an existing Folium `comparison_map.html`.
  - Writes a new output file (`map_insertion_attempt.html`) without modifying the source map.

### Existing code updated

- `data_loader.py`
  - `serialize_routes(...)` now includes operational fields needed by the new pipeline:
    - Per-route: `students_count`, `pickup_stops_count`, `capacity_total`, `capacity_used`, `capacity_remaining`, `occupancy_pct`
    - Per-stop: `students_count`

- `experiments/comparison/run_comparison.py`
  - Mode student records now include:
    - `stop_node_id`, `stop_latitude`, `stop_longitude`, `home_latitude`, `home_longitude`
  - Mode route records now include:
    - `bus_id`, `pickup_stops_count`, `capacity_total`, `capacity_used`, `capacity_remaining`, `occupancy_pct`

### Schema and request examples

- Updated:
  - `schemas/change_location_output_success.json`
  - `schemas/change_location_output_failed.json`
  - `schemas/routes_schema.json`
- Added:
  - `schemas/base_routes_schema.json`
  - `schemas/change_location_request_schema.json`
  - `api_requests/change_location_request.json`

---

## Pipeline (Simple)

### Step 1: Build base routes file

Input: algorithm `output.json`
Output: `base_routes.json`

Purpose:

- Keep only route and stop information needed for operations and change requests.

### Step 2: Process change request

Input:

- `base_routes.json`
- `change_location_request.json`

Output:

- `response.json`
- `updated_route.html` (success case)

Success response includes:

- `success: true`
- `assigned_route_id`
- `detour_minutes_added`
- `updated_route`
- `debug_stats` (runtime, routes considered, positions checked)

Failure response includes:

- `success: false`
- `reason`
- `debug_stats`

### Step 3: Map insertion attempt

Input:

- Existing `comparison_map.html`
- Requested new location coordinates

Output:

- `map_insertion_attempt.html`

---

## How To Run

Use the venv Python executable:

`c:/Users/phx25/Desktop/Grad Project/Bus-Logistics-Tool-Crossings/.venv/Scripts/python.exe`

### 1) Extract base routes

```powershell
& "c:/Users/phx25/Desktop/Grad Project/Bus-Logistics-Tool-Crossings/.venv/Scripts/python.exe" extract_base_routes.py --input <path-to-output.json> --output <path-to-base_routes.json>
```

Optional for experiment outputs:

```powershell
& "c:/Users/phx25/Desktop/Grad Project/Bus-Logistics-Tool-Crossings/.venv/Scripts/python.exe" extract_base_routes.py --input <path-to-output.json> --output <path-to-base_routes.json> --mode weakly_constrained
```

### 2) Process one change-location request

```powershell
& "c:/Users/phx25/Desktop/Grad Project/Bus-Logistics-Tool-Crossings/.venv/Scripts/python.exe" process_change_location_request.py --base-routes <path-to-base_routes.json> --request api_requests/change_location_request.json --response api_requests/response.json --updated-route-html api_requests/updated_route.html
```

### 3) Inject star pin into comparison map

```powershell
& "c:/Users/phx25/Desktop/Grad Project/Bus-Logistics-Tool-Crossings/.venv/Scripts/python.exe" inject_star_pin_into_comparison_map.py --comparison-map-html <path-to-comparison_map.html> --latitude 29.9615107 --longitude 31.2762834 --output api_requests/map_insertion_attempt.html
```

---

## Request Format (Current)

Example (`api_requests/change_location_request.json`):

```json
{
  "student_id": "S1",
  "route_id": "R1",
  "new_location": {
    "latitude": 29.9615107,
    "longitude": 31.2762834
  },
  "max_detour_minutes": 5,
  "change_type": "temporary"
}
```

Notes:

- `route_id` is optional and treated as a soft hint.
- `max_detour_minutes` is optional (defaults to 5).

---

## Important Notes

- This is a fast operational pipeline, not a full ALNS rerun.
- Detour checking in the standalone processor currently uses geometric segment deltas with route-average speed for quick evaluation.
- No cross-route reassignment of other students is performed.
- Existing comparison outputs that do not contain per-student stop coordinates cannot be extracted in experiment mode unless re-generated with the updated exporter.

---

## Quick File List

- `extract_base_routes.py`
- `process_change_location_request.py`
- `inject_star_pin_into_comparison_map.py`
- `schemas/base_routes_schema.json`
- `schemas/change_location_request_schema.json`
- `schemas/change_location_output_success.json`
- `schemas/change_location_output_failed.json`
- `api_requests/change_location_request.json`
