### Summary of changes

- **`api_new/schemas`**: Added three example contracts for the new flow: `school_config.json`, `change_location_request.json`, and `students_data.json` (minimal school admin fields + full route/stop/student shape).

- **`api_new/scripts`**: Added refactored helpers:
  - `data_extractors.py` — load a run folder (`snapshot_input.json`, optional `base_routes.json` / `output.json`), build/validate `school_config` and normalized `students_data`, merge API overrides.
  - `map_tools.py` — “before” map (star pin into `comparison_map.html` when present, else a simple multi-route map) and “after” map (previous vs updated route).
  - `change_location_pipeline.py` — fast insertion-based change-location, response JSON, and **strict invariants** so only the requested student can move; failures if non-requesting students would change.

- **`api_new/change_location_run.py`**: Runnable entry with top constants, pasteable command comment, `sys.path` fix for imports, override precedence (`api_new/inputs/*.json` over run-derived data), and writes to `api_new/outputs/` (`response.json`, `map_before.html`, `map_after.html`).

- **Behavior / bug focus**: Success responses now include both `updated_route` (compat) and **`updated_routes`** (full fleet) to avoid accidentally dropping other routes; before/after mapping uses a **true pre-removal** baseline for the affected route where relevant; invariant checks guard against “students disappeared” from bad data or bad merges.

- **`api_new/inputs`**: Seeded `change_location_request.json` (and during development, example `students_data` / `school_config` overrides for smoke runs when experiment `output.json` lacks stop-level detail).