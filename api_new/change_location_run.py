import sys
from pathlib import Path
import glob
import json
import argparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api_new.scripts.change_location_pipeline import process_change_location
from api_new.scripts.data_extractors import (
    build_operational_students_data,
    build_school_students_from_routes_payload,
    build_school_config,
    load_json,
    normalize_students_data_payload,
    resolve_inputs_from_run,
    validate_school_config,
    validate_students_data,
    write_json,
)

# Paste-and-run command:
# & "c:/Users/phx25/Desktop/Grad Project/Bus-Logistics-Tool-Crossings/.venv/Scripts/python.exe" "api_new/change_location_run.py"

# -------------------------
# Inputs / constants
# -------------------------

def _discover_latest_run_folder(experiments_dir_rel_path="experiments/experiment4_crossings"):
    """
    Auto-discover the latest experiment run folder by:
    1. Globbing all run folders (matching pattern with timestamps)
    2. Sorting by timestamp from snapshot_input.json or folder modification time
    3. Returns the latest run folder path (relative to REPO_ROOT)
    
    Falls back to hardcoded default if no runs found.
    """
    repo_root = REPO_ROOT
    experiments_dir = repo_root / experiments_dir_rel_path
    
    if not experiments_dir.exists():
        print(f"Warning: experiments directory not found at {experiments_dir}")
        return None
    
    # Find all run folders (pattern: folder with snapshot_input.json)
    run_folders = []
    for run_dir in experiments_dir.iterdir():
        if run_dir.is_dir() and (run_dir / "snapshot_input.json").exists():
            run_folders.append(run_dir)
    
    if not run_folders:
        print(f"Warning: no run folders found in {experiments_dir}")
        return None
    
    # Sort by modification time (latest first)
    latest_run = max(run_folders, key=lambda d: d.stat().st_mtime)
    rel_path = latest_run.relative_to(repo_root)
    
    print(f"Auto-discovered latest run: {rel_path}")
    return str(rel_path)


# Default run folder (used only when explicitly needed in main())
RUN_FOLDER_REL_PATH_DEFAULT = "experiments/experiment4_crossings/627daf88_dmrt_1_0505-2126"
RUN_MODE_FOR_EXTRACTION = "weakly_constrained"

# Optional override files. If present, these override run-derived defaults.
SCHOOL_CONFIG_OVERRIDE_REL_PATH = "api_new/inputs/school_config.json"
STUDENTS_DATA_OVERRIDE_REL_PATH = "api_new/inputs/students_data.json"
CHANGE_LOCATION_REQUEST_REL_PATH = "api_new/inputs/change_location_request.json"

# Output artifacts
OUTPUT_RESPONSE_REL_PATH = "api_new/outputs/response.json"
OUTPUT_BEFORE_MAP_REL_PATH = "api_new/outputs/map_before.html"
OUTPUT_AFTER_MAP_REL_PATH = "api_new/outputs/map_after.html"


def _validate_change_location_request(payload):
    if "student_id" not in payload:
        raise ValueError("change_location_request missing required field: student_id")
    if "new_location" not in payload or not isinstance(payload["new_location"], dict):
        raise ValueError("change_location_request missing required field: new_location")
    if "latitude" not in payload["new_location"] or "longitude" not in payload["new_location"]:
        raise ValueError("change_location_request.new_location must include latitude and longitude")


def _ensure_seed_request_file(path):
    if path.exists():
        return
    seed_payload = load_json(path.parents[1] / "schemas" / "change_location_request.json")
    write_json(path, seed_payload)
    raise FileNotFoundError(
        f"Seeded missing request file at {path}. Edit it, then rerun."
    )


def main():
    repo_root = REPO_ROOT
    parser = argparse.ArgumentParser(description="Run change-location using api_new inputs or a run folder.")
    parser.add_argument("--use-api-inputs", action="store_true", help="Prefer api_new/inputs files instead of auto-discovering runs")
    parser.add_argument("--no-auto-discover", action="store_true", help="Disable auto-discovery of latest run folder")
    parser.add_argument("--auto-discover", action="store_true", help="Enable auto-discovery of latest run folder when not using api_new inputs")
    parser.add_argument("--run-dir", type=str, help="Explicit run folder relative to repo root")
    parser.add_argument("--skip-map", action="store_true", help="Skip writing map files (fast)")
    args = parser.parse_args()

    # Determine run directory behavior
    run_dir = None
    if args.run_dir:
        run_dir = repo_root / args.run_dir
    else:
        # Prefer api_new inputs when they exist (no auto-discovery needed)
        response_routes_present = False
        response_path = repo_root / "api_new" / "outputs" / "response.json"
        if response_path.exists():
            try:
                response_payload = load_json(response_path)
                response_routes_present = bool(response_payload.get("updated_routes"))
            except Exception:
                response_routes_present = False
        api_inputs_present = (
            (repo_root / "api_new" / "inputs" / "students_data.json").exists()
            and (repo_root / "api_new" / "inputs" / "school_config.json").exists()
            and (
                (repo_root / "api_new" / "inputs" / "base_routes.json").exists()
                or response_routes_present
            )
        )
        if args.use_api_inputs:
            run_dir = None
        elif args.auto_discover and not args.no_auto_discover:
            discovered = _discover_latest_run_folder()
            if discovered:
                run_dir = repo_root / discovered
            else:
                run_dir = repo_root / RUN_FOLDER_REL_PATH_DEFAULT
        elif api_inputs_present:
            run_dir = None
        else:
            # Requested behavior: auto-discovery is fallback when api_new inputs are insufficient.
            if args.no_auto_discover:
                run_dir = repo_root / RUN_FOLDER_REL_PATH_DEFAULT
            else:
                discovered = _discover_latest_run_folder()
                if discovered:
                    run_dir = repo_root / discovered
                else:
                    run_dir = repo_root / RUN_FOLDER_REL_PATH_DEFAULT

    request_path = repo_root / CHANGE_LOCATION_REQUEST_REL_PATH
    school_override_path = repo_root / SCHOOL_CONFIG_OVERRIDE_REL_PATH
    students_override_path = repo_root / STUDENTS_DATA_OVERRIDE_REL_PATH
    base_routes_override_path = repo_root / "api_new" / "inputs" / "base_routes.json"

    output_response_path = repo_root / OUTPUT_RESPONSE_REL_PATH
    output_before_map_path = repo_root / OUTPUT_BEFORE_MAP_REL_PATH
    output_after_map_path = repo_root / OUTPUT_AFTER_MAP_REL_PATH

    _ensure_seed_request_file(request_path)

    # If run_dir is provided, use run-based extraction; otherwise prefer api inputs when requested
    comparison_map_source = None
    if run_dir is not None:
        if not run_dir.exists():
            raise FileNotFoundError(f"Run folder not found: {run_dir}")
        snapshot_payload = load_json(run_dir / "snapshot_input.json")
        school_config = build_school_config(snapshot_payload)
        run_inputs = resolve_inputs_from_run(run_dir, mode_name=RUN_MODE_FOR_EXTRACTION)
        routes_students_data = run_inputs["students_data"]
        comparison_map_source = run_inputs.get("comparison_map_html")
        # Allow api overrides for students_data when present
        if students_override_path.exists():
            school_students_data = normalize_students_data_payload(load_json(students_override_path))
            write_json(students_override_path, school_students_data)
        else:
            school_students_data = build_school_students_from_routes_payload(routes_students_data)
    else:
        # Using api_new inputs path
        if not school_override_path.exists():
            raise FileNotFoundError(f"api_new/inputs/school_config.json not found; cannot run with --use-api-inputs")
        if not students_override_path.exists():
            raise FileNotFoundError(f"api_new/inputs/students_data.json not found; cannot run with --use-api-inputs")
        school_config = load_json(school_override_path)
        school_students_data = normalize_students_data_payload(load_json(students_override_path))
        if base_routes_override_path.exists():
            routes_students_data = load_json(base_routes_override_path)
        else:
            response_path = repo_root / "api_new" / "outputs" / "response.json"
            if not response_path.exists():
                raise FileNotFoundError(
                    "api_new/inputs/base_routes.json not found and api_new/outputs/response.json not found; "
                    "provide one of them to use api inputs"
                )
            response_payload = load_json(response_path)
            updated_routes = response_payload.get("updated_routes") or []
            if not updated_routes:
                raise ValueError(
                    "api_new/outputs/response.json does not contain updated_routes; "
                    "provide api_new/inputs/base_routes.json"
                )
            routes_students_data = {
                "school": school_config.get("school") or {},
                "routes": updated_routes,
                "buses": [],
            }
        write_json(students_override_path, school_students_data)
        comparison_map_candidate = repo_root / "api_new" / "outputs" / "comparison_map.html"
        if comparison_map_candidate.exists():
            comparison_map_source = str(comparison_map_candidate)

    students_data = build_operational_students_data(school_students_data, routes_students_data)

    change_location_request = load_json(request_path)

    validate_school_config(school_config)
    validate_students_data(school_students_data)
    _validate_change_location_request(change_location_request)

    response = process_change_location(
        school_config=school_config,
        students_data=students_data,
        school_students_data=school_students_data,
        change_location_request=change_location_request,
        before_map_path=str(output_before_map_path),
        after_map_path=str(output_after_map_path),
        comparison_map_source=comparison_map_source,
        skip_map=args.skip_map,
    )

    write_json(output_response_path, response)
    print(f"success= {response.get('success')}")
    print(f"response= {output_response_path}")
    print(f"before_map= {output_before_map_path}")
    if response.get("success"):
        print(f"after_map= {output_after_map_path}")
    else:
        print(f"failure_reason= {response.get('reason')}")


if __name__ == "__main__":
    main()
