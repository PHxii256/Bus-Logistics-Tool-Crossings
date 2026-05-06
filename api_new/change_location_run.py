import sys
from pathlib import Path
import glob
import json

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


# Default run folder (will be overridden by auto-discovery if runs exist)
RUN_FOLDER_REL_PATH_DEFAULT = "experiments/experiment4_crossings/627daf88_dmrt_1_0505-2126"
RUN_FOLDER_REL_PATH = _discover_latest_run_folder() or RUN_FOLDER_REL_PATH_DEFAULT
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
    run_dir = repo_root / RUN_FOLDER_REL_PATH

    if not run_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {run_dir}")

    request_path = repo_root / CHANGE_LOCATION_REQUEST_REL_PATH
    school_override_path = repo_root / SCHOOL_CONFIG_OVERRIDE_REL_PATH
    students_override_path = repo_root / STUDENTS_DATA_OVERRIDE_REL_PATH

    output_response_path = repo_root / OUTPUT_RESPONSE_REL_PATH
    output_before_map_path = repo_root / OUTPUT_BEFORE_MAP_REL_PATH
    output_after_map_path = repo_root / OUTPUT_AFTER_MAP_REL_PATH

    _ensure_seed_request_file(request_path)

    snapshot_payload = load_json(run_dir / "snapshot_input.json")
    school_config = build_school_config(snapshot_payload)
    run_inputs = resolve_inputs_from_run(run_dir, mode_name=RUN_MODE_FOR_EXTRACTION)
    routes_students_data = run_inputs["students_data"]
    comparison_map_source = run_inputs.get("comparison_map_html")

    if school_override_path.exists():
        school_config = load_json(school_override_path)
    else:
        write_json(school_override_path, school_config)

    if students_override_path.exists():
        school_students_data = normalize_students_data_payload(load_json(students_override_path))
        write_json(students_override_path, school_students_data)
    else:
        school_students_data = build_school_students_from_routes_payload(routes_students_data)
        write_json(students_override_path, school_students_data)

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
