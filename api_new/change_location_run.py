import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api_new.scripts.change_location_pipeline import process_change_location
from api_new.scripts.data_extractors import (
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
RUN_FOLDER_REL_PATH = "experiments/experiment4_crossings/200_students/42/3ff9ad02_dmrt_1_0408-2240"
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
    comparison_map_candidate = run_dir / "comparison_map.html"
    comparison_map_source = str(comparison_map_candidate) if comparison_map_candidate.exists() else None

    if school_override_path.exists():
        school_config = load_json(school_override_path)
    else:
        write_json(school_override_path, school_config)

    if students_override_path.exists():
        students_data = normalize_students_data_payload(load_json(students_override_path))
        # Keep persisted payload canonical (routes -> stops -> students), even for override files.
        write_json(students_override_path, students_data)
    else:
        run_inputs = resolve_inputs_from_run(run_dir, mode_name=RUN_MODE_FOR_EXTRACTION)
        students_data = run_inputs["students_data"]
        if run_inputs.get("comparison_map_html"):
            comparison_map_source = run_inputs["comparison_map_html"]
        write_json(students_override_path, students_data)

    change_location_request = load_json(request_path)

    validate_school_config(school_config)
    validate_students_data(students_data)
    _validate_change_location_request(change_location_request)

    response = process_change_location(
        school_config=school_config,
        students_data=students_data,
        change_location_request=change_location_request,
        before_map_path=str(output_before_map_path),
        after_map_path=str(output_after_map_path),
        comparison_map_source=comparison_map_source,
    )

    write_json(output_response_path, response)
    print(f"success={response.get('success')}")
    print(f"response={output_response_path}")
    print(f"before_map={output_before_map_path}")
    if response.get("success"):
        print(f"after_map={output_after_map_path}")
    else:
        print(f"failure_reason={response.get('reason')}")


if __name__ == "__main__":
    main()
