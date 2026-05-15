"""
CLI "endpoint" for ETA: read api_new/inputs/eta_request.json, write api_new/outputs/eta_response.json.

Run from repository root (so ``cache/`` and imports resolve consistently), for example:
  python api_new/eta_run.py

Paste-and-run (Windows):
  .\\.venv\\Scripts\\python.exe api_new\\eta_run.py
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api_new.scripts.data_extractors import load_json, write_json
from api_new.scripts.eta_calculator import build_eta_response

ETA_REQUEST_REL_PATH = "api_new/inputs/eta_request.json"
ETA_RESPONSE_REL_PATH = "api_new/outputs/eta_response.json"
SCHEMA_REQUEST = REPO_ROOT / "api_new" / "schemas" / "eta_request.json"


def _ensure_seed_request_file(path: Path):
    if path.exists():
        return
    if SCHEMA_REQUEST.exists():
        payload = load_json(SCHEMA_REQUEST)
        clean = {k: v for k, v in payload.items() if not str(k).startswith("_")}
        write_json(path, clean)
    else:
        write_json(
            path,
            {
                "from_coord": [29.9615107, 31.2762834],
                "to_coord": [29.9700, 31.2900],
                "congestion_factor": 1.0,
            },
        )
    raise FileNotFoundError(
        f"Seeded missing request file at {path}. Edit it, then rerun."
    )


def main():
    os.chdir(REPO_ROOT)
    request_path = REPO_ROOT / ETA_REQUEST_REL_PATH
    response_path = REPO_ROOT / ETA_RESPONSE_REL_PATH

    _ensure_seed_request_file(request_path)
    payload = load_json(request_path)
    response = build_eta_response(payload)
    write_json(response_path, response)

    if response.get("error"):
        print(f"error= {response['error']}")
    else:
        print(f"eta_minutes= {response.get('eta')}")
    print(f"wrote= {response_path}")


if __name__ == "__main__":
    main()
