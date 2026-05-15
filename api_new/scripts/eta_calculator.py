"""
ETA between two lat/lon points on the drive road network.

Uses the same pipeline as main route timing: OSM drive graph with per-edge
``travel_time`` (minutes) and turn-aware routing via ``find_shortest_path_with_turns``.
"""

import math
import sys
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import networkx as nx
import osmnx as ox

import detour_engine as _detour_engine
from detour_engine import find_shortest_path_with_turns
from run_algorithm import setup_graph


def _float_pair(raw: Any, field_name: str) -> Tuple[float, float]:
    if raw is None:
        raise ValueError(f"{field_name} is required")
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raise ValueError(f"{field_name} must be a list or tuple [latitude, longitude]")
    lat = float(raw[0])
    lon = float(raw[1])
    if not math.isfinite(lat) or not math.isfinite(lon):
        raise ValueError(f"{field_name} has non-finite coordinates")
    if abs(lat) > 90 or abs(lon) > 180:
        raise ValueError(f"{field_name} coordinates out of range (lat in [-90,90], lon in [-180,180])")
    return lat, lon


def _padded_bbox(
    lat1: float, lon1: float, lat2: float, lon2: float, pad_deg: float = 0.06
) -> list:
    """Expand endpoints so OSM drive subgraphs stay strongly connected around the corridor."""
    min_lat = min(lat1, lat2)
    max_lat = max(lat1, lat2)
    min_lon = min(lon1, lon2)
    max_lon = max(lon1, lon2)
    span = max(max_lat - min_lat, max_lon - min_lon, 1e-6)
    # Tiny bboxes often clip one-way components so A* finds no directed route; keep a floor pad.
    pad = max(pad_deg, 0.12 * span)
    return [
        min_lat - pad,
        min_lon - pad,
        max_lat + pad,
        max_lon + pad,
    ]


def parse_eta_request(payload: Mapping[str, Any]) -> Tuple[float, float, float, float, float]:
    lat_a, lon_a = _float_pair(payload.get("from_coord"), "from_coord")
    lat_b, lon_b = _float_pair(payload.get("to_coord"), "to_coord")
    factor_raw = payload.get("congestion_factor", 1.0)
    try:
        congestion = float(factor_raw)
    except (TypeError, ValueError):
        raise ValueError("congestion_factor must be a number") from None
    if congestion <= 0:
        raise ValueError("congestion_factor must be positive")
    return lat_a, lon_a, lat_b, lon_b, congestion


def compute_eta_minutes(
    from_coord: Any,
    to_coord: Any,
    congestion_factor: float = 1.0,
    *,
    graph_cache_enabled: bool = True,
    road_meta: Optional[MutableMapping[str, Any]] = None,
) -> float:
    """
    Road-network travel time between two [lat, lon] pairs (minutes), with optional multiplicative congestion.

    Raises ValueError on bad input; RuntimeError when no drive path exists.
    """
    payload = {
        "from_coord": from_coord,
        "to_coord": to_coord,
        "congestion_factor": congestion_factor,
    }
    lat_a, lon_a, lat_b, lon_b, factor = parse_eta_request(payload)
    bbox = _padded_bbox(lat_a, lon_a, lat_b, lon_b)
    meta = dict(road_meta or {})
    meta.setdefault("graph", {})
    gcfg = meta["graph"]
    if not isinstance(gcfg, dict):
        raise ValueError("road_meta['graph'] must be a dict when provided")
    gcfg = dict(gcfg)
    gcfg["bbox"] = bbox
    gcfg.setdefault("boundary_mode", "bbox")
    gcfg.setdefault("cache", {"enabled": graph_cache_enabled})
    meta["graph"] = gcfg

    # Avoid stale (source, target) entries from other graphs / earlier failed lookups.
    _detour_engine._path_cache.clear()
    _detour_engine._MATRIX_CACHE.clear()

    G = setup_graph(meta, unconstrained=False)
    node_a = ox.distance.nearest_nodes(G, X=float(lon_a), Y=float(lat_a))
    node_b = ox.distance.nearest_nodes(G, X=float(lon_b), Y=float(lon_b))

    _path, minutes = find_shortest_path_with_turns(G, node_a, node_b, weight="travel_time")
    if minutes is None or not math.isfinite(minutes) or minutes >= 1e6:
        try:
            minutes = nx.shortest_path_length(G, node_a, node_b, weight="travel_time")
        except nx.NetworkXNoPath as exc:
            raise RuntimeError(
                "No drive route found between the snapped road nodes (directed graph)."
            ) from exc

    if minutes is None or not math.isfinite(minutes) or minutes >= 1e6:
        raise RuntimeError("No finite drive route found between the snapped road nodes.")

    return float(minutes) * float(factor)


def build_eta_response(payload: Mapping[str, Any]) -> dict:
    """Validate request body and return ``{\"eta\": <minutes>}`` or error fields."""
    try:
        lat_a, lon_a, lat_b, lon_b, factor = parse_eta_request(payload)
        eta = compute_eta_minutes(
            [lat_a, lon_a],
            [lat_b, lon_b],
            factor,
        )
        return {"eta": round(eta, 4)}
    except (ValueError, RuntimeError) as exc:
        return {
            "eta": None,
            "error": str(exc),
        }
