"""
Loader for Lewis SBVRP benchmark .bus files.

File format (reverse-engineered from Porthcawl.bus and ProblemFormat.pdf):
─────────────────────────────────────────────────────────────────────────────
Line 1  (header):
    n_stops,n_pupils,n_combos,K,max_walk_km,<extra>,<bearing_flag>,
    WalkingDists=GoogleMaps,Created=<date>

    n_stops  — number of `s,` lines (stop 0 = school, 1..n-1 = candidates)
    n_pupils — number of `a,` lines (pupil address records; each may cover
               several children via the group_size field)
    n_combos — number of `w,` lines (valid pupil→stop walk pairs ≤ max_walk_km)

Lines 2 .. n_stops+1:
    s,<lat>,<lon>,<name>
    Index 0 is the school; indices 1..n_stops-1 are candidate bus stops.

Lines n_stops+2 .. n_stops+n_pupils+1:
    a,<lat>,<lon>,<group_size>,<surname>
    group_size = number of siblings at this address who attend the same school.

Next n_stops² lines:
    d,<i>,<j>,<walk_dist_km>,<walk_time_sec>
    Full stop-to-stop (bus travel) distance/time matrix.
    i,j ∈ [0, n_stops-1].  d[0][0] = 0.

Final n_combos lines:
    w,<pupil_addr_idx>,<stop_idx>,<walk_dist_km>,<walk_time_sec>
    Only those pairs where walk_dist_km ≤ max_walk_km are listed.

Usage
─────
    from bus_loader import load_bus_file
    inst = load_bus_file('busprobs/Porthcawl.bus')

    # Entity objects ready for the ALNS:
    inst.students       → list[Student]
    inst.routes         → list[Route]   (one [school, school] shell per bus)
    inst.school_coords  → (lat, lon)
    inst.n_total_pupils → int
    inst.instance_name  → str (e.g. "Porthcawl")

After load_bus_file() returns:
    • detour_engine._SNAP_OVERRIDE is populated for every student coord →
      snap_address_to_edge() works without a real OSMnx graph.
    • detour_engine._MATRIX_CACHE is pre-filled from the d-matrix (minutes).
    • detour_engine._MATRIX_CACHE_LENGTH is pre-filled from the d-matrix (meters).
    • student.direct_time_to_school and .direct_time_from_school are set (minutes).
"""

import os
import math


# ---------------------------------------------------------------------------
# Helper: Haversine distance in km (used as fallback when no w-line exists)
# ---------------------------------------------------------------------------

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class BusProblemInstance:
    """Container for parsed benchmark data and pre-built ALNS-ready entities."""

    def __init__(self):
        self.instance_name  = ""
        self.school_coords  = (0.0, 0.0)   # (lat, lon) of school (stop 0)
        self.n_stops        = 0            # total s-lines (incl. school)
        self.n_pupils       = 0            # a-lines (pupil address records)
        self.n_combos       = 0            # w-lines
        self.max_walk_km    = 0.0
        self.n_total_pupils = 0            # sum of group_sizes → individual pupils

        self.stops_coords   = []           # (lat, lon) per stop index
        self.stops_names    = []           # stop name strings

        # Already-built ALNS entity objects
        self.students = []   # list[Student]
        self.routes   = []   # list[Route]  — empty [school, school] shells
        self.buses    = []   # list[Bus]

        # Raw d-matrix: (i, j) -> (dist_km, time_sec)
        self.dist_matrix = {}


def load_bus_file(path: str,
                  bus_capacity: int = 70,
                  bus_fixed_cost: float = 0.0,
                  bus_var_cost_km: float = 1.0,
                  alns_constraints: dict | None = None) -> BusProblemInstance:
    """Parse a .bus file and return a BusProblemInstance ready for the ALNS.

    Parameters
    ----------
    path            : Path to the .bus file.
    bus_capacity    : Seat capacity for every bus (default 70 per Lewis paper).
    bus_fixed_cost  : Cost per bus (not used in objective, kept for compatibility).
    bus_var_cost_km : Variable cost per km (not used in objective by default).
    alns_constraints: Optional dict with keys ride_time_multiplier, floor_minutes,
                      ceiling_minutes that are injected into each Route object.

    Returns
    -------
    BusProblemInstance
        Fully populated; _MATRIX_CACHE and _SNAP_OVERRIDE in detour_engine
        are side-effect-populated.
    """
    from entities import Student, Bus, Route, Stop, School_Stage
    import detour_engine as _eng

    inst = BusProblemInstance()
    inst.instance_name = os.path.splitext(os.path.basename(path))[0]

    constraints = alns_constraints or {}

    # ------------------------------------------------------------------
    # 1. Parse file
    # ------------------------------------------------------------------
    with open(path, encoding='utf-8', errors='replace') as fh:
        lines = fh.readlines()

    # Header
    header = lines[0].strip().split(',')
    inst.n_stops  = int(header[0])
    inst.n_pupils = int(header[1])
    inst.n_combos = int(header[2])
    # header[3] is the literal string 'K' — ignore
    inst.max_walk_km = float(header[4])

    # Stop lines
    stop_lines_start = 1
    stop_lines_end   = stop_lines_start + inst.n_stops      # exclusive
    for raw in lines[stop_lines_start:stop_lines_end]:
        parts = raw.strip().split(',', 3)                   # s, lat, lon, name
        lat, lon = float(parts[1]), float(parts[2])
        name = parts[3] if len(parts) > 3 else ""
        inst.stops_coords.append((lat, lon))
        inst.stops_names.append(name)

    inst.school_coords = inst.stops_coords[0]

    # Pupil address lines
    pupil_lines_start = stop_lines_end
    pupil_lines_end   = pupil_lines_start + inst.n_pupils
    pupil_records = []   # list of (lat, lon, group_size, name)
    for raw in lines[pupil_lines_start:pupil_lines_end]:
        parts = raw.strip().split(',', 4)                   # a, lat, lon, grp, name
        lat, lon  = float(parts[1]), float(parts[2])
        grp_size  = int(parts[3])
        name      = parts[4] if len(parts) > 4 else f"Pupil{len(pupil_records)}"
        pupil_records.append((lat, lon, grp_size, name))
        inst.n_total_pupils += grp_size

    # d-matrix (full stop×stop bus travel matrix)
    d_lines_start = pupil_lines_end
    d_lines_count = inst.n_stops * inst.n_stops
    d_lines_end   = d_lines_start + d_lines_count
    for raw in lines[d_lines_start:d_lines_end]:
        parts = raw.strip().split(',')
        i, j         = int(parts[1]), int(parts[2])
        dist_km      = float(parts[3])
        time_sec     = float(parts[4])
        inst.dist_matrix[(i, j)] = (dist_km, time_sec)

    # w-lines (valid pupil→stop walk assignments)
    walk_combos = {}    # pupil_addr_idx -> list of (stop_idx, walk_km, walk_sec)
    w_lines_start = d_lines_end
    w_lines_end   = w_lines_start + inst.n_combos
    for raw in lines[w_lines_start:w_lines_end]:
        parts = raw.strip().split(',')
        pu_idx, st_idx = int(parts[1]), int(parts[2])
        walk_km  = float(parts[3])
        walk_sec = float(parts[4])
        walk_combos.setdefault(pu_idx, []).append((st_idx, walk_km, walk_sec))

    # ------------------------------------------------------------------
    # 2. Pre-populate _MATRIX_CACHE and _MATRIX_CACHE_LENGTH
    # ------------------------------------------------------------------
    # _MATRIX_CACHE        (source, target) -> travel_time in MINUTES
    # _MATRIX_CACHE_LENGTH (source, target) -> distance in METERS
    _eng._MATRIX_CACHE.clear()
    _eng._MATRIX_CACHE_LENGTH.clear()
    _eng._SNAP_OVERRIDE.clear()

    for (i, j), (dist_km, time_sec) in inst.dist_matrix.items():
        _eng._MATRIX_CACHE[(i, j)]        = time_sec / 60.0          # → minutes
        _eng._MATRIX_CACHE_LENGTH[(i, j)] = dist_km * 1000.0         # → metres

    # ------------------------------------------------------------------
    # 3. Assign each pupil address to its nearest walkable stop
    # ------------------------------------------------------------------
    # If a pupil has valid w-lines use the closest one; otherwise fall back
    # to the min-haversine candidate stop.

    pupil_stop_assignments = []   # (pupil_addr_idx, assigned_stop_idx, travel_time_sec_to_school)
    for addr_idx, (p_lat, p_lon, grp_size, p_name) in enumerate(pupil_records):
        combos = walk_combos.get(addr_idx, [])
        if combos:
            # Nearest walkable stop by walk distance
            best = min(combos, key=lambda x: x[1])
            stop_idx = best[0]
        else:
            # No walkable stop: snap to nearest candidate stop by straight-line distance
            # (exclude school at index 0 — school is destination, not pickup stop)
            best_dist = float('inf')
            stop_idx  = 1  # default to first candidate
            for si in range(1, inst.n_stops):
                s_lat, s_lon = inst.stops_coords[si]
                d = _haversine_km(p_lat, p_lon, s_lat, s_lon)
                if d < best_dist:
                    best_dist = d
                    stop_idx  = si

        # Travel time from assigned stop → school (stop 0)
        travel_sec = inst.dist_matrix.get((stop_idx, 0), (0.0, 0.0))[1]
        pupil_stop_assignments.append((addr_idx, stop_idx, travel_sec))

    # ------------------------------------------------------------------
    # 4. Build Student objects (one per individual pupil in group_size)
    # ------------------------------------------------------------------
    student_id_counter = 1
    for addr_idx, stop_idx, travel_sec in pupil_stop_assignments:
        p_lat, p_lon, grp_size, p_name = pupil_records[addr_idx]
        s_lat, s_lon = inst.stops_coords[stop_idx]

        # Travel time from school → assigned stop (afternoon direction)
        travel_sec_from_school = inst.dist_matrix.get((0, stop_idx), (0.0, 0.0))[1]

        for sibling in range(grp_size):
            student = Student(
                id           = f"{inst.instance_name}_{student_id_counter:04d}",
                lat          = s_lat,          # coords = assigned stop (already snapped)
                lon          = s_lon,
                age          = 15,             # generic high-school age
                school_stage = School_Stage.HIGH,
                fee          = 100.0,
                assignment   = "permanent",
            )
            # Zero walk_radius → bypass all walk-penalty / safe-node machinery
            student.walk_radius = 0

            # Pre-set direct travel times so compute_direct_time never needs the graph
            student.direct_time_to_school   = travel_sec / 60.0         # → minutes
            student.direct_time_from_school = travel_sec_from_school / 60.0

            # Register in _SNAP_OVERRIDE so snap_address_to_edge() returns immediately
            _eng._SNAP_OVERRIDE[(s_lat, s_lon)] = (stop_idx, (s_lat, s_lon))

            inst.students.append(student)
            student_id_counter += 1

    # ------------------------------------------------------------------
    # 5. Build Bus and empty Route objects
    # ------------------------------------------------------------------
    n_buses_needed = math.ceil(inst.n_total_pupils / bus_capacity) + 2  # slack for ALNS
    sch_lat, sch_lon = inst.school_coords
    school_node = 0   # stop index 0 = school

    ride_time_multiplier = constraints.get('ride_time_multiplier', 2.5)
    floor_minutes        = constraints.get('floor_minutes',        45)
    ceiling_minutes      = constraints.get('ceiling_minutes',      60)

    for i in range(n_buses_needed):
        bus = Bus(
            bus_type      = 'standard',
            capacity      = bus_capacity,
            fixed_cost    = bus_fixed_cost,
            var_cost_km   = bus_var_cost_km,
        )
        bus.bus_id = f"{inst.instance_name}_Bus{i+1}"
        inst.buses.append(bus)

        route = Route(
            bus                  = bus,
            route_id             = f"{inst.instance_name}_R{i+1}",
            route_tmax           = 90,
            ride_time_multiplier = ride_time_multiplier,
            floor_minutes        = floor_minutes,
            ceiling_minutes      = ceiling_minutes,
        )
        # Initialise with [school_start, school_end] shells
        start_stop = Stop(school_node, sch_lat, sch_lon,
                          stop_id=f"{inst.instance_name}_R{i+1}-Start",
                          stop_type="school")
        end_stop   = Stop(school_node, sch_lat, sch_lon,
                          stop_id=f"{inst.instance_name}_R{i+1}-End",
                          stop_type="school")
        route.stops = [start_stop, end_stop]
        inst.routes.append(route)

    # Also register the school node in _SNAP_OVERRIDE (used by bus-reachability checks)
    _eng._SNAP_OVERRIDE[(sch_lat, sch_lon)] = (school_node, (sch_lat, sch_lon))

    print(f"[bus_loader] Loaded '{inst.instance_name}': "
          f"{inst.n_stops} stops, {inst.n_total_pupils} pupils -> "
          f"{len(inst.students)} students, {n_buses_needed} buses.")
    print(f"  Matrix entries: {len(_eng._MATRIX_CACHE)}, "
          f"Snap overrides: {len(_eng._SNAP_OVERRIDE)}")

    return inst
