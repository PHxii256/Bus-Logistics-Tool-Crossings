import copy
import math
from detour_engine import (
    calculate_walk_penalty,
    compute_direct_time,
    find_shortest_path_with_turns,
    _MATRIX_CACHE,
)

class ServiceSolution:
    """Represents a complete state of student-to-route assignments."""
    
    def __init__(self, students, routes, graph, cap_penalty_per_minute=0.0):
        """
        Args:
            students: List of all Student objects.
            routes: List of active Route objects.
            graph: Reference to the road network (read-only).
        """
        self.students = students
        self.routes = routes
        self.graph = graph
        self.cap_penalty_per_minute = float(cap_penalty_per_minute or 0.0)
        
    def calculate_objective(self):
        """
        Formula: (count_served * 10000) - sum(route_travel_times) - walk_penalties
        Walk penalties discourage assigning students to stops beyond their
        recommended walking radius, while still allowing it when necessary.
        """
        served_count = sum(1 for s in self.students if s.is_served)
        total_time = sum(r.total_time for r in self.routes)
        
        # Calculate walk penalties for all served students
        total_walk_penalty = 0.0
        for student in self.students:
            if not student.is_served or not student.assigned_stop:
                continue
            penalty, walk_m, over_limit = calculate_walk_penalty(
                student, student.assigned_stop.node_id, self.graph
            )
            if penalty == float('inf'):
                total_walk_penalty += 5000  # Heavy penalty but don't fully reject
            else:
                total_walk_penalty += penalty
        
        # Penalise ride-time cap violations (soft penalty mode)
        cap_penalty = 0.0
        if self.cap_penalty_per_minute > 0 and self.graph is not None:
            for route in self.routes:
                if not route.stops:
                    continue
                school_node = route.stops[-1].node_id
                k = getattr(route, 'ride_time_multiplier', 2.5)
                floor_min = getattr(route, 'floor_minutes', 45)
                ceiling_min = getattr(route, 'ceiling_minutes', 60)
                bidir = getattr(route, 'bidirectional_check', True)

                # AM ride times by stop (stop -> school)
                am_time_by_stop = {}
                total = 0.0
                ok = True
                for i in range(len(route.stops) - 1, 0, -1):
                    u = route.stops[i - 1].node_id
                    v = route.stops[i].node_id
                    t = _MATRIX_CACHE.get((u, v), None)
                    if t is None:
                        _, t = find_shortest_path_with_turns(self.graph, u, v)
                    if not math.isfinite(t):
                        ok = False
                        break
                    total += t
                    am_time_by_stop[route.stops[i - 1]] = total
                if not ok:
                    am_time_by_stop = {}

                # PM ride times by stop (school -> stop in reversed route)
                pm_time_by_stop = {}
                if bidir:
                    interior = route.stops[1:-1][::-1]
                    pm_stops = [route.stops[0]] + interior + [route.stops[-1]]
                    total = 0.0
                    ok = True
                    for i in range(len(pm_stops) - 1):
                        u = pm_stops[i].node_id
                        v = pm_stops[i + 1].node_id
                        t = _MATRIX_CACHE.get((u, v), None)
                        if t is None:
                            _, t = find_shortest_path_with_turns(self.graph, u, v)
                        if not math.isfinite(t):
                            ok = False
                            break
                        total += t
                        pm_time_by_stop[pm_stops[i + 1]] = total
                    if not ok:
                        pm_time_by_stop = {}

                for stop in route.stops:
                    if stop.stop_type == 'school':
                        continue
                    ride_am = am_time_by_stop.get(stop)
                    ride_pm = pm_time_by_stop.get(stop) if bidir else None
                    if ride_am is None:
                        continue
                    for student in stop.students:
                        t_direct = compute_direct_time(student, school_node, self.graph)
                        if t_direct is None or not math.isfinite(t_direct) or t_direct <= 0:
                            continue
                        cap = max(floor_min, min(k * t_direct, t_direct + ceiling_min))
                        am_over = ride_am - cap
                        if bidir and ride_pm is not None:
                            pm_over = ride_pm - cap
                            if am_over > 0 and pm_over > 0:
                                cap_penalty += max(am_over, pm_over) * self.cap_penalty_per_minute
                        else:
                            if am_over > 0:
                                cap_penalty += am_over * self.cap_penalty_per_minute

        # Penalise each active bus — strong enough to prefer fewer buses
        # but weaker than serving one more student (10 000 pts).
        active_routes = sum(1 for r in self.routes if r.get_student_count() > 0)
        return (served_count * 10000) - (active_routes * 5000) - total_time - total_walk_penalty - cap_penalty
        
    def clone(self):
        """
        Creates a deep clone by rebuilding the student-stop relationships.
        Ensures the new solution's objects do not point back to the old one.
        """
        # 1. Clone students and reset temporary state
        new_students = [copy.copy(s) for s in self.students]
        for s in new_students:
            s.assigned_stop = None
            s.is_served = False
            
        student_map = {s.id: s for s in new_students}
        
        # 2. Clone routes and their internal stops
        new_routes = []
        for old_route in self.routes:
            new_route = copy.copy(old_route)
            new_route.stops = []
            
            for old_stop in old_route.stops:
                new_stop = copy.copy(old_stop)
                new_stop.students = [] # Clear the old reference list
                
                # Re-link corresponding new students to this new stop
                for old_val_student in old_stop.students:
                    if old_val_student.id in student_map:
                        # add_student updates student.is_served and student.assigned_stop
                        new_stop.add_student(student_map[old_val_student.id])
                
                new_route.stops.append(new_stop)
            
            new_routes.append(new_route)
            
        return ServiceSolution(new_students, new_routes, self.graph,
                       cap_penalty_per_minute=self.cap_penalty_per_minute)

    def __repr__(self):
        served = sum(1 for s in self.students if s.is_served)
        return f"ServiceSolution(served={served}/{len(self.students)}, objective={self.calculate_objective():.2f})"
