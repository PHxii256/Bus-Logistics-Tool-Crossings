import copy
from detour_engine import (
    calculate_walk_penalty,
)

class ServiceSolution:
    """Represents a complete state of student-to-route assignments."""
    
    def __init__(self, students, routes, graph):
        """
        Args:
            students: List of all Student objects.
            routes: List of active Route objects.
            graph: Reference to the road network (read-only).
        """
        self.students = students
        self.routes = routes
        self.graph = graph
        
    def calculate_objective(self):
        """
        Formula: (count_served * 10000) - (active_routes * 5000)
                 - sum(route_travel_times) - walk_penalties
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
        
        # Penalise each active bus — strong enough to prefer fewer buses
        # but weaker than serving one more student (10 000 pts).
        active_routes = sum(1 for r in self.routes if r.get_student_count() > 0)
        return (served_count * 10000) - (active_routes * 5000) - total_time - total_walk_penalty
        
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
            
        return ServiceSolution(new_students, new_routes, self.graph)

    def __repr__(self):
        served = sum(1 for s in self.students if s.is_served)
        return f"ServiceSolution(served={served}/{len(self.students)}, objective={self.calculate_objective():.2f})"
