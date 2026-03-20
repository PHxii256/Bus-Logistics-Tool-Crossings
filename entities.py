from enum import Enum

class School_Stage(Enum):
    KG = 1
    ELEMENTARY = 2
    MIDDLE = 3
    HIGH = 4

def calc_max_walk_distance(school_stage):
    match school_stage:
        case School_Stage.KG:
            return 0
        case School_Stage.ELEMENTARY:
            return 0
        case School_Stage.MIDDLE:
            return 100
        case School_Stage.HIGH:
            return 200


class Student:  
    def __init__(self, id, lat, lon, age, school_stage, fee,
                 assignment="permanent", valid_from=None, valid_until=None):
        self.id = id
        self.coords = (lat, lon)          # home location
        self.age = age
        self.school_stage = school_stage
        self.fee = fee
        self.walk_radius = calc_max_walk_distance(school_stage)
        self.assigned_stop = None
        self.is_served = False
        self.failure_reason = ""
        # Assignment type: "permanent" or "temporary"
        self.assignment = assignment
        # Date range for temporary assignments (ISO strings or None)
        self.valid_from = valid_from
        self.valid_until = valid_until
        # Cached direct travel time (minutes) from home node to school node
        # Computed once during precompute phase; used for per-student Tmax constraint
        self.direct_time_to_school   = None  # home → school  (morning)
        self.direct_time_from_school = None  # school → home  (afternoon, may differ on one-way streets)

        

class Bus:
    def __init__(self, bus_type, capacity, fixed_cost, var_cost_km):
        self.bus_type = bus_type
        self.capacity = capacity
        self.fixed_cost = fixed_cost
        self.var_cost_km = var_cost_km


class Stop:
    """Represents a pickup/dropoff location on a route.
    
    A stop is a location (snapped to the road network) where one or more
    students can be picked up or dropped off. Each stop tracks which students
    are assigned to it and ensures they can reach it within their walk radius.
    """
    def __init__(self, node_id, lat, lon, stop_id=None, stop_type="pickup"):
        """Initialize a Stop.
        
        Args:
            node_id: The OSM node ID from the road network
            lat: Latitude coordinate
            lon: Longitude coordinate
            stop_id: Optional unique identifier for the stop
            stop_type: "school" or "pickup"
        """
        self.node_id = node_id
        self.coords = (lat, lon)  # (lat, lon) tuple
        self.students = []
        self.stop_id = stop_id
        self.stop_type = stop_type      # "school" or "pickup"
        
    def add_student(self, student):
        """Add a student to this stop.
        
        Args:
            student: Student object
        """
        if student not in self.students:
            self.students.append(student)
            student.assigned_stop = self
            student.is_served = True
            
    def remove_student(self, student):
        """Safely remove a student and reset their status."""
        if student in self.students:
            self.students.remove(student)
            student.assigned_stop = None
            student.is_served = False
                
    def is_full(self, bus_capacity):
        """Check if this stop has reached capacity.
        
        Args:
            bus_capacity: The bus capacity
            
        Returns:
            bool: True if number of students equals bus capacity
        """
        return len(self.students) >= bus_capacity
    
    def get_student_count(self):
        """Get number of students at this stop."""
        return len(self.students)
    
    def __repr__(self):
        return f"Stop(node={self.node_id}, type={self.stop_type}, students={len(self.students)})"


class Route:
    """Represents a bus route with ordered pickup/dropoff stops.
    
    A route consists of a bus, a sequence of stops, and associated metrics
    like distance, time, and cost. The route tracks temporary detour time
    used during the day.
    """
    def __init__(self, bus, route_id=None, route_tmax=60,
                 ride_time_multiplier=2.5, floor_minutes=45, ceiling_minutes=30,
                 bidirectional_check=True):
        """Initialize a Route.
        
        Args:
            bus: Bus object assigned to this route
            route_id: Optional unique identifier for the route
            route_tmax: Legacy flat Tmax (minutes); used as fallback only
            ride_time_multiplier: Ratio cap k; cap = clamp(k*T_direct, floor, T_direct+ceiling)
            floor_minutes: Minimum cap — ensures nearby students don't over-penalise fleet (default 45)
            ceiling_minutes: Max EXTRA minutes allowed beyond direct route time (default 30)
                             Absolute cap = T_direct + ceiling_minutes
            bidirectional_check: If True, a student is only rejected for ride-time when BOTH
                                 the morning (home→school) AND afternoon (school→home) rides
                                 exceed their cap.  If False, only the morning ride is checked.
        """
        self.bus = bus
        self.stops = [] # List of Stop objects in order
        self.total_distance = 0
        self.total_time = 0  # Total travel time in minutes
        self.route_id = route_id
        self.route_tmax = route_tmax
        # Per-student ride time constraint: T_max = clamp(k * T_direct, floor_minutes, ceiling_minutes)
        self.ride_time_multiplier = ride_time_multiplier
        self.floor_minutes        = floor_minutes
        self.ceiling_minutes      = ceiling_minutes
        self.bidirectional_check  = bidirectional_check
        self.detour_time_used = 0  # Track temporary detour time used today
        
    def get_revenue(self):
        """Calculate total revenue from all students on this route."""
        return sum(student.fee for stop in self.stops for student in stop.students)

    def get_total_cost(self):
        """Calculate total cost: fixed + variable (distance-based)."""
        return self.bus.fixed_cost + (self.total_distance * self.bus.var_cost_km)

    def get_profit_margin(self):
        """Calculate profit margin as (revenue - cost) / revenue."""
        revenue = self.get_revenue()
        if revenue == 0: 
            return 0
        return (revenue - self.get_total_cost()) / revenue
    
    def get_current_detour_time(self):
        """Get cumulative temporary detour time used today in minutes."""
        return self.detour_time_used
    
    def add_detour_time(self, time_minutes):
        """Add time to the detour counter (for temporary requests).
        
        Args:
            time_minutes: Time in minutes to add
        """
        self.detour_time_used += time_minutes
    
    def reset_daily_detour_time(self):
        """Reset the daily detour time counter (call at start of new day)."""
        self.detour_time_used = 0
    
    def get_student_count(self):
        """Get total number of students on this route."""
        return sum(stop.get_student_count() for stop in self.stops)
    
    def is_at_capacity(self):
        """Check if any stop on the route is at or over bus capacity."""
        return any(stop.is_full(self.bus.capacity) for stop in self.stops)
    
    def __repr__(self):
        return f"Route(id={self.route_id}, stops={len(self.stops)}, students={self.get_student_count()})"