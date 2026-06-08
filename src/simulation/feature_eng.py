from geopy.distance import geodesic

def calculate_detour_cost(driver_start, driver_end, rider_start, rider_end):
    """
    Computes the detour distance for a driver to deviate course, pick up 
    a rider, drop them off, and complete their original commute.
    
    Returns:
        detour (float): The exact additional kilometers added to Driver.
        new_dist (float): Total length of the carpooled trip.
        original_dist (float): Original expected length of the solo trip.
    """
    # Original solo trip
    original_dist = geodesic(driver_start, driver_end).km
    
    # New shared trip vector: driver_start -> rider_start -> rider_end -> driver_end
    dist_to_pickup = geodesic(driver_start, rider_start).km
    dist_rider_trip = geodesic(rider_start, rider_end).km
    dist_to_dropoff = geodesic(rider_end, driver_end).km
    
    new_dist = dist_to_pickup + dist_rider_trip + dist_to_dropoff
    detour = new_dist - original_dist
    
    return detour, new_dist, original_dist

def compute_spatial_overlap(detour_km, original_driver_dist):
    """
    Overlap ratio: what fraction of the trip was the driver's original intent.
    Near 1.0 = rider is on the way. Near 0.0 = major detour.
    Clamped to [0, 1] since negative detours can push ratio > 1.
    """
    if original_driver_dist == 0:
        return 0
    # Overlap Ratio represents how much of the trip was 'original intent'.
    # Near 1.0 means almost pure overlap. Near 0.0 means complete deviation.
    # Clamp to [0, 1] -negative detours (rider on the way) can push ratio > 1.0
    ratio = original_driver_dist / (original_driver_dist + detour_km)
    return min(1.0, max(0.0, ratio))

def calculate_time_overlap(driver_time, rider_time):
    """
    Returns the absolute time difference in minutes between driver and rider pickup times.
    """
    return abs((driver_time - rider_time).total_seconds()) / 60
