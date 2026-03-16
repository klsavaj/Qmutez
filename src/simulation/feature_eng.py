import numpy as np
from geopy.distance import geodesic

def calculate_haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees).
    
    Why this is used instead of Euclidian Distance:
    Because Euclidean grids do not account for the spherical curvature 
    of the earth, utilizing standard pythagorean math leads to high variance 
    errors in multi-objective routing algorithms.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Standard mathematical radius of earth in kilometers
    return c * r

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
    Computes a simple proportional heuristic representing "Overlap Ratio". 
    
    Why:
    An absolute detour of 2km is unacceptable for a 3km commute, but 
    highly acceptable for a 45km commute. This ratio creates a normalized
    feature allowing XGBoost to learn relative tolerance.
    """
    if original_driver_dist == 0:
        return 0
    # Overlap Ratio represents how much of the trip was 'original intent'.
    # Near 1.0 means almost pure overlap. Near 0.0 means complete deviation.
    # Clamp to [0, 1] — negative detours (rider on the way) can push ratio > 1.0
    ratio = original_driver_dist / (original_driver_dist + detour_km)
    return min(1.0, max(0.0, ratio))

def calculate_time_overlap(driver_time, rider_time, threshold_mins=15):
    """
    Validates temporal compatibility between a driver's commute start and 
    a rider's request timestamp.
    """
    time_diff = abs((driver_time - rider_time).total_seconds()) / 60
    is_compatible = time_diff <= threshold_mins
    return is_compatible, time_diff
