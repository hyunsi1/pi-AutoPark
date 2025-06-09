import os
import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the Earth's surface given in decimal degrees.
    Returns distance in meters.
    """
    # Convert decimal degrees to radians
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    R = 6371000  # Earth radius in meters
    return R * c

def euclidean_distance(x1, y1, x2, y2):
    """
    Calculate Euclidean distance between two points in planar coordinates.
    """
    return math.hypot(x2 - x1, y2 - y1)
