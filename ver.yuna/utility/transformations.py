import math
import numpy as np
import cv2

# Earth's radius in meters
EARTH_RADIUS = 6371000.0

def latlon_to_xy(lat, lon, lat0, lon0):
    """
    Convert latitude/longitude to local planar coordinates (x East, y North) in meters
    relative to a reference point (lat0, lon0).
    """
    # Convert degrees to radians once for lat0, lon0
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lat0_rad = math.radians(lat0)
    lon0_rad = math.radians(lon0)

    # Differences
    dlat = lat_rad - lat0_rad
    dlon = lon_rad - lon0_rad

    # Approximate conversions
    x = EARTH_RADIUS * dlon * math.cos(lat0_rad)
    y = EARTH_RADIUS * dlat
    return x, y

def xy_to_latlon(x, y, lat0, lon0):
    """
    Convert local planar coordinates (x East, y North) in meters back to latitude/longitude
    relative to a reference point (lat0, lon0).
    """
    # Convert lat0 to radians once
    lat0_rad = math.radians(lat0)

    dlat = y / EARTH_RADIUS
    dlon = x / (EARTH_RADIUS * math.cos(lat0_rad))

    lat = math.degrees(dlat + math.radians(lat0))
    lon = math.degrees(dlon + math.radians(lon0))
    return lat, lon

def compute_homography(src_pts, dst_pts):
    """
    Compute homography matrix H such that dst_pts ~ H * src_pts.
    src_pts and dst_pts are Nx2 arrays of corresponding points.
    Returns 3x3 homography matrix.
    """
    src = np.asarray(src_pts, dtype=np.float32)
    dst = np.asarray(dst_pts, dtype=np.float32)

    # Check that we have at least 4 points to compute the homography
    if len(src) < 4 or len(dst) < 4:
        raise ValueError("At least 4 points are required to compute the homography.")

    H, status = cv2.findHomography(src, dst, method=cv2.RANSAC)
    
    if H is None:
        raise ValueError("Homography computation failed.")
    
    return H

def warp_point(pt, H):
    """
    Apply homography H to a 2D point pt=(x, y).
    Returns warped point (x', y').
    """
    vec = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
    warped = H.dot(vec)
    warped /= warped[2]  # Normalize by the third (homogeneous) coordinate
    return float(warped[0]), float(warped[1])

def pixel_to_world(x_pixel, y_pixel, homography_matrix):
    """
    Convert pixel coordinates to world coordinates using homography matrix.
    
    Args:
        x_pixel (float): X coordinate in pixel space
        y_pixel (float): Y coordinate in pixel space
        homography_matrix (np.array): 3x3 homography matrix

    Returns:
        (float, float): World coordinates (X, Y)
    """
    pixel_point = np.array([x_pixel, y_pixel, 1])
    world_point = homography_matrix @ pixel_point
    world_point /= world_point[2]  # Normalize by the third (homogeneous) coordinate

    return world_point[0], world_point[1]