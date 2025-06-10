import cv2
import numpy as np
import math

def find_front_reference(frame, 
                         edge_thresh1=50, edge_thresh2=150,
                         hough_thresh=50, min_line_length=50, max_line_gap=10,
                         angle_thresh_deg=10):
    """
    Detect front reference line (approximately horizontal) in the frame.
    Returns ((x1,y1), (x2,y2)) or None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, edge_thresh1, edge_thresh2)
    lines = cv2.HoughLinesP(edges,
                            rho=1,
                            theta=np.pi/180,
                            threshold=hough_thresh,
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)
    if lines is None:
        return None, None

    # Find lines with angle near 0�� (horizontal)
    best_line = None
    best_score = float('inf')
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        angle = math.degrees(math.atan2(dy, dx))
        score = abs(angle)  # horizontal lines have angle ~0 or ~180
        if score > 90:
            score = abs(score - 180)
        if score < angle_thresh_deg and score < best_score:
            best_score = score
            best_line = ((x1, y1), (x2, y2))

    return best_line, best_score


def find_side_reference(frame,
                        edge_thresh1=50, edge_thresh2=150,
                        hough_thresh=50, min_line_length=50, max_line_gap=10,
                        angle_thresh_deg=10):
    """
    Detect side reference line (approximately vertical) in the frame.
    Returns ((x1,y1), (x2,y2)) or None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, edge_thresh1, edge_thresh2)
    lines = cv2.HoughLinesP(edges,
                            rho=1,
                            theta=np.pi/180,
                            threshold=hough_thresh,
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)
    if lines is None:
        return None, None

    # Find lines with angle near ��90�� (vertical)
    best_line = None
    best_score = float('inf')
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        angle = math.degrees(math.atan2(dy, dx))
        score = abs(abs(angle) - 90)
        if score < angle_thresh_deg and score < best_score:
            best_score = score
            best_line = ((x1, y1), (x2, y2))

    return best_line, best_score

def front_reference_gone(frame, **kwargs):
    """
    Returns True if front reference line is no longer detected.
    """
    line, _ = find_front_reference(frame, **kwargs)
    return line is None

def side_reference_gone(frame, **kwargs):
    """
    Returns True if side reference line is no longer detected.
    """
    line, _ = find_side_reference(frame, **kwargs)
    return line is None