import numpy as np

class PathPlanner:
    """
    PathPlanner for parking navigation.

    - plan(): straight-line waypoints between start and goal
    - replan_around(): forward-detour waypoints to avoid an obstacle with clearance
    """
    def __init__(self, num_segments: int = 5):
        # number of segments for straight-line interpolation
        self.num_segments = num_segments

    def plan(self, start: tuple, goal: tuple) -> list:
        """
        Generate straight-line waypoints from start to goal.

        Args:
            start: (x, y) start position in world coordinates
            goal:  (x, y) goal position in world coordinates

        Returns:
            list of (x, y) waypoints including start and goal
        """
        x0, y0 = start
        x1, y1 = goal
        waypoints = []
        for i in range(self.num_segments + 1):
            t = i / self.num_segments
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            waypoints.append((x, y))
        return waypoints

    def replan_around(self,
                        start: tuple,
                        goal: tuple,
                        obstacle: tuple,
                        clearance: float,
                        area_bounds: np.ndarray) -> list:
        start = np.array(start, dtype=float)
        goal = np.array(goal, dtype=float)
        obs = np.array(obstacle, dtype=float)

        dir_vec = goal - start
        norm = np.linalg.norm(dir_vec)
        if norm == 0:
            return [tuple(start)]
        dir_unit = dir_vec / norm
        perp_left  = np.array([-dir_unit[1], dir_unit[0]])
        perp_right = np.array([ dir_unit[1], -dir_unit[0]])

        def side_clearance_distance(side_vec):
            side_pt = obs + side_vec * clearance
            dists = np.linalg.norm(area_bounds - side_pt, axis=1)
            return np.min(dists), side_pt

        dist_left,  side_left  = side_clearance_distance(perp_left)
        dist_right, side_right = side_clearance_distance(perp_right)

        if dist_left > dist_right:
            side_pt = side_left
        else:
            side_pt = side_right

        ahead  = side_pt + dir_unit * clearance
        behind = side_pt - dir_unit * clearance

        return [tuple(start), tuple(behind), tuple(ahead), tuple(goal)]
