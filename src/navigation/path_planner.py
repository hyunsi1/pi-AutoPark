import numpy as np

class PathPlanner:
    """
    Vision-based PathPlanner for parking slot navigation.

    Responsibilities:
    - plan(): Decide direction based on bbox center offset and size
    - replan_around(): Adjust path to avoid obstacles using image-space logic
    - pid_step(): Step towards goal with proportional control
    """

    def __init__(self, num_segments: int = 5, img_width: int = 640, step_size: float = 0.1):
        self.num_segments = num_segments
        self.img_width = img_width
        self.step_size = step_size 

    def plan(self, bbox_center_x: int, bbox_height: int, frame_height: int) -> str:
        center_offset = bbox_center_x - self.img_width // 2

        if abs(center_offset) > 40:
            direction = "right" if center_offset > 0 else "left"
        else:
            direction = "forward"

        if bbox_height > 0.6 * frame_height:
            direction = "stop"

        return direction

    def replan_around(self,
                      current_pixel: tuple,
                      goal_pixel: tuple,
                      obstacle_pixel: tuple,
                      clearance_pixel: int,
                      frame_shape: tuple) -> list:
        start = np.array(current_pixel, dtype=float)
        goal = np.array(goal_pixel, dtype=float)
        obs = np.array(obstacle_pixel, dtype=float)

        dir_vec = goal - start
        norm = np.linalg.norm(dir_vec)
        if norm == 0:
            return [tuple(start)]

        dir_unit = dir_vec / norm
        perp_left = np.array([-dir_unit[1], dir_unit[0]])
        perp_right = np.array([ dir_unit[1], -dir_unit[0]])

        def side_clearance(side_vec):
            side_pt = obs + side_vec * clearance_pixel
            h, w = frame_shape
            if 0 <= side_pt[0] < w and 0 <= side_pt[1] < h:
                return np.linalg.norm(goal - side_pt), side_pt
            return float('inf'), side_pt

        dist_left,  side_left  = side_clearance(perp_left)
        dist_right, side_right = side_clearance(perp_right)

        side_pt = side_left if dist_left < dist_right else side_right
        ahead  = side_pt + dir_unit * clearance_pixel
        behind = side_pt - dir_unit * clearance_pixel

        return [tuple(start), tuple(behind), tuple(ahead), tuple(goal)]

    def pid_step(self, current_pos, goal_pos):
        dx = goal_pos[0] - current_pos[0]
        dy = goal_pos[1] - current_pos[1]
        dist = (dx**2 + dy**2) ** 0.5

        if dist == 0 or dist <= self.step_size:
            return goal_pos  # ⬅ 바로 도달

        ratio = self.step_size / dist
        new_x = current_pos[0] + dx * ratio
        new_y = current_pos[1] + dy * ratio
        return (new_x, new_y)
