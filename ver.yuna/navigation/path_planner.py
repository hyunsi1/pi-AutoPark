import numpy as np

class PathPlanner:
    """
    Vision-based PathPlanner for parking slot navigation.

    Responsibilities:
    - plan(): Decide direction based on bbox center offset and size
    - replan_around(): Adjust path to avoid obstacles using image-space logic
    - pid_step(): Step towards goal with proportional control
    """

    def __init__(self, num_segments: int = 5, img_width: int = 640, step_size: float = 0.2):
        self.num_segments = num_segments
        self.img_width = img_width
        self.step_size = step_size
        self.obstacle_clearance = 30  # 장애물 회피 여유 거리
        self.goal_threshold = 0.3  # 목표 도달 임계값

    def plan(self, bbox_center_x: int, bbox_height: int, frame_height: int) -> str:
        """
        Decide direction (left, right, forward, stop) based on the bbox center offset.
        """
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
        """
        장애물을 피하기 위한 경로를 다시 계산하고, 새로운 경로를 반환합니다.
        """
        start = np.array(current_pixel, dtype=float)
        goal = np.array(goal_pixel, dtype=float)
        obs = np.array(obstacle_pixel, dtype=float)

        # 목표 방향 벡터
        dir_vec = goal - start
        norm = np.linalg.norm(dir_vec)
        if norm == 0:
            return [tuple(start)]

        dir_unit = dir_vec / norm  # 단위 벡터

        # 왼쪽과 오른쪽으로 장애물 회피를 위한 수직 벡터 생성
        perp_left = np.array([-dir_unit[1], dir_unit[0]])
        perp_right = np.array([dir_unit[1], -dir_unit[0]])

        def side_clearance(side_vec):
            side_pt = obs + side_vec * clearance_pixel
            h, w = frame_shape
            if 0 <= side_pt[0] < w and 0 <= side_pt[1] < h:
                return np.linalg.norm(goal - side_pt), side_pt
            return float('inf'), side_pt

        dist_left, side_left = side_clearance(perp_left)
        dist_right, side_right = side_clearance(perp_right)

        # 가장 짧은 방향을 선택
        side_pt = side_left if dist_left < dist_right else side_right
        ahead = side_pt + dir_unit * clearance_pixel
        behind = side_pt - dir_unit * clearance_pixel

        return [tuple(start), tuple(behind), tuple(ahead), tuple(goal)]

    def pid_step(self, current_pos, goal_pos, step_size=None):
        """
        주어진 현재 위치에서 목표 위치까지 **PID 기반**으로 이동
        """
        step_size = step_size or self.step_size  # 기본 step_size 값 사용

        dx = goal_pos[0] - current_pos[0]
        dy = goal_pos[1] - current_pos[1]
        dist = np.hypot(dx, dy)

        if dist <= self.goal_threshold:
            return goal_pos  # 목표에 도달하면 목표 지점 반환

        # 목표 방향으로 비례 제어
        ratio = step_size / dist
        new_x = current_pos[0] + dx * ratio
        new_y = current_pos[1] + dy * ratio
        return (new_x, new_y)
