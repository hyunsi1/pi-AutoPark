import heapq
import math
import time
import numpy as np

class PathPlanner:
    def __init__(self, pixel_to_meter=0.1, speed_mps=0.2, angle_threshold_deg=20):
        self.pixel_to_meter = pixel_to_meter
        self.speed_mps = speed_mps
        self.angle_threshold = angle_threshold_deg
        self.obstacles = set()

    def heuristic(self, a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def set_obstacle(self, world_coord):
        self.obstacles.add(world_coord)

    def a_star(self, start, goal, grid_size=0.1):
        """A* 알고리즘으로 start에서 goal까지의 경로를 계산"""
        open_set = []
        heapq.heappush(open_set, (0 + self.heuristic(start, goal), 0, start, [start]))
        visited = set()

        while open_set:
            _, cost, current, path = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)

            if self.heuristic(current, goal) < grid_size:
                return path + [goal]

            for dx in [-grid_size, 0, grid_size]:
                for dy in [-grid_size, 0, grid_size]:
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (round(current[0] + dx, 2), round(current[1] + dy, 2))
                    if neighbor in visited or self._is_obstacle(neighbor):
                        continue
                    new_cost = cost + self.heuristic(current, neighbor)
                    priority = new_cost + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (priority, new_cost, neighbor, path + [neighbor]))

        return []  # 경로를 찾을 수 없음

    def _is_obstacle(self, point):
        for obs in self.obstacles:
            if self.heuristic(point, obs) < 0.2:
                return True
        return False

    def plan(self, start, goal):
        raw_path = self.a_star(start, goal)
        waypoints = []

        for i in range(1, len(raw_path)):
            prev = raw_path[i - 1]
            curr = raw_path[i]
            dx = curr[0] - prev[0]
            dy = curr[1] - prev[1]
            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)
            distance = math.hypot(dx, dy)

            waypoints.append({
                "pos": curr,
                "angle": angle_deg,
                "distance": distance
            })

        return waypoints

    def navigate(self, controller, path, obstacle_detector=None):
        for wp in path:
            angle_deg = max(-45, min(45, wp["angle"]))
            servo_angle = controller.map_physical_angle_to_servo(angle_deg)
            controller.set_angle(servo_angle)
            controller.set_speed(30, reverse=False)

            travel_time = wp["distance"] / self.speed_mps
            start_time = time.time()

            while time.time() - start_time < travel_time:
                if obstacle_detector and obstacle_detector():
                    controller.stop()
                    print("[Obstacle] 장애물 감지! 경로 재계산 필요")
                    return False
                time.sleep(0.1)

            controller.stop()

        print("[Navigate] 목표 지점 도달 완료")
        return True

    def obstacle_detector(self, detections, danger_classes=(0,), danger_distance=1.5):
        for det in detections:
            if det["class_id"] in danger_classes and det.get("distance", float('inf')) < danger_distance:
                return True
        return False

    def annotate_path_with_angles(self, path):
        annotated = []
        for i in range(len(path) - 1):
            curr = path[i]
            nxt = path[i + 1]
            dx = nxt[0] - curr[0]
            dy = nxt[1] - curr[1]
            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)
            angle_deg = max(-45, min(45, angle_deg))
            annotated.append({"pos": curr, "angle": angle_deg})
        # 마지막 지점은 방향 0
        annotated.append({"pos": path[-1], "angle": 0})
        return annotated
