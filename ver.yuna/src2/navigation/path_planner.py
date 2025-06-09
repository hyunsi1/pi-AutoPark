import heapq
import math
import time
import numpy as np

class PathPlanner:
    def __init__(self,
                 pixel_to_meter=0.1,
                 speed_mps=0.2,
                 angle_threshold_deg=20,
                 max_plan_time=0.5,
                 obstacle_radius=0.2):
        self.pixel_to_meter = pixel_to_meter
        self.speed_mps = speed_mps
        self.angle_threshold = angle_threshold_deg
        self.max_plan_time = max_plan_time    # s
        self.obstacle_radius = obstacle_radius
        self.obstacles = set()

    def heuristic(self, a, b):
        return math.hypot(b[0]-a[0], b[1]-a[1])

    def set_obstacle(self, world_coord):
        self.obstacles.add(world_coord)

    def _is_obstacle(self, pt):
        for obs in self.obstacles:
            if self.heuristic(pt, obs) < self.obstacle_radius:
                return True
        return False

    def _is_clear_line(self, start, goal, samples=10):
        """start→goal 직선 위에 장애물이 없으면 True"""
        for t in np.linspace(0, 1, samples):
            pt = (start[0] + (goal[0]-start[0])*t,
                  start[1] + (goal[1]-start[1])*t)
            if self._is_obstacle(pt):
                return False
        return True

    def _a_star(self, start, goal, grid_size, start_time, max_iters=10000):
        open_set = []
        heapq.heappush(open_set, (self.heuristic(start, goal), 0, start, [start]))
        closed = set()
        iters = 0

        # 4방향 이웃
        directions = [(grid_size,0),(-grid_size,0),(0,grid_size),(0,-grid_size)]

        while open_set:
            if time.time() - start_time > self.max_plan_time or iters > max_iters:
                raise TimeoutError("A* 타임아웃/최대반복 초과")
            iters += 1

            est_tot, cost, curr, path = heapq.heappop(open_set)
            if curr in closed:
                continue
            closed.add(curr)

            # goal 도달 판정
            if self.heuristic(curr, goal) < grid_size:
                return path + [goal]

            # 이웃 탐색
            for dx, dy in directions:
                nbr = (round(curr[0]+dx,2), round(curr[1]+dy,2))
                if nbr in closed or self._is_obstacle(nbr):
                    continue
                new_cost = cost + self.heuristic(curr, nbr)
                pri = new_cost + self.heuristic(nbr, goal)
                heapq.heappush(open_set, (pri, new_cost, nbr, path+[nbr]))

        raise TimeoutError("A* 경로 미발견")

    def plan(self, start, goal, grid_size=0.2):
        """
        start, goal: (x,y) in meters
        grid_size: 탐색 그리드 크기 (m) — 클수록 빠름, 정밀도 ↓
        """
        # 1) 직선 확인
        if self._is_clear_line(start, goal, samples=5):
            # 장애물 없으면 곧장
            raw = [start, goal]
            print("[PLANNER] direct path (no obstacles)")
        else:
            # 2) A* 수행
            t0 = time.time()
            try:
                raw = self._a_star(start, goal, grid_size, t0)
            except TimeoutError as e:
                print(f"[PLANNER] Warning: {e}, fallback to direct")
                raw = [start, goal]

        # 3) waypoints 생성
        waypoints = []
        for p0, p1 in zip(raw, raw[1:]):
            dx, dy = p1[0]-p0[0], p1[1]-p0[1]
            ang = math.degrees(math.atan2(dy, dx))
            dist = math.hypot(dx, dy)
            waypoints.append({
                "pos": p1,
                "angle": max(-45, min(45, ang)),
                "distance": dist
            })
        return waypoints

    def navigate(self, controller, path, obstacle_detector=None):
        for wp in path:
            servo = controller.map_physical_angle_to_servo(wp["angle"])
            controller.set_angle(servo)
            controller.set_speed(30, reverse=False)

            t_move = wp["distance"] / self.speed_mps
            t0 = time.time()
            while time.time() - t0 < t_move:
                if obstacle_detector and obstacle_detector():
                    controller.stop()
                    print("[NAVIGATE] obstacle! abort")
                    return False
                time.sleep(0.1)

            controller.stop()

        print("[NAVIGATE] done")
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
