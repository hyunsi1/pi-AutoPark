import numpy as np

class PathPlanner:
    def __init__(self, num_segments=5, img_width=640, step_size=0.1):
        self.num_segments = num_segments
        self.img_width = img_width
        self.step_size = step_size

    def replan_around(self, start, goal, obs, clearance_pixel, frame_shape):
        start = np.array(start, dtype=float)
        goal = np.array(goal, dtype=float)
        obs = np.array(obs, dtype=float)

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

    def plan_full_path(self, start, goal, obstacle_list, frame_shape, clearance=40):
        path = [start]
        cur_pos = start

        for obs in obstacle_list:
            segment = self.replan_around(cur_pos, goal, obs, clearance, frame_shape)
            path.extend(segment[1:])
            cur_pos = segment[-1]

        path.append(goal)
        return path

    def pid_step(self, current_pos, goal_pos):
        dx = goal_pos[0] - current_pos[0]
        dy = goal_pos[1] - current_pos[1]
        dist = (dx**2 + dy**2) ** 0.5
        if dist == 0:
            return current_pos

        ratio = min(self.step_size / dist, 1.0)
        new_x = current_pos[0] + dx * ratio
        new_y = current_pos[1] + dy * ratio
        return (new_x, new_y)
    

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    
    # 1. PathPlanner 인스턴스 생성
    planner = PathPlanner(step_size=0.2)

    # 2. 시작점, 목표점, 장애물 좌표 (모두 [x, y])
    start = (50, 50)
    goal = (400, 400)
    obstacle_list = [
        (200, 180),    # 장애물1
        (320, 300)     # 장애물2
    ]
    frame_shape = (480, 640)  # (height, width)
    clearance = 50            # 장애물 회피 최소 거리(px)

    # 3. 경로 생성
    path = planner.plan_full_path(start, goal, obstacle_list, frame_shape, clearance)
    print("경로 포인트:")
    for pt in path:
        print(pt)

    # 4. 시각화
    plt.figure(figsize=(8, 6))
    # 전체 경로
    path_arr = np.array(path)
    plt.plot(path_arr[:,0], path_arr[:,1], '-o', label='Path')

    # 장애물 표시
    for idx, obs in enumerate(obstacle_list):
        plt.scatter(*obs, color='red', s=100, label=f'Obstacle {idx+1}' if idx==0 else None)

    # 시작/목표점 표시
    plt.scatter(*start, color='green', s=120, label='Start')
    plt.scatter(*goal, color='blue', s=120, label='Goal')

    plt.xlim(0, frame_shape[1])
    plt.ylim(0, frame_shape[0])
    plt.gca().invert_yaxis()  # 이미지 좌표계처럼 (좌상단 원점)
    plt.legend()
    plt.title("PathPlanner 테스트 경로 시각화")
    plt.show()