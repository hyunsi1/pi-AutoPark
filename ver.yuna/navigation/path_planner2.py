# path_planner.py
import math
import heapq

class PathPlanner:
    def __init__(self, map_size=(100, 100), resolution=0.1):
        self.map_width, self.map_height = map_size
        self.resolution = resolution
        self.obstacles = set()

    def world_to_grid(self, pos):
        gx = int(pos[0] / self.resolution)
        gy = int(pos[1] / self.resolution)
        return gx, gy

    def grid_to_world(self, grid):
        wx = grid[0] * self.resolution
        wy = grid[1] * self.resolution
        return wx, wy

    def set_obstacle(self, world_pos):
        grid_pos = self.world_to_grid(world_pos)
        self.obstacles.add(grid_pos)

    def is_obstacle(self, grid_pos):
        return grid_pos in self.obstacles

    def get_neighbors(self, node):
        x, y = node
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                if not self.is_obstacle((nx, ny)):
                    neighbors.append((nx, ny))
        return neighbors

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def a_star(self, start_world, goal_world):
        start = self.world_to_grid(start_world)
        goal = self.world_to_grid(goal_world)

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return [self.grid_to_world(p) for p in self.reconstruct_path(came_from, current)]

            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # 경로 없음

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
