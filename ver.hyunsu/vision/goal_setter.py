import cv2
import numpy as np
from utility.transformations import pixel_to_world  # pixel → world 변환 함수

from vision.find_black_rect import find_black_rect 

class GoalSetter:
    def __init__(self, min_gap=80, homography=None):
        self.min_gap = min_gap
        self.homography = homography

    def get_goal_point(self, frame, boxes):
        # ① 검정 네모 우선 탐색
        goal_world, corners, _, _ = find_black_rect(frame)
        if goal_world:
            return goal_world, corners, 'parking'

        # ② 킥보드 gap 탐색
        if len(boxes) < 2:
            return None, None, 'gap'

        centers = [((x1+x2)//2, (y1+y2)//2) for x1, y1, x2, y2 in boxes]
        centers.sort()
        for (c1x, _), (c2x, _) in zip(centers, centers[1:]):
            gap = c2x - c1x
            if gap > 100:
                gap_center = (c1x + c2x)//2
                goal_world = pixel_to_world(gap_center, frame.shape[0]//2, self.homography)[:2]
                return goal_world, None, 'gap'

        return None, None, 'gap'
