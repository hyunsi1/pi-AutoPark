import cv2
import numpy as np
from utility.transformations import p2w  # pixel → world 변환 함수

from vision.find_black_rect import find_black_rect 

class GoalSetter:
    def __init__(self, slot_gap=0.1):
        self.slot_gap = slot_gap

    def get_goal_point(self, frame, boxes):
        """
        ▸ 검정 네모(주차장) 감지되면 → 왼쪽 위 꼭짓점을 목표로 (mode='parking')
        ▸ 아니면 킥보드 gap 탐색 (mode='gap')
        """
        # ① 검정 네모 우선 탐색
        goal, corners, _ = find_black_rect(frame)
        if goal:
            goal_world = p2w(goal[0], goal[1])[:2]  # x, y만
            return goal_world, corners, 'parking'

        # ② 킥보드 gap 탐색
        if len(boxes) < 2:
            return None, None, 'gap'  # gap 찾을 수 없음

        # x 중심좌표 기준으로 정렬
        centers = [((x1+x2)//2, (y1+y2)//2) for x1, y1, x2, y2 in boxes]
        centers.sort()

        # 간격 분석
        for (c1x, _), (c2x, _) in zip(centers, centers[1:]):
            gap = c2x - c1x
            if gap > 100:   # 픽셀 단위 gap 최소치 (현장 튜닝 필요)
                gap_center = (c1x + c2x)//2
                goal_world = p2w(gap_center, frame.shape[0]//2)[:2]
                return goal_world, None, 'gap'

        # gap 못 찾음
        return None, None, 'gap'
