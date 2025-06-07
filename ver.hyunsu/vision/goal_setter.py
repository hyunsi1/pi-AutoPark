import cv2, numpy as np
from utility.transformations import pixel_to_world


class GoalSetter:
    """킥보드 gap 또는 빈 슬롯 왼쪽-위 꼭짓점을 픽셀/월드 좌표로 반환"""
    def __init__(self, min_gap=100, homography=None):
        self.min_gap = min_gap
        self.H = homography

    # ──────────────────────────────────────────────────────────
    def get_goal_point(self, img, detections):
        # ① 킥보드가 있을 때 – 가장 넓은 gap 중앙
        if detections:
            cx = [((x1+x2)//2, (y1+y2)//2) for x1,y1,x2,y2 in detections]
            cx.sort(key=lambda p: p[0])
            gaps, best = [], None
            for (x0,_), (x1,_) in zip(cx, cx[1:]):
                g = x1-x0
                gaps.append(g)
                if g > self.min_gap and (best is None or g > best[0]):
                    best = (g, (x0+x1)//2, (img.shape[0]//2))
            if best:
                _, gx, gy = best
                return self._p2w((gx, gy)), None, 'gap'

        # ② 킥보드가 없을 때 – 검은 테이프 사각형
        center, corners = self._detect_black_rect(img)
        if corners:
            # 네 변 길이 계산
            edges = [np.linalg.norm(np.subtract(corners[i], corners[(i+1)%4]))
                     for i in range(4)]
            idx = np.argsort(edges)[:2]          # 짧은 변 두 개
            # 더 왼쪽인 변
            left_idx = min(idx, key=lambda i: sum(pt[0] for pt in
                                                  (corners[i], corners[(i+1)%4])))
            a, b = corners[left_idx], corners[(left_idx+1)%4]
            goal_px = a if a[1] < b[1] else b    # 위쪽 꼭짓점
            return self._p2w(goal_px), corners, 'parking'
        return None, None, 'none'

    # ──────────────────────────────────────────────────────────
    def _detect_black_rect(self, img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(g, 90, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((45,45), np.uint8))
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect = max((c for c in cnts if len(cv2.approxPolyDP(c,0.04*cv2.arcLength(c,True),True))==4),
                   key=cv2.contourArea, default=None)
        if rect is None: return None, None
        corners = [tuple(pt[0]) for pt in cv2.approxPolyDP(rect, 0.04*cv2.arcLength(rect,True), True)]
        c = tuple(map(int, np.mean(corners, axis=0)))
        return c, corners

    # ──────────────────────────────────────────────────────────
    def _p2w(self, pt):
        return pixel_to_world(*pt, self.H) if self.H is not None else pt
