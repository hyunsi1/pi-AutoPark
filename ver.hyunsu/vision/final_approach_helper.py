import cv2, numpy as np
import os

# ──────────────────────────────────────────────────
def find_left_reference(frame, min_length=500, slope_thresh=1.0, debug=False):
    """
    왼쪽 1/3 ROI에서
      ① 노란 발판 가장자리 → priority 1
      ② 검은 세로 테이프 → priority 2
    둘 중 하나의 slope를 반환. 없으면 None
    """
    h, w = frame.shape[:2]
    roi = frame[:, : w//3]

    # ① 노란 발판 가장자리 찾기
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (25, 80, 100), (35, 255, 255))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 가장 큰 컨투어
        c = max(contours, key=cv2.contourArea)
        contour_pts = c.reshape(-1, 2)
        x_max = np.max(contour_pts[:, 0])
        right_edge_pts = contour_pts[contour_pts[:, 0] >= x_max - 50]

        y_min = np.min(right_edge_pts[:, 1])
        y_max = np.max(right_edge_pts[:, 1])
        y1 = y_min + (y_max - y_min) / 3
        y2 = y_min + (y_max - y_min) * 2 / 3
        right_edge_pts = right_edge_pts[
            (right_edge_pts[:, 1] >= y1) & (right_edge_pts[:, 1] <= y2)
        ]

        if len(right_edge_pts) >= 2:
            [vx, vy, x0, y0] = cv2.fitLine(right_edge_pts, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x0, y0 = vx[0], vy[0], x0[0], y0[0]

            left_y = int(y0 - (x0) * (vy / vx))
            right_y = int(y0 + (w//3 - x0) * (vy / vx))
            pt1 = (0, left_y)
            pt2 = (w//3, right_y)

            dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
            slope = abs(dy / dx) if dx != 0 else float('inf')
            length = np.hypot(dx, dy)

            if debug:
                debug_final = roi.copy()
                cv2.line(debug_final, pt1, pt2, (0, 0, 255), 2)
                cv2.imshow("Yellow Final Line", cv2.resize(debug_final, (0,0), fx=0.5, fy=0.5))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if slope >= slope_thresh and length >= min_length:
                return slope  # 노란 발판 가장자리 성공

    # ② 검은 세로 테이프 찾기
    # ROI: 왼쪽 아래 영역
    roi_black = frame[h//2:h, 0:w//2]

    # 그레이스케일 → 이진화
    gray = cv2.cvtColor(roi_black, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    # Morphology
    kernel = np.ones((5, 5), np.uint8)
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 엣지 검출
    edges = cv2.Canny(binary_clean, 50, 150, apertureSize=3)

    # Hough Line 검출
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=50)

    leftmost_x = w
    selected_slope = None

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # ROI에서 전체 이미지 좌표로 변환
            y1 += h // 2
            y2 += h // 2

            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length < 100:
                continue

            if x2 - x1 == 0:
                slope = float('inf')
            else:
                slope = (y2 - y1) / (x2 - x1)

            if abs(slope) < 1:
                continue

            if min(x1, x2) < leftmost_x:
                leftmost_x = min(x1, x2)
                selected_slope = slope

    return selected_slope  # 검은 세로 테이프 없으면 None


# ──────────────────────────────────────────────────
def steering(slope):
    """
    slope를 바로 받아서 steering 각도로 변환한다.
    slope가 음수일 경우 (180 - 25)를 더해 각도를 조정한다.
    """
    # 1. 기본 각도 계산 (일단 25를 뺀다)
    angle_deg = np.degrees(np.arctan(slope)) - 25

    # 2. slope가 음수일 경우 추가 조정
    if slope < 0:
        angle_deg = angle_deg + (180 - 25) # 또는 angle_deg + 155

    # 3. 최종 각도를 30에서 100 사이로 클리핑
    return np.clip(angle_deg, 30, 100)
# ──────────────────────────────────────────────────


def count_front_lines(frame) -> int:
    h, w = frame.shape[:2]
    
    # 가운데 50% 가로 ROI 설정
    x_start = w // 4
    x_end = x_start + w // 2
    roi = frame[:, x_start:x_end]

    # 그레이 변환 및 임계처리
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # 컨투어 검출
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 가로로 긴 컨투어 개수 세기
    count = 0
    for c in cnts:
        _, _, cw, ch = cv2.boundingRect(c)
        if cw > w * 0.5 * 0.5 and ch < 20:
            count += 1

    return count


def count_front_lines_with_draw(frame):
    h, w = frame.shape[:2]
    
    # 가운데 50% 가로 ROI 설정
    x_start = w // 4
    x_end = x_start + w // 2
    roi = frame[:, x_start:x_end]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw > w * 0.5 * 0.5 and ch < 20:
            count += 1
            # ROI 좌표를 원본 프레임 좌표로 변환해서 박스 그림
            cv2.rectangle(frame, (x + x_start, y), (x + x_start + cw, y + ch), (0, 255, 0), 2)
    
    # 가운데 50% ROI 영역 표시 (초록색 반투명)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_start, 0), (x_end, h), (0, 255, 0), -1)
    alpha = 0.2
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return count, frame
