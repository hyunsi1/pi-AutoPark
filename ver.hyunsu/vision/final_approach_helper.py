import cv2, numpy as np
import os

# ──────────────────────────────────────────────────
def find_left_reference(frame, min_length=500, slope_thresh=0.8, debug=False):
    """
    왼쪽 1/3 ROI에서
      ① 검은 세로 테이프 → priority 1
      ② 노란 발판 가장자리 → priority 2
    둘 중 하나의 (선분, 색타입, slope)를 반환. 없으면 (None, None, None)
    """
    h, w = frame.shape[:2]
    roi = frame[:, : w//3]

    # ① 검은 세로 테이프 검출 (detect_black_line_center_roi 방식)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        length = cv2.arcLength(cnt, True)

        if length < min_length or area < 1000:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w_rect, h_rect), angle = rect

        if w_rect > h_rect:
            dx = w_rect / 2 * np.cos(np.deg2rad(angle))
            dy = w_rect / 2 * np.sin(np.deg2rad(angle))
        else:
            angle += 90
            dx = h_rect / 2 * np.cos(np.deg2rad(angle))
            dy = h_rect / 2 * np.sin(np.deg2rad(angle))

        pt1 = (int(cx - dx), int(cy - dy))
        pt2 = (int(cx + dx), int(cy + dy))

        line_length = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])

        if line_length < min_length or area < 1000:
            continue

        if pt2[0] - pt1[0] == 0:
            slope = float('inf')
        else:
            slope = abs((pt2[1] - pt1[1]) / (pt2[0] - pt1[0]))

        if slope >= 0.8:
            # ROI 기준 좌표와 기울기 함께 반환
            return slope

    # ② 노란 발판 가장자리 찾기 (기존 그대로 유지, slope는 None 반환)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (25, 80, 100), (35, 255, 255))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    if debug:
        cv2.imshow("01 - ROI", cv2.resize(roi, (0,0), fx=0.5, fy=0.5))
        cv2.imshow("02 - Yellow Mask", cv2.resize(mask, (0,0), fx=0.5, fy=0.5))
        cv2.imshow("03 - Morph Close", cv2.resize(mask_closed, (0,0), fx=0.5, fy=0.5))
        cv2.waitKey(0)

    # 2. 컨투어 찾기
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("노란색 컨투어 없음!")
        return None

    # 3. 가장 큰 컨투어 사용
    c = max(contours, key=cv2.contourArea)

    # 4. x 좌표 기준으로 오른쪽 경계점들만 추출
    contour_pts = c.reshape(-1, 2)
    x_max = np.max(contour_pts[:, 0])
    right_edge_pts = contour_pts[contour_pts[:, 0] >= x_max - 50]

    # 전체 y 범위 계산
    y_min = np.min(right_edge_pts[:, 1])
    y_max = np.max(right_edge_pts[:, 1])

    # 전체 y범위를 3등분
    y1 = y_min + (y_max - y_min) / 3
    y2 = y_min + (y_max - y_min) * 2 / 3

    # 중간 1/3만 선택
    right_edge_pts = right_edge_pts[
        (right_edge_pts[:, 1] >= y1) & (right_edge_pts[:, 1] <= y2)
    ]

    if debug:
        debug_edge = cv2.cvtColor(mask_closed, cv2.COLOR_GRAY2BGR)
        for p in right_edge_pts:
            cv2.circle(debug_edge, tuple(p), 2, (0, 0, 255), -1)
        cv2.imshow("04 - Right Edge Points", cv2.resize(debug_edge, (0,0), fx=0.5, fy=0.5))
        cv2.waitKey(0)

    if len(right_edge_pts) < 2:
        print("오른쪽 경계선 점이 너무 적음!")
        return None

    # 5. 직선 근사
    [vx, vy, x0, y0] = cv2.fitLine(right_edge_pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = vx[0], vy[0], x0[0], y0[0]

    # 6. 직선 위 두 점 생성 (ROI 경계까지 연장)
    left_y = int(y0 - (x0) * (vy / vx))
    right_y = int(y0 + (w//2 - x0) * (vy / vx))
    pt1 = (0, left_y)
    pt2 = (w//2, right_y)

    # 7. 기울기와 길이 계산
    dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
    slope = abs(dy / dx) if dx != 0 else float('inf')
    length = np.hypot(dx, dy)

    if debug:
        debug_final = roi.copy()
        cv2.line(debug_final, pt1, pt2, (0, 0, 255), 2)
        cv2.imshow("05 - Final Line", cv2.resize(debug_final, (0,0), fx=0.5, fy=0.5))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if slope >= slope_thresh and length >= min_length:
        return  slope

    return None


# ──────────────────────────────────────────────────
def steering(slope):
    """
    slope를 바로 받아서 steering 각도로 변환한다.
    """
    angle_deg = np.degrees(np.arctan(slope)) - 25

    return np.clip(angle_deg, 30, 100)
# ──────────────────────────────────────────────────
def front_reference_gone(frame) -> bool:

    h, w = frame.shape[:2]           # 하단 20 %
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)

    # 긴 가로 컨투어가 있으면 아직 보이는 것
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        _, _, cw, ch = cv2.boundingRect(c)
        if cw > w * 0.5 and ch > 10:
            return False        # 주차선이 남아있다
    return True                 # 사라졌다 → 충분히 진입



def process(folder=r"C:\Users\82104\Desktop\final"):
    for f in os.listdir(folder):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(folder, f)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[{f}] 이미지를 불러올 수 없습니다.")
                continue

            line, color_type, slope = find_left_reference(img, min_length=500, slope_thresh=0.8, debug=True)

            if line:
                x1, y1, x2, y2 = line
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                print(f"[{f}] {color_type} 검출: pt1=({x1},{y1}), pt2=({x2},{y2}), slope={slope:.2f}")
            else:
                print(f"[{f}] 검출 실패!")

            cv2.imshow("Result", cv2.resize(img, (0,0), fx=0.5, fy=0.5))
            cv2.waitKey(0)
    cv2.destroyAllWindows()



def test_all_images(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(folder_path, file_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[{file_name}] 이미지 읽기 실패!")
                continue

            gone = front_reference_gone(img)
            status = "⚠️ 주차선 남아있음" if not gone else "✅ 주차선 사라짐"
            print(f"[{file_name}] {status}")


            small_roi = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            small_mask = cv2.resize(
                cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 40, 255, cv2.THRESH_BINARY_INV)[1],
                (0, 0), fx=0.5, fy=0.5)

            cv2.imshow("ROI", small_roi)
            cv2.imshow("Mask", small_mask)
            cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    folder = r"C:\Users\82104\Desktop\forward"
    test_all_images(folder)
