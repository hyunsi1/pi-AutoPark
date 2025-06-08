import cv2
import numpy as np
import os

# --- 파라미터 설정 ---
REAL_RECT_WIDTH_M = 0.5    # 실제 테이프 사각형 가로 길이 (m)
FOV_DEG          = 60.0     # 카메라 수평 시야각 (°)

THRESHOLD        = 80       # 그레이스케일 임계값
CLOSE_KSIZE      = 45       # 모폴로지 closing 커널 크기 (px)

# 검출 대상 면적의 최소/최대 비율 (이미지 전체 대비)
MIN_AREA_RATIO   = 0.005    # 0.5% 이하 건너뜀
MAX_AREA_RATIO   = 0.9      # 90% 이상 건너뜀

def estimate_distance(real_w, px_w, img_w, fov_deg=FOV_DEG):
    fov   = np.radians(fov_deg)
    focal = img_w / (2 * np.tan(fov/2))
    return real_w * focal / (px_w + 1e-6)

def order_corners(pts):
    pts = sorted(pts, key=lambda p: (p[1], p[0]))
    top = sorted(pts[:2], key=lambda p: p[0])
    bot = sorted(pts[2:], key=lambda p: p[0])
    return [top[0], top[1], bot[1], bot[0]]


def find_black_rect(image, debug=False):
    h, w      = image.shape[:2]
    img_area  = h * w
    min_area  = MIN_AREA_RATIO * img_area
    max_area  = MAX_AREA_RATIO * img_area

    # 1) 전처리
    gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    kernel  = np.ones((CLOSE_KSIZE, CLOSE_KSIZE), np.uint8)
    closed  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 2) 컨투어 검출 & 면적 필터링
    cnts, _    = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_cnts = []
    for c in cnts:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            valid_cnts.append((c, area))
    valid_cnts.sort(key=lambda x: x[1], reverse=True)

    best_poly = None

    # 3) approxPolyDP 시도 (다중 epsilon)
    for cnt, _ in valid_cnts:
        for eps in (0.02, 0.04, 0.06):
            approx = cv2.approxPolyDP(cnt, eps * cv2.arcLength(cnt, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                # top-left 계산
                pts_ord = order_corners([tuple(pt[0]) for pt in approx])
                if pts_ord[0] == (0, 0):
                    continue  # (0,0)이면 다음 후보
                best_poly = approx
                break
        if best_poly is not None:
            break

    # 4) fallback: minAreaRect on 가장 큰 valid contour
    if best_poly is None and valid_cnts:
        cnt0 = valid_cnts[0][0]
        rect = cv2.minAreaRect(cnt0)
        box  = cv2.boxPoints(rect)
        pts_box = [tuple(pt) for pt in np.intp(box)]
        pts_ord = order_corners(pts_box)
        # fallback 면적·크기 체크 + (0,0) 제외
        box_area = rect[1][0] * rect[1][1]
        if (min_area < box_area < max_area and
            rect[1][0] < w*0.9 and rect[1][1] < h*0.9 and
            pts_ord[0] != (0, 0)):
            best_poly = np.array(pts_box, dtype=np.int32).reshape(-1,1,2)

    vis = image.copy()
    if best_poly is None:
        if debug:
            cv2.imshow("Mask", closed)
        return None, None, None, vis  # 검출 실패

    # 5) 코너 정렬 및 거리 계산
    corners = [tuple(pt[0]) for pt in best_poly]
    ordered = order_corners(corners)
    tl, tr = ordered[0], ordered[1]
    px_w    = np.hypot(tr[0]-tl[0], tr[1]-tl[1])
    dist    = estimate_distance(REAL_RECT_WIDTH_M, px_w, w)

    # 시각화
    cv2.drawContours(vis, [np.array(ordered)], -1, (0,255,0), 3)
    for p in ordered:
        cv2.circle(vis, p, 8, (0,0,255), -1)
    cv2.circle(vis, tl, 12, (255,0,0), -1)
    cv2.putText(vis, f"Dist: {dist:.2f} m", (10, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
    if debug:
        cv2.imshow("Detection Debug", vis)

    return tl, ordered, dist, vis




def main():
    folder = "C:/Users/82104/Desktop/kick"
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    for fname in image_files:
        image_path = os.path.join(folder, fname)
        image = cv2.imread(image_path)
        if image is None:
            print(f"[{fname}] 이미지를 불러올 수 없습니다: {image_path}")
            continue

        # 검정 네모 검출 및 시각화
        goal_world, corners, dist, vis_img = find_black_rect(image)

        print(f"--- {fname} ---")
        if goal_world:
            print(f"검정 네모 좌상단 좌표 (픽셀): {goal_world}")
            print(f"꼭짓점 (픽셀): {corners}")
            print(f"추정 거리: {dist:.2f} m")
        else:
            print("검정 네모(테이프 사각형)를 찾지 못했습니다.")

        # 검출 여부와 상관없이 시각화는 항상!
        cv2.imshow(f"결과 - {fname}", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        # Detection
        goal_world, corners, dist, vis_img = find_black_rect(frame)

        if goal_world:
            print(f"Top-left corner: {goal_world}")
            print(f"Corners: {corners}")
            print(f"Distance: {dist:.2f} m")
        else:
            print("No rectangle detected.")

        cv2.imshow("Detection Result", vis_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_webcam()
