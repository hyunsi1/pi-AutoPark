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

def find_black_rect_and_distance(image, debug=False):
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

import cv2
import numpy as np

def find_left_reference(frame, threshold=60, min_line_len=50):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(bin_img, 50, 150, apertureSize=3)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=min_line_len, maxLineGap=10)
    if lines is None:
        return None, bin_img

    leftmost = None
    min_x = float('inf')
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x_avg = (x1 + x2) / 2
        if x_avg < min_x:
            min_x = x_avg
            leftmost = (x1, y1, x2, y2)

    return leftmost, bin_img

def front_reference_gone(frame, threshold=60, min_area=5000):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            return False  # 여전히 기준선 존재

    return True  # 기준선 사라짐

def main():
    folder = r"C:\Users\user\OneDrive\Documents\VSCode\pi_AutoPark\data"
    for fname in os.listdir(folder):
        if not fname.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
            continue
        path = os.path.join(folder, fname)
        img  = cv2.imread(path)
        if img is None:
            print(f"[{fname}] 로드 실패"); continue

        tl, corners, dist, vis = find_black_rect_and_distance(img, debug=True)
        print(f"--- {fname} ---")
        if tl:
            print("Top-left:", tl)
            print("Corners :", corners)
            print(f"Distance: {dist:.2f} m")
        else:
            print("사각형 검출 실패")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
