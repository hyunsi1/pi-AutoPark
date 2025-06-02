import cv2
import numpy as np
from itertools import combinations

def detect_parking_lines(image, debug=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=60, maxLineGap=10)

    line_image = image.copy()
    line_coords = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_coords.append(((x1, y1), (x2, y2)))
            if debug:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return line_coords, line_image

def is_rectangle(pts, tol=10):
    if len(pts) != 4:
        return False

    pts = sorted(pts, key=lambda x: (x[1], x[0]))  
    top = sorted(pts[:2], key=lambda x: x[0])  
    bottom = sorted(pts[2:], key=lambda x: x[0])  

    p1, p2, p3, p4 = top[0], top[1], bottom[1], bottom[0]

    def dist(a, b): return np.linalg.norm(np.array(a) - np.array(b))
    d1 = dist(p1, p3)
    d2 = dist(p2, p4)
    return abs(d1 - d2) < tol

def find_parking_slot_center(lines, image_shape):
    if not lines or len(lines) < 4:
        return None

    points = []
    for (x1, y1), (x2, y2) in lines:
        points.append((x1, y1))
        points.append((x2, y2))

    clustered = []
    for pt in points:
        matched = False
        for c in clustered:
            if np.linalg.norm(np.array(pt) - np.array(c)) < 20: 
                matched = True
                break
        if not matched:
            clustered.append(pt)

    if len(clustered) < 4:
        return None

    # 가능한 4점 조합에서 사각형인지 확인
    for combo in combinations(clustered, 4):
        if is_rectangle(combo):
            cx = int(np.mean([p[0] for p in combo]))
            cy = int(np.mean([p[1] for p in combo]))

            h, w = image_shape[:2]
            if 20 < cx < w - 20 and 20 < cy < h - 20:
                return (cx, cy)

    return None

def main():
    image_path = "C:\\Users\\user\OneDrive\\Documents\\VSCode\\pi_AutoPark\\data\\logs\\hough.png"

    # 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return

    lines, debug_image = detect_parking_lines(image, debug=True)

    center = find_parking_slot_center(lines, image.shape)
    if center:
        cx, cy = center
        cv2.circle(debug_image, (cx, cy), 6, (0, 0, 255), -1)
        print(f"주차 슬롯 중심: {center}")
    else:
        print("직선 부족: 슬롯 중심 계산 불가")

    cv2.imshow("Detected Parking Lines", debug_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
