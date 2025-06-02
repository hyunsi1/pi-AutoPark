import cv2
import numpy as np
from itertools import combinations

def detect_parking_slot_by_contour(image, debug=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. 블러 처리 → 노이즈 제거
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. 이진화 → 선을 강조
    _, binary = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)

    # 3. Morphological closing → 선 끊김 연결
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 4. 윤곽선 검출
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate_centers = []

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, epsilon=0.02 * cv2.arcLength(cnt, True), closed=True)
        if len(approx) == 4 and cv2.contourArea(approx) > 1000:
            # 4각형이면 주차 슬롯 후보로 간주
            pts = [tuple(p[0]) for p in approx]
            cx = int(np.mean([p[0] for p in pts]))
            cy = int(np.mean([p[1] for p in pts]))
            candidate_centers.append(((cx, cy), pts))

            if debug:
                cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
                cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(image, f"({cx}, {cy})", (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    if candidate_centers:
        # 화면 아래에 가까운 사각형 우선 선택
        candidate_centers.sort(key=lambda x: x[0][1], reverse=True)
        return candidate_centers[0][0], image

    return None, image

'''
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

    # 정렬된 네 꼭지점
    pts = sorted(pts, key=lambda x: (x[1], x[0]))  # 우선 Y로, 다음 X로 정렬
    top = sorted(pts[:2], key=lambda x: x[0])   # top-left, top-right
    bottom = sorted(pts[2:], key=lambda x: x[0])  # bottom-left, bottom-right

    p1, p2, p3, p4 = top[0], top[1], bottom[1], bottom[0]

    # 거리 조건 확인 (대각선 길이 유사성)
    def dist(a, b): return np.linalg.norm(np.array(a) - np.array(b))
    d1 = dist(p1, p3)
    d2 = dist(p2, p4)
    return abs(d1 - d2) < tol

def find_parking_slot_center(lines, image_shape):
    if not lines or len(lines) < 4:
        return None

    # 모든 선분의 끝점 추출
    points = []
    for (x1, y1), (x2, y2) in lines:
        points.append((x1, y1))
        points.append((x2, y2))

    # 가까운 점들을 클러스터링 (중복 제거)
    clustered = []
    for pt in points:
        matched = False
        for c in clustered:
            if np.linalg.norm(np.array(pt) - np.array(c)) < 20:  # 20픽셀 이내면 같은 점
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

            # 이미지 경계 체크
            h, w = image_shape[:2]
            if 20 < cx < w - 20 and 20 < cy < h - 20:
                return (cx, cy)

    return None

def main():
    # 이미지 파일 경로 (본인의 이미지 경로로 수정)
    image_path = "C:\\Users\\user\OneDrive\\Documents\\VSCode\\pi_AutoPark\\data\\logs\\hough.png"

    # 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return

    # 1. 직선 검출
    lines, debug_image = detect_parking_lines(image, debug=True)

    # 2. 중심점 계산
    center = find_parking_slot_center(lines, image.shape)
    if center:
        cx, cy = center
        cv2.circle(debug_image, (cx, cy), 6, (0, 0, 255), -1)
        print(f"주차 슬롯 중심: {center}")
    else:
        print("직선 부족: 슬롯 중심 계산 불가")

    # 3. 결과 이미지 출력
    cv2.imshow("Detected Parking Lines", debug_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''

def main():
    image_path = "C:\\Users\\user\\OneDrive\\Documents\\VSCode\\pi_AutoPark\\data\\logs\\hough.png"
    image = cv2.imread(image_path)

    if image is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return

    center, debug_image = detect_parking_slot_by_contour(image, debug=True)

    if center:
        print(f"[검출 성공] 주차 슬롯 중심: {center}")
    else:
        print("[검출 실패] 사각형 형태의 슬롯을 찾지 못했습니다.")

    cv2.imshow("Contour-based Parking Slot Detection", debug_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()