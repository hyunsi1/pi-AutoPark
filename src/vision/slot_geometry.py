import cv2
import numpy as np

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

def find_parking_slot_center(lines, image_shape):
    if not lines or len(lines) < 2:
        return None

    xs, ys = [], []
    for (x1, y1), (x2, y2) in lines:
        xs += [x1, x2]
        ys += [y1, y2]

    cx = int(np.mean(xs))
    cy = int(np.mean(ys))
    return (cx, cy)

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

if __name__ == "__main__":
    main()