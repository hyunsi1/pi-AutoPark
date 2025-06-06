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