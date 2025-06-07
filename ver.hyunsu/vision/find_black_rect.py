import cv2
import numpy as np
import os

def find_black_rect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # 1. 커널 크기 늘려가며, 내부가 하얗게 될 때까지 반복 (최대 41, 51, 61까지도 가능!)
    kernel = np.ones((45, 45), np.uint8)  # 커널 크기를 크게!
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Closed Mask", closed)  # 내부가 꽉 찬 네모가 되면 OK
    cv2.waitKey(0)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_rect = None
    best_area = 0
    img_draw = image.copy()
    h, w = img_draw.shape[:2]
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, epsilon=0.04*cv2.arcLength(cnt, True), closed=True)
        if len(approx) == 4 and 1000 < cv2.contourArea(approx) < w*h*0.95:
            area = cv2.contourArea(approx)
            if area > best_area:
                best_area = area
                best_rect = approx
    if best_rect is not None:
        corners = [tuple(pt[0]) for pt in best_rect]
        cx = int(np.mean([p[0] for p in corners]))
        cy = int(np.mean([p[1] for p in corners]))
        cv2.drawContours(img_draw, [best_rect], -1, (0,255,0), 5)
        for pt in corners:
            cv2.circle(img_draw, pt, 18, (0,0,255), -1)
        cv2.circle(img_draw, (cx,cy), 24, (255,0,0), -1)
        return (cx, cy), corners, img_draw
    return None, None, image

def main():
    folder = "C:/Users/82104/Desktop/kick"
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    for fname in image_files:
        image_path = os.path.join(folder, fname)
        image = cv2.imread(image_path)
        if image is None:
            print(f"[{fname}] 이미지를 불러올 수 없습니다: {image_path}")
            continue

        goal, corners, vis_img = find_black_rect(image)

        print(f"--- {fname} ---")
        if goal:
            print(f"검정 네모 중심: {goal}")
            print(f"꼭짓점: {corners}")
            cv2.imshow(f"결과 - {fname}", vis_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("검정 네모(테이프 사각형)를 찾지 못했습니다.")

if __name__ == "__main__":
    main()