import cv2, numpy as np

# ──────────────────────────────────────────────────
def find_left_reference(frame):
    """
    왼쪽 1/3 ROI에서
      ① 검은 세로 테이프  → priority 1
      ② 노란 발판 가장자리 → priority 2
    둘 중 하나의 (선분, 색타입)를 반환. 없으면 None
    """
    h, w = frame.shape[:2]
    roi = frame[:, : w//3]

    # ① 검은 선
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                            threshold=60, minLineLength=80, maxLineGap=10)
    if lines is not None:
        longest = max(lines, key=lambda l: np.linalg.norm(l[0][:2]-l[0][2:]))
        return longest[0], 'black'      # (x1,y1,x2,y2), type

    # ② 노란 발판
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (20,80,80), (35,255,255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        x,y,wc,hc = cv2.boundingRect(c)
        return (x,y,x,y+hc), 'yellow'   # 세로선처럼 취급
    return None, None
# ──────────────────────────────────────────────────
def steering_and_offset(line, ctrl, w_des_px=60):
    """
    line = (x1,y1,x2,y2) 기준선.
    1) 기울기로 steering 결정
    2) 차량-왼공간(픽셀) ↔ 목표 w_des_px 간의 차이(error)을 반영해
       steering ±5° 보정 값을 리턴
    """
    x1,y1,x2,y2 = line
    angle_deg = np.degrees(np.arctan2(y2-y1, x2-x1)) - 90    # 세로=0
    base_servo = ctrl.map_physical_angle_to_servo(angle_deg/2)

    # 왼쪽 여유 거리 측정
    cur_left = (x1+x2)/2
    err = w_des_px - cur_left           # +면 더 오른쪽으로 붙어야
    k = 0.1                             # 보정 계수(튜닝)
    physical_corr = k * err             # 픽셀→°(근사)
    servo = ctrl.map_physical_angle_to_servo((angle_deg+physical_corr)/2)
    return np.clip(servo, 30, 100)
# ──────────────────────────────────────────────────
def front_reference_gone(frame) -> bool:
    """
    전방(하단 20 %)에 ‘검은 가로 주차선’이 여전히 보이는가?
    """
    h, w = frame.shape[:2]
    roi = frame[int(h * 0.8):, :]              # 하단 20 %
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    # 얇고 긴 가로 컨투어가 있으면 아직 보이는 것
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        _, _, cw, ch = cv2.boundingRect(c)
        if cw > w * 0.6 and ch < h * 0.05:
            return False        # 주차선이 남아있다
    return True                 # 사라졌다 → 충분히 진입
