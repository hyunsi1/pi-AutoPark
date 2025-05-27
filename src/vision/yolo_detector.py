import os
import sys
import cv2
import yaml
import torch
import time
import numpy as np

# YOLOv5 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'yolov5'))

from models.common import AutoShape
from models.experimental import attempt_load

class YOLODetector:
    def __init__(self, weights_path: str, config_path: str = None):
        self.device = "cpu"

        # config.yaml 로드 (옵션)
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "models", "yolo", "config.yaml"
            )
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            self.conf_thres = cfg.get("conf_thres", 0.3)
            self.iou_thres  = cfg.get("iou_thres", 0.45)
            self.max_det    = cfg.get("max_det", 100)
        else:
            self.conf_thres = 0.3
            self.iou_thres = 0.45
            self.max_det = 100

        model = attempt_load(weights_path, device=self.device)
        self.model = AutoShape(model)

    def detect(self, image: np.ndarray):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb, size=640)

        detections = []
        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            if conf < self.conf_thres:
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": float(conf),
                "class_id": int(cls)
            })

        return detections


if __name__ == "__main__":
    # 기본 COCO 모델과 커스텀 모델 경로
    coco_model_path = "yolov5n.pt"
    custom_model_path = "C:/Users/user/OneDrive/Documents/VSCode/pi_AutoPark/yolov5/runs/train/scooter_detector1/weights/best.pt"

    # 두 모델 초기화
    coco_detector = YOLODetector(coco_model_path)
    custom_detector = YOLODetector(custom_model_path)

    cap = cv2.VideoCapture(0)

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # 두 모델로 탐지 실행
        detections_coco = coco_detector.detect(frame)
        detections_custom = custom_detector.detect(frame)

        # COCO 모델 결과 (파란색)
        for det in detections_coco:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"COCO:{det['confidence']:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # 커스텀 모델 결과 (초록색)
        for det in detections_custom:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"SCOOTER:{det['confidence']:.2f}", (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 결과 표시
        cv2.imshow("YOLOv5 Dual Detection", frame)
        print(f"Frame Time: {time.time() - start_time:.3f} sec")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
