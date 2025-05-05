import os
import sys
import cv2
import yaml
import torch
import numpy as np
import time

# yolov5 디렉터리 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'yolov5'))

from models.common import AutoShape
from models.experimental import attempt_load

class YOLODetector:
    def __init__(
        self,
        weights_path: str = "yolov5n.pt",
        config_path: str = None
    ):
        # 강제로 CPU로 설정
        self.device = "cpu"

        # config.yaml 로드
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "models", "yolo", "config.yaml"
            )
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.conf_thres = cfg.get("conf_thres", 0.3)
        self.iou_thres  = cfg.get("iou_thres", 0.45)
        self.max_det    = cfg.get("max_det", 100)

        # 모델 로드 (CPU에만 올림)
        model = attempt_load(weights_path, device=self.device)
        self.model = AutoShape(model)

        # 내부 상태 변수
        self.last_frame = None
        self.last_detection = None
        self.detected_last_frame = False

    def detect(self, image: cv2.Mat):
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

        self.detected_last_frame = len(detections) > 0
        self.last_detection = detections[0] if detections else None
        self.last_frame = detections

        return detections



if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = YOLODetector()

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        dets = detector.detect(frame)
        for det in dets:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{det['confidence']:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("YOLOv5 Detection", frame)
        end_time = time.time()
        print(end_time-start_time)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
