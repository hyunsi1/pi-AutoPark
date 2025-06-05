import os
import sys
import cv2
import yaml
import torch
import time
import numpy as np
import psutil

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'yolov5'))

from models.common import AutoShape
from models.experimental import attempt_load

class YOLODetector:
    def __init__(self, name: str, weights_path: str, config_path: str = None):
        self.name = name
        self.device = "cpu"

        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "yolo", "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            self.conf_thres = cfg.get("conf_thres", 0.3)
        else:
            self.conf_thres = 0.3

        model = attempt_load(weights_path, device=self.device)
        self.model = AutoShape(model)
        print(f"[INFO] YOLO model '{self.name}' loaded from {weights_path}")

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
                "class_id": int(cls),
                "source": self.name
            })

        return detections

    def draw_results(self, frame, detections, color=(0, 255, 0)):
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            label = det.get("label", "object")
            conf = det.get("confidence", 0.0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

if __name__ == "__main__":
    coco_model_path = "yolov5n.pt"
    custom_model_path = "C:/Users/user/OneDrive/Documents/VSCode/pi_AutoPark/yolov5/best5.pt"

    coco_detector = YOLODetector(name="COCO", weights_path=coco_model_path)
    custom_detector = YOLODetector(name="SCOOTER", weights_path=custom_model_path)

    cap = cv2.VideoCapture(0)

    use_coco = True
    prev_detections_coco = []
    prev_detections_custom = []

    while True:
        #cpu_usage = psutil.cpu_percent(interval=None)
        #start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        if use_coco:
            prev_detections_coco = coco_detector.detect(frame)
        else:
            prev_detections_custom = custom_detector.detect(frame)
        use_coco = not use_coco 

        # COCO 결과 시각화 (파란색)
        for det in prev_detections_coco:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"COCO:{det['confidence']:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # 커스텀 결과 시각화 (초록색)
        for det in prev_detections_custom:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"SCOOTER:{det['confidence']:.2f}", (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        '''fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"CPU: {cpu_usage:.1f}%", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        '''
        cv2.imshow("YOLOv5 Dual Detector (Alternating)", frame)
        #print(f"Frame Time: {time.time() - start_time:.3f} sec")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
