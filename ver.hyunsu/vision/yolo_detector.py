import os
import sys
import cv2
import numpy as np
import torch
import pathlib

if sys.platform != "win32":
    pathlib.WindowsPath = pathlib.PosixPath

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'yolov5'))
from models.experimental import attempt_load
from utils.general import non_max_suppression


class YOLODetector:
    def __init__(self, weights_path="yolov5n.pt", conf_thres=0.3, iou_thres=0.45, input_size=320):
        self.device = "cpu"
        self.model = attempt_load(weights_path, device=self.device)
        self.model.eval()
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.input_size = input_size

    def detect(self, frame):
        img_resized = cv2.resize(frame, (self.input_size, self.input_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(img, augment=False)[0]
            pred = pred.cpu()

        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False, max_det=20)

        detections = []
        if pred[0] is not None:
            for *xyxy, conf, cls in pred[0]:
                x1, y1, x2, y2 = map(int, xyxy)
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(conf),
                    "class_id": int(cls)
                })
        return detections

if __name__ == "__main__":
    detector = YOLODetector(weights_path="yolov5n.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame.")
                break

            detections = detector.detect(frame)

            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{det['confidence']:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("YOLODetector Webcam Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("YOLODetector webcam test complete.")
