import cv2
import time
import logging
import os
from pathlib import Path
import torch
import yaml
import numpy as np
import psutil
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

class YOLODetector:
    def __init__(self, name: str, weights_path: str, config_path: str = None, device="cpu", fps=15):
        self.name = name
        self.device = device
        self.fps = fps  # Set desired FPS for detection
        self.conf_thres = 0.3  # Default confidence threshold
        self.last_frame_time = time.time()

        # Load config file if provided
        if config_path is None:
            config_path = "models/yolo/config.yaml"
        
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            self.conf_thres = cfg.get("conf_thres", self.conf_thres)
        
        # Load YOLO model
        self.model = self._load_model(weights_path)

    def _load_model(self, weights_path: str):
        model = torch.hub.load("ultralytics/yolov5", "custom", path=weights_path)
        return model  # No need to call .autoshape(), it's already integrated in YOLOv5

    def detect(self, frame: np.ndarray):
        """
        Perform object detection on a frame.

        Args:
            frame (np.ndarray): Input image frame.
        
        Returns:
            List of detections in the format:
                [{"bbox": (x1, y1, x2, y2), "confidence": float, "class_id": int}]
        """
        current_time = time.time()
        if current_time - self.last_frame_time >= 1 / self.fps:  # Limit FPS
            self.last_frame_time = current_time  # Update the last processed time
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            results = self.model(img_rgb, size=640)  # Run inference

            detections = []
            for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
                if conf < self.conf_thres:
                    continue  # Skip detections below the confidence threshold
                x1, y1, x2, y2 = map(int, xyxy)
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(conf),
                    "class_id": int(cls),
                    "source": self.name
                })
            return detections
        return []  # Return empty if frame was not processed due to FPS control

    def draw_results(self, frame, detections, color=(0, 255, 0)):
        """
        Draw bounding boxes on the frame.

        Args:
            frame (np.ndarray): Image frame.
            detections (list): List of detections with bounding boxes.
            color (tuple): Color for the bounding boxes.

        Returns:
            np.ndarray: Frame with bounding boxes drawn.
        """
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            label = det.get("label", "object")
            conf = det.get("confidence", 0.0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
    
logging.basicConfig(level=logging.INFO)

def main():
    # 모델 경로 설정 (현재 경로를 기준으로 설정)
    coco_model_path = "C:/Users/user/OneDrive/Documents/VSCode/pi_AutoPark/yolov5_weight/yolov5n.pt"
    custom_model_path = "C:/Users/user/OneDrive/Documents/VSCode/pi_AutoPark/yolov5_weight/best5.pt"
    
    coco_detector = YOLODetector(name="COCO", weights_path=str(coco_model_path))
    custom_detector = YOLODetector(name="SCOOTER", weights_path=str(custom_model_path))
    
    # 웹캡처 객체 초기화 (0번 웹캠 사용)
    cap = cv2.VideoCapture(0)
    
    # FPS 설정 (15 FPS로 제한)
    fps = 15
    prev_time = time.time()

    # 객체 탐지 모델 번갈아 사용 (COCO -> Custom)
    use_coco = True

    while True:
        # FPS에 맞춰 프레임 캡처 주기 조절
        current_time = time.time()
        if current_time - prev_time >= 1 / fps:
            prev_time = current_time
            
            # 웹캡에서 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to capture frame from webcam.")
                break

            # 선택된 모델로 객체 탐지
            if use_coco:
                detections = coco_detector.detect(frame)
                use_coco = False  # 다음엔 custom 모델 사용
            else:
                detections = custom_detector.detect(frame)
                use_coco = True  # 다시 coco 모델 사용

            # 탐지 결과 시각화
            frame = coco_detector.draw_results(frame, detections, color=(255, 0, 0) if use_coco else (0, 255, 0))
            
            # 탐지된 객체들 화면에 표시
            cv2.imshow("YOLOv5 Object Detection", frame)

            # 'q' 키를 눌러 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 캡처 종료
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()