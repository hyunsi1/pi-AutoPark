# vision/tflite_yolo_detector.py
import cv2
import numpy as np
import tensorflow as tf
import sys

try:
    import tflite_runtime.interpreter as tflite
    USE_TFLITE = True
except ImportError:
    USE_TFLITE = False

class TFLiteYOLODetector:
    def __init__(self, model_path="autopark/yolov5_weight/yolov5n.tflite", conf_threshold=0.3, input_size=640):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.conf_threshold = conf_threshold
        self.input_size = input_size

    def detect(self, frame):
        img_resized = cv2.resize(frame, (self.input_size, self.input_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        expected_shape = self.input_details[0]['shape']
        if expected_shape[1] == 3:  # NCHW
            img = img_rgb.transpose(2, 0, 1)  # (3, H, W)
            input_data = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
        else:  # NHWC
            input_data = np.expand_dims(img_rgb, axis=0).astype(np.float32) / 255.0

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        detections = []
        for det in output_data:
            if det[4] < self.conf_threshold:
                continue
            x_center, y_center, width, height, conf = det[:5]
            x1 = int((x_center - width / 2) * frame.shape[1])
            y1 = int((y_center - height / 2) * frame.shape[0])
            x2 = int((x_center + width / 2) * frame.shape[1])
            y2 = int((y_center + height / 2) * frame.shape[0])
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": float(conf)
            })
        return detections


if __name__ == "__main__":
    print("TFLiteYOLODetector unit test start.")

    if not USE_TFLITE:
        print("TFLite not installed. Cannot run test.")
        sys.exit(1)

    detector = TFLiteYOLODetector(model_path="/home/pi/autopark/yolov5_weight/yolov5n.tflite")

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

            cv2.imshow("TFLiteYOLODetector Webcam Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
