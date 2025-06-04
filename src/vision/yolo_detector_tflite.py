import numpy as np
import cv2
import tensorflow as tf

class YOLODetector:
    def __init__(self, name: str, weights_path: str, input_size=(320, 320), threshold=0.25, nms_threshold=0.4):
        self.name = name
        self.input_size = input_size
        self.threshold = threshold
        self.nms_threshold = nms_threshold

        self.interpreter = tf.lite.Interpreter(model_path=weights_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_dtype = self.input_details[0]['dtype']

    def preprocess(self, image):
        self.orig_h, self.orig_w = image.shape[:2]
        img_resized = cv2.resize(image, self.input_size)

        if self.input_dtype == np.float32:
            input_data = img_resized.astype(np.float32) / 255.0
        elif self.input_dtype == np.uint8:
            input_data = img_resized.astype(np.uint8)
        else:
            raise ValueError(f"Unsupported input dtype: {self.input_dtype}")

        input_data = np.expand_dims(input_data, axis=0)
        return input_data

    def detect(self, image):
        input_data = self.preprocess(image)
        expected_shape = self.input_details[0]['shape']
        input_tensor = input_data.astype(self.input_details[0]['dtype'])

        if expected_shape[1] == 3 and input_tensor.shape[1] != 3:
            input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))

        input_tensor = np.reshape(input_tensor, expected_shape)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        output_data = [self.interpreter.get_tensor(t['index']).copy() for t in self.output_details]

        boxes, scores, classes = self.postprocess(output_data)
        detections = []
        for i in range(len(boxes)):
            detections.append({
                "bbox": boxes[i],
                "score": scores[i],
                "class_id": classes[i],
                "model": self.name
            })
        return detections

    def postprocess(self, output_data):
        detections = output_data[0]
        if detections.ndim == 3:
            detections = np.squeeze(detections, axis=0)

        boxes, scores, classes = [], [], []

        for det in detections:
            if det.shape[-1] == 85:
                x, y, w, h = det[0:4]
                obj_conf = det[4]
                cls_scores = det[5:]
                if np.any(np.isnan(cls_scores)) or np.any(cls_scores > 1.0):
                    continue
                cls_id = int(np.argmax(cls_scores))
                cls_conf = cls_scores[cls_id]
                conf = obj_conf * cls_conf
                if conf < self.threshold:
                    continue
                
                if max(x, y, w, h) <= 1.0:
                    x *= self.orig_w
                    y *= self.orig_h
                    w *= self.orig_w
                    h *= self.orig_h

                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)
            
            elif det.shape[-1] == 6:
                x1, y1, x2, y2, conf, cls_id = det
                if conf < self.threshold:
                    continue
                if 0 <= x1 <= 1 and 0 <= x2 <= 1:
                    x1 *= self.orig_w
                    x2 *= self.orig_w
                if 0 <= y1 <= 1 and 0 <= y2 <= 1:
                    y1 *= self.orig_h
                    y2 *= self.orig_h

                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                conf = float(conf)
                cls_id = int(cls_id)
            else:
                continue

            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            classes.append(cls_id)

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.threshold, self.nms_threshold)
        filtered_boxes, filtered_scores, filtered_classes = [], [], []
        for i in indices:
            i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            filtered_boxes.append(boxes[i])
            filtered_scores.append(scores[i])
            filtered_classes.append(classes[i])

        return filtered_boxes, filtered_scores, filtered_classes

    def draw_results(self, image, detections, color=(0, 255, 0)):
        img_copy = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            score = det["score"]
            cls = det["class_id"]
            label = f"{self.name}_{cls} ({score:.2f})"
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_copy, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img_copy

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] 웹캠을 열 수 없습니다.")
        exit()

    detector_coco = YOLODetector(
        name="coco",
        weights_path="C:/Users/user/OneDrive/Documents/VSCode/pi_AutoPark/yolov5/yolov5n-fp16.tflite",
        threshold=0.5
    )

    detector_custom = YOLODetector(
        name="custom",
        weights_path="C:/Users/user/OneDrive/Documents/VSCode/pi_AutoPark/yolov5/last4-fp16.tflite",
        threshold=0.5
    )

    print("[INFO] 웹캠 탐지 시작 (종료하려면 Q 키 누르세요)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results_coco = detector_coco.detect(frame)
        results_custom = detector_custom.detect(frame)

        frame_out = frame.copy()
        frame_out = detector_coco.draw_results(frame_out, results_coco, color=(255, 0, 0))  # 파랑
        frame_out = detector_custom.draw_results(frame_out, results_custom, color=(0, 255, 0))  # 초록

        cv2.imshow("YOLO TFLite Webcam Detection (float32/FP16)", frame_out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
