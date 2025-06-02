import numpy as np
import cv2
import tensorflow as tf

class YOLODetector:
    def __init__(self, model_paths: dict, input_size=(320, 320), threshold=0.25):
        self.input_size = input_size
        self.threshold = threshold
        self.models = {}

        for name, path in model_paths.items():
            interpreter = tf.lite.Interpreter(model_path=path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            self.models[name] = {
                "interpreter": interpreter,
                "input_details": input_details,
                "output_details": output_details
            }

    def preprocess(self, image):
        img_resized = cv2.resize(image, self.input_size)
        input_data = np.expand_dims(img_resized, axis=0)
        return input_data.astype(np.uint8)

    def postprocess(self, output_data):
        detections = output_data[0]
        if len(detections.shape) == 3 and detections.shape[1] == 1:
            detections = np.squeeze(detections, axis=1)  # [N, 1, 85] → [N, 85]

        boxes, scores, classes = [], [], []
        for det in detections:
            conf = float(np.array(det[4]).flatten()[0])  # ← 여기가 핵심
            if conf > self.threshold:
                x, y, w, h = [float(np.array(v).flatten()[0]) for v in det[0:4]]  # 각각 안전하게 flatten
                x1 = x - w / 2
                y1 = y - h / 2
                x2 = x + w / 2
                y2 = y + h / 2
                boxes.append([x1, y1, x2, y2])
                scores.append(conf)
                classes.append(int(np.argmax(np.array(det[5:]).flatten())))
        return boxes, scores, classes


    def detect(self, image):
        input_data = self.preprocess(image)
        result_dict = {}

        for name, model in self.models.items():
            interpreter = model["interpreter"]
            input_details = model["input_details"]
            output_details = model["output_details"]

            # 정확한 입력 shape 맞춰주기
            expected_shape = input_details[0]['shape']
            input_tensor = input_data.astype(input_details[0]['dtype'])

            # HWC → CHW 변환이 필요한 경우
            if expected_shape[1] == 3 and input_tensor.shape[1] != 3:
                input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))

            # shape 강제 적용
            input_tensor = np.reshape(input_tensor, expected_shape)

            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            output_data = [interpreter.get_tensor(t['index']) for t in output_details]
            boxes, scores, classes = self.postprocess(output_data)
            detections = []
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "score": float(scores[i]),
                    "class_id": int(classes[i]),
                    "model": name
                })

            result_dict[name] = detections

        return result_dict


    def draw_results(self, image, detections, color_map=None):
        h, w, _ = image.shape
        img_copy = image.copy()

        for model_name, (boxes, scores, classes) in detections.items():
            color = color_map[model_name] if color_map and model_name in color_map else (0, 255, 0)
            for i, box in enumerate(boxes):
                x1 = int(box[0] * w / self.input_size[0])
                y1 = int(box[1] * h / self.input_size[1])
                x2 = int(box[2] * w / self.input_size[0])
                y2 = int(box[3] * h / self.input_size[1])
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
                label = f"{model_name}_{classes[i]} ({scores[i]:.2f})"
                cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img_copy
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] 웹캠을 열 수 없습니다.")
        exit()

    detector = YOLODetector(
        model_paths={
            "custom": "C:/Users/user/OneDrive/Documents/VSCode/pi_AutoPark/yolov5/custom_model.tflite",
            "coco": "C:/Users/user/OneDrive/Documents/VSCode/pi_AutoPark/yolov5/coco_model.tflite"
        },
        input_size=(320, 320),
        threshold=0.25
    )

    print("[INFO] 웹캠 탐지 시작 (종료하려면 Q 키 누르세요)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 프레임을 읽지 못했습니다.")
            break

        results = detector.detect(frame)
        frame_out = frame.copy()

        # 각 모델의 탐지 결과를 프레임 위에 그리기
        color_map = {
            "custom": (0, 255, 0),
            "coco": (255, 0, 0)
        }

        for model_name, detections in results.items():
            color = color_map.get(model_name, (0, 255, 255))
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                score = det["score"]
                cls = det["class_id"]
                label = f"{model_name}_{cls} ({score:.2f})"
                cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_out, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow("YOLO TFLite Webcam Detection", frame_out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
