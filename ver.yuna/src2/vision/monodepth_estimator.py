import cv2
import numpy as np
import yaml
from camera.calibration import load_camera_parameters

class MonoDepthEstimator:
    """
    킥보드 실측 높이와 카메라 파라미터로 단안 거리(depth) 추정
    """
    def __init__(self, config_dict=None):
        if config_dict is None:
            raise ValueError("MonoDepthEstimator requires a config_dict")

        self.object_height = float(config_dict.get('object_height', 0.3))  # m
        self.camera_height = float(config_dict.get('camera_height', 0.3))  # m
        self.tilt_angle = np.deg2rad(float(config_dict.get('tilt_angle', 0.0)))  # radians

        cam_mtx, _ = load_camera_parameters()
        self.fy = cam_mtx[1, 1]
        self.cy = cam_mtx[1, 2]

        self.method = config_dict.get('method', 'ratio')  # 'ratio', 'ground_plane', or 'hybrid'
        self.hybrid_w = float(config_dict.get('hybrid_weight', 0.5))

    def _ratio_depth(self, pixel_h: float) -> float:
        return self.object_height * self.fy / pixel_h

    def _ground_plane_depth(self, y2: float) -> float:
        v = (y2 - self.cy) / self.fy
        angle = self.tilt_angle + np.arctan(v)
        return self.camera_height / np.tan(angle)

    def estimate_depth(self, bbox: tuple) -> float:
        x1, y1, x2, y2 = bbox
        pixel_h = y2 - y1
        if pixel_h <= 0:
            return None

        d_ratio = self._ratio_depth(pixel_h)

        if self.method == 'ground_plane':
            return float(self._ground_plane_depth(y2))
        if self.method == 'hybrid':
            d_gp = self._ground_plane_depth(y2)
            return float(self.hybrid_w * d_ratio + (1 - self.hybrid_w) * d_gp)

        return float(d_ratio)

    def annotate_detections(self, image: np.ndarray, detections: list) -> np.ndarray:
        for det in detections:
            depth = self.estimate_depth(det['bbox'])
            det['depth'] = depth
            x1, y1, x2, y2 = det['bbox']
            label = f"{depth:.2f}m"
            cv2.putText(image, label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return image

    def estimate_current_position_from_y2(self, y2: float) -> tuple:
        v = (y2 - self.cy) / self.fy
        alpha = self.tilt_angle + np.arctan(v)
        distance = self.camera_height / np.tan(alpha)
        return (0, distance)
