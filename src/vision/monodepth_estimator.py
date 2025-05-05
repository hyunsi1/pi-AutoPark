import os
import yaml
import numpy as np
import cv2
from camera.calibration import load_camera_parameters

class MonoDepthEstimator:
    """
    킥보드 실측 높이와 카메라 파라미터로 단안 거리(depth) 추정

    - ratio 방식: 픽셀 높이 대비 실제 높이 비례식
    - ground_plane 방식: 카메라 틸트 각도 고려한 지면 투영 방식
    """
    def __init__(
        self,
        config_path: str = None
    ):
        # config.yaml에서 monodepth 파라미터 로드
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'config', 'config.yaml'
            )
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        md_cfg = cfg.get('monodepth', {})
        # 킥보드 실제 높이(m)
        self.object_height = float(md_cfg.get('object_height', 1.0))
        # 카메라 지면 높이(m)
        self.camera_height = float(md_cfg.get('camera_height', 1.2))
        # 카메라 틸트 각도(deg), 아래쪽 기울기 양수
        self.tilt_angle = np.deg2rad(float(md_cfg.get('tilt_angle', 0.0)))
        # 카메라 내부 파라미터(focal length, principal point)
        cam_mtx, _ = load_camera_parameters()
        self.fy = cam_mtx[1,1]
        self.cy = cam_mtx[1,2]
        # 계산 방식: 'ratio' 또는 'ground_plane'
        self.method = md_cfg.get('method', 'ratio')

    def estimate_depth(self, bbox: tuple) -> float:
        """
        bbox: (x1, y1, x2, y2) 픽셀 좌표
        returns: depth in meters
        """
        x1, y1, x2, y2 = bbox
        pixel_h = y2 - y1
        if pixel_h <= 0:
            return None

        if self.method == 'ground_plane':
            # 바닥면 가정, 바운딩박스 하단 y2 픽셀로부터 지면 방향 각도 계산
            v = (y2 - self.cy) / self.fy
            angle = self.tilt_angle + np.arctan(v)
            depth = self.camera_height / np.tan(angle)
        else:
            # ratio 방식: depth ≈ f_y * H / h_pixels
            depth = self.object_height * self.fy / pixel_h

        return float(depth)

    def annotate_detections(self, image: np.ndarray, detections: list) -> np.ndarray:
        """
        detections 리스트에 depth 추가하고 이미지에 표시
        """
        for det in detections:
            depth = self.estimate_depth(det['bbox'])
            det['depth'] = depth
            x1, y1, x2, y2 = det['bbox']
            label = f"{depth:.2f}m"
            cv2.putText(image, label, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return image
