import os
import cv2
import time
import yaml
import torch
import numpy as np
from camera.calibration import load_camera_parameters

# MonoDepthEstimator Class
class MonoDepthEstimator:
    """
    킥보드 실측 높이와 카메라 파라미터로 단안 거리(depth) 추정
    """
    def __init__(self, config_path: str = None):
        # config 로드
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'config.yaml')
        
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        md_cfg = cfg.get('monodepth', {})
        
        # 카메라 파라미터 로드
        self.object_height = float(md_cfg.get('object_height', 1.0))  # 객체 높이 (m)
        self.camera_height = float(md_cfg.get('camera_height', 1.2))  # 카메라 높이 (m)
        self.tilt_angle = np.deg2rad(float(md_cfg.get('tilt_angle', 0.0)))  # 카메라 기울기 각도 (rad)
        
        cam_mtx, _ = load_camera_parameters()
        self.fy = cam_mtx[1,1]  # 카메라의 focal length (y-axis)
        self.cy = cam_mtx[1,2]  # 카메라의 principal point (y-axis)

        self.method = md_cfg.get('method', 'ratio')  # depth 추정 방법 (ratio/ground_plane/hybrid)
        self.hybrid_w = float(md_cfg.get('hybrid_weight', 0.5))  # hybrid 방식 시 가중치

    def _ratio_depth(self, pixel_h: float) -> float:
        """ratio 방식으로 depth 계산"""
        return self.object_height * self.fy / pixel_h

    def _ground_plane_depth(self, y2: float) -> float:
        """ground_plane 방식으로 depth 계산"""
        v = (y2 - self.cy) / self.fy
        angle = self.tilt_angle + np.arctan(v)
        return self.camera_height / np.tan(angle)

    def estimate_depth(self, bbox: tuple) -> float:
        """
        Depth 추정 함수
        - ratio: 픽셀 높이를 사용한 비례식으로 거리 추정
        - ground_plane: 카메라 기울기를 고려한 거리 추정
        - hybrid: ratio와 ground_plane 방식을 결합한 방식
        """
        x1, y1, x2, y2 = bbox
        pixel_h = y2 - y1
        if pixel_h <= 0:
            return None  # 유효하지 않은 bbox일 경우 None 반환

        # ratio 방식으로 거리 추정
        d_ratio = self._ratio_depth(pixel_h)

        # 방법에 따라 다르게 처리
        if self.method == 'ground_plane':
            return float(self._ground_plane_depth(y2))
        if self.method == 'hybrid':
            # ratio와 ground_plane 방식의 가중합
            d_gp = self._ground_plane_depth(y2)
            return float(self.hybrid_w * d_ratio + (1 - self.hybrid_w) * d_gp)
        
        # 기본은 ratio 방식
        return float(d_ratio)

    def annotate_detections(self, image: np.ndarray, detections: list) -> np.ndarray:
        """
        객체 탐지 결과에 depth 추가하고 이미지에 표시
        """
        for det in detections:
            depth = self.estimate_depth(det['bbox'])
            det['depth'] = depth  # 탐지된 객체에 depth 값을 추가
            x1, y1, x2, y2 = det['bbox']
            label = f"{depth:.2f}m"
            # 이미지에 depth 값 표시
            cv2.putText(image, label, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return image

    def estimate_current_position_from_y2(self, y2: float) -> tuple:
        """
        y2 좌표(픽셀)로부터 차량의 현재 위치 (x, y)를 추정

        Args:
            y2 (float): bbox 하단 또는 슬롯 중심의 y 픽셀 좌표

        Returns:
            tuple: (x, y) 카메라 기준의 현재 위치
        """
        # 이미지 중심 기준 위치 계산
        v = (y2 - self.cy) / self.fy
        alpha = self.tilt_angle + np.arctan(v)
        
        # 거리(depth) 계산 (삼각측량)
        distance = self.camera_height / np.tan(alpha)

        # 정면 기준으로 현재 위치 (x는 0으로 간주)
        x = 0
        y = distance

        return (x, y)  # x, y 위치 반환
