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
    def __init__(self, config_path: str = None):
        # config 로드
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'config', 'config.yaml'
            )
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        md_cfg = cfg.get('monodepth', {})
        self.object_height = float(md_cfg.get('object_height', 1.0))
        self.camera_height = float(md_cfg.get('camera_height', 1.2))
        self.tilt_angle = np.deg2rad(float(md_cfg.get('tilt_angle', 0.0)))
        cam_mtx, _ = load_camera_parameters()
        self.fy = cam_mtx[1,1]
        self.cy = cam_mtx[1,2]
        self.method = md_cfg.get('method', 'ratio')
        # hybrid 시 ratio:ground weight
        self.hybrid_w = float(md_cfg.get('hybrid_weight', 0.5))

    def _ratio_depth(self, pixel_h: float) -> float:
        k = 100.0  # 조정 가능한 상수
        if pixel_h <= 0:
            return 999.0  # 너무 작으면 무한 거리
        return k / pixel_h

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
            d_gp = self._ground_plane_depth(y2)
            print(f"[DepthEstimation] bbox={bbox}, pixel_h={pixel_h}, d_gp={d_gp:.2f}")
            return float(d_gp)

        if self.method == 'hybrid':
            d_gp = self._ground_plane_depth(y2)
            depth = self.hybrid_w * d_ratio + (1 - self.hybrid_w) * d_gp
            print(f"[DepthEstimation] bbox={bbox}, pixel_h={pixel_h}, d_ratio={d_ratio:.2f}, d_gp={d_gp:.2f}, depth={depth:.2f}")
            return float(depth)

        # default: ratio
        print(f"[DepthEstimation] bbox={bbox}, pixel_h={pixel_h}, d_ratio={d_ratio:.2f}")
        return float(d_ratio)


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

        return (x, y)
    
    def estimate_clearance(self, detections):
        """
        장애물 탐지 결과에서 가장 가까운 객체와의 거리 또는 평균 거리 반환
        값이 작을수록 장애물이 많다고 판단
        """
        if not detections:
            return float('inf')  # 장애물 없음 → 가장 좋은 상태

        depths = []
        for det in detections:
            depth = self.estimate_depth(det["bbox"])
            if depth is not None:
                depths.append(depth)

        if not depths:
            return float('inf')
        return min(depths)  # 가장 가까운 장애물까지 거리 (낮을수록 위험)


if __name__ == "__main__":
    import unittest
    import tempfile
    import os
    import yaml
    import numpy as np

    class TestMonoDepthEstimator(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            # ① 임시 config 파일 생성 (내용은 setUp()에서 덮어씁니다)
            cls.tf = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False)
            yaml.safe_dump({"monodepth": {}}, cls.tf)
            cls.tf.close()

        @classmethod
        def tearDownClass(cls):
            # 테스트 종료 후 임시 파일 삭제
            os.remove(cls.tf.name)

        def setUp(self):
            # ② 매 테스트마다 새로운 인스턴스 생성
            self.est = MonoDepthEstimator(config_path=self.tf.name)
            # ③ 카메라 행렬을 고정된 값(fy=1000, cy=500)으로 세팅
            self.est.fy = 1000.0
            self.est.cy = 500.0

        def test_ratio_depth(self):
            """
            [ratio 모드 테스트]
            object_height=2m, pixel_h=250px 일 때
            depth = object_height * fy / pixel_h = 2*1000/250 = 8m
            """
            self.est.object_height = 2.0
            self.est.method = 'ratio'
            d = self.est.estimate_depth((0, 0, 0, 250))
            self.assertAlmostEqual(d, 8.0, places=6)

        def test_ground_plane_depth(self):
            """
            [ground_plane 모드 테스트]
            tilt=0°, camera_height=1m, y2=600px
            angle = atan((y2-cy)/fy) 이고
            depth = camera_height / tan(angle)
            """
            self.est.camera_height = 1.0
            self.est.tilt_angle = 0.0
            self.est.method = 'ground_plane'
            d = self.est.estimate_depth((0, 0, 0, 600))
            angle = np.arctan((600.0 - 500.0) / 1000.0)
            expected = 1.0 / np.tan(angle)
            self.assertAlmostEqual(d, expected, places=6)

        def test_hybrid_depth(self):
            """
            [hybrid 모드 테스트]
            ratio 와 ground_plane 두 값을 weighted sum 하도록 설정 후,
            예상한 가중합 결과와 일치하는지 확인합니다.
            """
            self.est.object_height = 2.0
            self.est.camera_height = 1.0
            self.est.tilt_angle = 0.0
            self.est.method = 'hybrid'
            self.est.hybrid_w = 0.3

            pixel_h = 200.0
            y2 = 200.0
            d = self.est.estimate_depth((0, 0, 0, pixel_h))

            # ratio 결과
            d_ratio = 2.0 * 1000.0 / pixel_h  # =10.0
            # ground_plane 결과
            angle = np.arctan((y2 - 500.0) / 1000.0)
            d_gp = 1.0 / np.tan(angle)
            expected = 0.3 * d_ratio + 0.7 * d_gp

            self.assertAlmostEqual(d, expected, places=6)

        def test_invalid_bbox(self):
            """
            [잘못된 bbox 처리 테스트]
            y2 <= y1(높이<=0) 인 경우 None 을 반환하는지 확인합니다.
            """
            self.est.method = 'ratio'
            # y2=y1 → pixel_h=0
            self.assertIsNone(self.est.estimate_depth((0, 5, 5, 5)))
            # y2<y1 → pixel_h<0
            self.assertIsNone(self.est.estimate_depth((0, 10, 5, 3)))

        def test_annotate_detections(self):
            """
            [annotate_detections 테스트]
            1) det['depth'] 필드가 추가되는지
            2) 반환 이미지 타입이 numpy.ndarray 인지
            """
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            dets = [{"bbox": [10, 20, 30, 80]}]
            out = self.est.annotate_detections(img.copy(), dets)

            # depth 키가 생성되고, float 타입
            self.assertIn("depth", dets[0])
            self.assertIsInstance(dets[0]["depth"], float)
            # annotate_detections 반환값이 ndarray
            self.assertIsInstance(out, np.ndarray)

    unittest.main()