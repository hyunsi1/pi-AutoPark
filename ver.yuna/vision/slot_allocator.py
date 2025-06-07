import yaml
import numpy as np
from utility.transformations import pixel_to_world
import os

class SlotAllocator:
    """
    주차 슬롯 할당기

    하나의 큰 사각형 주차 구역 좌표(4개 점)를 받아서,
    킥보드 크기와 간격에 맞게 격자 형태로 슬롯을 생성합니다.
    Monodepth를 통해 추정된 킥보드 위치(세계 좌표)를 이용해
    빈 슬롯을 필터링하고, 우선순위에 따라 할당합니다.
    """
    def __init__(self,
                 area_coords_px: list,
                 config_path: str = None):
        """
        Args:
            area_coords_px: 이미지(혹은 BEV)상의 주차 구역 네 꼭짓점 픽셀 좌표,
                            순서: [top_left, top_right, bottom_right, bottom_left]
            config_path: config/config.yaml 경로
        """
        # config 불러오기
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 'config', 'config.yaml'
            )
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        md_cfg = cfg.get('monodepth', {})
        self.board_width  = float(md_cfg.get('object_width', 0.5))
        self.board_length = float(md_cfg.get('object_height', 1.0))
        self.slot_gap = float(cfg.get('slot_gap', 0.1))

        # homography 행렬 불러오기
        homography_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'camera_params.npz')
        if os.path.exists(homography_path):
            data = np.load(homography_path)
            self.homography_matrix = data['homography_matrix']
        else:
            import logging
            logging.warning(f"'{homography_path}' 파일이 없습니다. 기본 homography 행렬을 사용합니다.")
            self.homography_matrix = np.eye(3)

        # (x, y) → 월드 좌표로 변환하는 함수 정의
        self.p2w = lambda x, y: pixel_to_world(x, y, self.homography_matrix)

        # 픽셀 좌표를 월드 좌표로 변환
        self.area_px = np.array(area_coords_px, dtype=float)
        self.area_world = np.array([self.p2w(x, y) for x, y in self.area_px])  # (4, 2)

        self.slots = self._generate_slots()

    def _generate_slots(self):
        """사전 정의된 주차 구역을 킥보드 크기에 맞게 격자 슬롯으로 분할"""
        tl, tr, br, bl = self.area_world
        # 구역 폭(가로) 및 높이(세로) 벡터
        width_vec  = tr - tl
        height_vec = bl - tl
        total_width  = np.linalg.norm(width_vec)
        total_height = np.linalg.norm(height_vec)

        # 한 슬롯 크기: 킥보드 길이(전후) + 여유, 너비 + 여유
        slot_w = self.board_width  + self.slot_gap
        slot_h = self.board_length + self.slot_gap

        # 슬롯 개수
        n_cols = int(np.floor(total_width  / slot_w))
        n_rows = int(np.floor(total_height / slot_h))

        # 슬롯 중심 좌표 리스트
        slots = []
        for r in range(n_rows):
            for c in range(n_cols):
                # 비율 위치
                u = (c + 0.5) * slot_w / total_width
                v = (r + 0.5) * slot_h / total_height
                world_pt = tl + u * width_vec + v * height_vec
                slots.append(tuple(world_pt.tolist()))
        return slots

    def allocate(self, detections: list):
        """킥보드 위치 기준 가장 가까운 슬롯 우선 할당"""
        if not detections:
            return None

        # 킥보드 중심 좌표들을 월드 좌표로 변환
        occ_pts = [self.p2w((d['bbox'][0] + d['bbox'][2]) // 2,
                            (d['bbox'][1] + d['bbox'][3]) // 2) for d in detections]

        # 슬롯을 행 기준으로 그룹화
        row_dict = {}
        for slot in self.slots:
            y_key = round(slot[1], 2)  # 행 구분용
            row_dict.setdefault(y_key, []).append(slot)

        # 행 단위로 가장 위쪽(카메라에서 가까운)부터 검사
        for row_y in sorted(row_dict.keys()):
            row_slots = row_dict[row_y]
            # 킥보드와의 거리 기준 정렬
            slot_distances = []
            for slot in row_slots:
                min_dist = min(np.linalg.norm(np.array(slot) - np.array(occ)) for occ in occ_pts)
                slot_distances.append((min_dist, slot))
            slot_distances.sort()  # 거리순 정렬

            for _, slot in slot_distances:
                # 주변 킥보드들과 충분히 떨어져 있으면 선택
                dists = [np.linalg.norm(np.array(slot) - np.array(occ)) for occ in occ_pts]
                if min(dists) > max(self.board_width, self.board_length):
                    return slot  # 가장 가까우면서 비어있는 슬롯

        return None  # 모두 점유된 경우

if __name__ == "__main__":
    import unittest
    import tempfile
    import yaml
    import os
    import numpy as np

    class TestSlotAllocator(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            # 1) 임시 config.yaml 만들기
            cls.tmp_cfg = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False)
            yaml.safe_dump({
                "monodepth": {"object_width":1.0, "object_height":1.0},
                "slot_gap": 0.0
            }, cls.tmp_cfg)
            cls.tmp_cfg.close()
            # 2) homography 파일이 없다고 속이기
            cls._orig_exists = os.path.exists
            os.path.exists = lambda path: False

        @classmethod
        def tearDownClass(cls):
            # 테스트 후 정리
            os.remove(cls.tmp_cfg.name)
            os.path.exists = cls._orig_exists

        def test_generate_slots(self):
            # 4×4 영역, 보드 1×1, 간격 0 → 16슬롯
            area = [(0,0),(4,0),(4,4),(0,4)]
            alloc = SlotAllocator(area, config_path=self.tmp_cfg.name)
            self.assertEqual(len(alloc.slots), 16)

        def test_allocate_nearest(self):
            # detection bbox 중심 (0.5,0.5)일 때 가장 가까운 빈 슬롯 검증
            area = [(0,0),(4,0),(4,4),(0,4)]
            alloc = SlotAllocator(area, config_path=self.tmp_cfg.name)
            det = {"bbox":[0,0,1,1]}
            slot = alloc.allocate([det])
            self.assertAlmostEqual(slot[0], 1.5, places=2)
            self.assertAlmostEqual(slot[1], 0.5, places=2)

        def test_allocate_empty(self):
            # detections=[] 일 때 None 반환 확인
            alloc = SlotAllocator([(0,0),(2,0),(2,2),(0,2)], config_path=self.tmp_cfg.name)
            self.assertIsNone(alloc.allocate([]))

    # 이 파일을 직접 실행하면 unittest가 동작합니다.
    unittest.main()