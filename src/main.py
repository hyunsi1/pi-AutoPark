import yaml
import logging
from camera.capture import FrameCapture
from vision.yolo_detector import YOLODetector
from vision.monodepth_estimator import MonoDepthEstimator
from vision.slot_allocator import SlotAllocator
from navigation.path_planner import PathPlanner
from navigation.controller import Controller
from camera.pan_tilt_control import PanTiltController
from interface.user_io import UserIO
from fsm.state_machine import StateMachine

import time


def load_config(path='config/config.yaml'):
    # UTF-8로 파일을 읽어 cp949 디코딩 오류 방지
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    # 1. 설정 로드 및 로깅 초기화
    cfg = load_config()
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info('Starting PI AutoPark')

    # 2. 컴포넌트 인스턴스 생성
    frame_capture = FrameCapture(
        source=cfg['camera']['source'],
        image_folder=cfg['camera']['image_folder'],
        loop=cfg['camera']['loop']
    )
    # YOLODetector는 config 파일 내 값을 읽어 초기화합니다
    yolo_coco = YOLODetector(weights_path=cfg['yolo']['coco_weights'])
    yolo_custom = YOLODetector(weights_path=cfg['yolo']['custom_weights'])
        # MonoDepthEstimator는 config 파일에서 설정을 로드하여 초기화합니다
    depth = MonoDepthEstimator()
        # SlotAllocator는 픽셀 좌표 리스트만 전달합니다
    allocator = SlotAllocator(
        cfg['slot_area_coords']
    )
        # PathPlanner는 segment 개수만 전달합니다
    planner = PathPlanner(
        cfg['path_segments']
    )
    controller = Controller()
    pan_tilt = PanTiltController(
    tilt_channel=cfg['pan_tilt']['tilt_channel']
    )
    ui = UserIO()
    yolo_detectors = {
        "coco": yolo_coco,
        "custom": yolo_custom
    }
    # 3. 상태 머신 실행
    sm = StateMachine(cfg,
                  frame_capture,
                  yolo_detectors=yolo_detectors,
                  monodepth_estimator=depth,
                  slot_allocator=allocator,
                  path_planner=planner,
                  controller=controller,
                  pan_tilt_controller=pan_tilt,
                  user_io=ui)
    sm.run()
    


if __name__ == '__main__':
    main()
