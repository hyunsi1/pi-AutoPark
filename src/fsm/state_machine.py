import cv2
import time
import logging
import threading
import queue
import numpy as np
from enum import Enum, auto

# DetectionWorker: 비디오 캡처와 YOLO 검출을 비동기로 처리
class DetectionWorker(threading.Thread):
    def __init__(self, capture, detector, out_q, event):
        super().__init__(daemon=True)
        self.capture = capture
        self.detector = detector
        self.out_q = out_q
        self.event = event
        self.failure_count = 0
        self.logger = logging.getLogger(__name__)

    def run(self):
        while True:
            try:
                ret, frame = self.capture.read()
            except Exception as e:
                self.logger.error(f"Frame capture exception: {e}")
                time.sleep(0.5)
                continue
            if not ret or frame is None:
                self.failure_count += 1
                if self.failure_count > 50 and hasattr(self.capture, 'reopen'):
                    self.logger.warning("Too many frame grab failures, attempting to reopen capture")
                    try:
                        self.capture.reopen()
                        self.logger.info("Reopened frame capture successfully")
                    except Exception as e:
                        self.logger.error(f"Failed to reopen capture: {e}")
                    self.failure_count = 0
                time.sleep(0.01)
                continue
            self.failure_count = 0
            detections = self.detector.detect(frame)
            try:
                self.out_q.put_nowait((frame.copy(), detections))
            except queue.Full:
                _ = self.out_q.get_nowait()
                self.out_q.put_nowait((frame.copy(), detections))
            self.event.set()

# 상태 정의
class State(Enum):
    SEARCH = auto()
    NAVIGATE = auto()
    OBSTACLE_AVOID = auto()
    FINAL_APPROACH = auto()
    COMPLETE = auto()

# StateMachine: 전체 FSM을 관리
class StateMachine:
    def __init__(
        self,
        config,
        frame_capture,
        yolo_detector,
        monodepth_estimator,
        slot_allocator,
        path_planner,
        controller,
        pan_tilt_controller,
        user_io
    ):
        # 컴포넌트
        self.cfg = config
        self.capture = frame_capture
        self.detector = yolo_detector
        self.depth = monodepth_estimator
        self.allocator = slot_allocator
        self.planner = path_planner
        self.ctrl = controller
        self.pan_tilt = pan_tilt_controller
        self.ui = user_io

        # 내부 상태
        self.state = State.SEARCH
        self.goal_slot = None
        self.path = []
        self.current_pos = None
        self.logger = logging.getLogger(__name__)

        # 동기화 큐 및 이벤트
        self.det_q = queue.Queue(maxsize=1)
        self.new_det_event = threading.Event()
        self.det_worker = DetectionWorker(
            self.capture,
            self.detector,
            self.det_q,
            self.new_det_event
        )
        self.det_worker.start()

    def run(self):
        # 시작 대기
        self.ui.prompt_start()
        while True:
            # 취소 입력 확인 (메시지 내부)
            if self.ui.wait_cancel(timeout=0):
                self.logger.info("User cancelled operation")
                break

            # 새 프레임·검출 대기
            if not self.new_det_event.wait(timeout=0.1):
                continue
            self.new_det_event.clear()
            frame, detections = self.det_q.get()

            # 상태 및 프레임 표시
            self.ui.show_status(f"State: {self.state.name}")
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.logger.info("User pressed 'q' to cancel parking")
                break

            # 상태별 동작
            if self.state == State.SEARCH:
                self._search_step(detections)
            elif self.state == State.NAVIGATE:
                self._navigate_step()
            elif self.state == State.OBSTACLE_AVOID:
                self._avoid_step()
            elif self.state == State.FINAL_APPROACH:
                self._final_approach_step()
            elif self.state == State.COMPLETE:
                self._complete_step()
                break

        cv2.destroyAllWindows()
        self.ui.notify_complete()

    def _search_step(self, detections):
        # detections 비어 있으면 대기
        if not detections:
            return
        # 빈 슬롯 탐색
        slot = self.allocator.allocate(detections)
        if slot:
            # 최초 장애물 중심을 현재 위치로 변환
            x1, y1, x2, y2 = detections[0]['bbox']
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            self.current_pos = self.allocator.p2w(cx, cy)
            self.goal_slot = slot
            self.state = State.NAVIGATE
            self.logger.info(f"Target slot set: {self.goal_slot}")

    def _navigate_step(self):
        # 현재 위치나 목표 누락 시 SEARCH로
        if self.current_pos is None or self.goal_slot is None:
            self.state = State.SEARCH
            return
        # 경로 계획 (현재 위치 → 목표 슬롯)
        if not self.path:
            self.path = self.planner.plan(self.current_pos, self.goal_slot)
        # 다음 웨이포인트 선택
        next_wp = self.path[0]
        # 주행 제어: 현재 위치 → 다음 웨이포인트
        self.ctrl.navigate_to(self.current_pos, next_wp)
        # 카메라 pan 고정
        self.pan_tilt.set_tilt(0)
        cx, cy = self.current_pos
        nx, ny = next_wp
        # 도달 여부 판단
        from utility.distance import euclidean_distance
        if euclidean_distance(cx, cy, nx, ny) < self.cfg.get('waypoint_tolerance', 0.1):
            self.current_pos = next_wp
            self.path.pop(0)

        if not self.path:
            self.state = State.FINAL_APPROACH
            self.logger.info("Reached final waypoint, switching to FINAL_APPROACH")

    def _avoid_step(self):
        # 장애물 첫 검출 위치
        dets = self.det_q.queue[0][1]
        if not dets:
            return
        x1, y1, x2, y2 = dets[0]['bbox']
        obs_px = ((x1 + x2) / 2, (y1 + y2) / 2)
        obs_world = self.allocator.p2w(*obs_px)
        bounds = np.array(self.allocator.area_world)
        self.path = self.planner.replan_around(
            self.current_pos,
            self.goal_slot,
            obs_world,
            self.cfg['clearance'],
            bounds
        )
        self.state = State.NAVIGATE

    def _final_approach_step(self):
        dets = self.det_q.queue[0][1]
        if not dets:
            return
        dist = self.depth.estimate_depth(dets[0]['bbox'])
        if dist is None:
            return
        if dist < self.cfg['final_approach']['threshold']:
            self.state = State.COMPLETE
        else:
            self.pan_tilt.set_tilt(self.cfg['pan_tilt']['final_tilt_angle'])
            self.ctrl.navigate_to(self.goal_slot, dist)

    def _complete_step(self):
        self.logger.info(f"Parked at slot {self.goal_slot}")
        # 초기화
        self.state = State.SEARCH
        self.goal_slot = None
        self.path = []
        self.current_pos = None


if __name__ == '__main__':
    import unittest

    class Dummy:
        def __init__(self): self.called = False
        def __call__(self, *args, **kwargs): self.called = True; return []

    class TestStateMachine(unittest.TestCase):
        def setUp(self):    # 더미 객체 생성
            self.cfg = {'pan_tilt': {'pan_channel': 0, 'tilt_channel': 1}, 'final_approach': {'threshold': 0.5}}
            self.capture = type('C', (), {'read': lambda s: (True, 'frame')})()
            self.detector = type('D', (), {'detect': lambda s, f: [(0,0,1,1)]})()
            self.depth = type('M', (), {'estimate_center': lambda s, f, slot: 0.3})()
            self.allocator = type('A', (), {'allocate': lambda s, det: ['slot1']})()
            self.planner = type('P', (), {'plan': lambda s, slot: ['p1'], 'replan_around': lambda s,f,slot: ['p2']})()
            self.controller = type('C3', (), {'navigate_to': lambda s, path: True})()
            self.pan_tilt = type('PT', (), {'goto': lambda s, ch, ang: None})()
            self.ui = type('UI', (), {
                'prompt_start': lambda s: None,
                'wait_cancel': lambda s, timeout=0: True,
                'show_status': lambda s, st: None,
                'notify_complete': lambda s: None
            })()
            self.sm = StateMachine(self.cfg, self.capture, self.detector, self.depth,
                                   self.allocator, self.planner, self.controller,
                                   self.pan_tilt, self.ui)
            # 워커 스레드 중지
            self.sm.det_worker.join(timeout=0)

        def test_search_to_navigate(self):
            self.sm.state = State.SEARCH
            self.sm._search_step([(0,0,1,1)])
            self.assertEqual(self.sm.state, State.NAVIGATE)
            self.assertEqual(self.sm.goal_slot, 'slot1')

        def test_navigate_to_final(self):
            self.sm.state = State.NAVIGATE
            self.sm.goal_slot = 'slot1'
            self.sm.path = []
            self.sm._navigate_step('frame')
            self.assertEqual(self.sm.state, State.FINAL_APPROACH)

        def test_final_approach_to_complete(self):
            self.sm.state = State.FINAL_APPROACH
            self.sm.goal_slot = 'slot1'
            self.sm._final_approach_step('frame')
            self.assertEqual(self.sm.state, State.COMPLETE)

        def test_complete_resets(self):
            self.sm.state = State.COMPLETE
            self.sm.goal_slot = 'slot1'
            self.sm.path = ['p']
            self.sm._complete_step()
            self.assertEqual(self.sm.state, State.SEARCH)
            self.assertIsNone(self.sm.goal_slot)
            self.assertEqual(self.sm.path, [])

    unittest.main(argv=[''], exit=False)
