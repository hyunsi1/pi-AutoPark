import cv2
import time
import threading
import queue
import numpy as np
from enum import Enum, auto
import os, sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility.distance import euclidean_distance
from vision.slot_geometry import find_parking_slot_center,detect_parking_lines

class State(Enum):
    SEARCH = auto()
    NAVIGATE = auto()
    OBSTACLE_AVOID = auto()
    WAIT = auto()
    FINAL_APPROACH = auto()
    COMPLETE = auto()

class DetectionWorker(threading.Thread):
    def __init__(self, capture, detectors: dict, out_q, event):
        super().__init__(daemon=True)
        self.capture = capture
        self.detectors = detectors
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
                    try:
                        self.capture.reopen()
                        self.logger.info("Reopened frame capture successfully")
                    except Exception as e:
                        self.logger.error(f"Failed to reopen capture: {e}")
                    self.failure_count = 0
                time.sleep(0.01)
                continue
            self.failure_count = 0
            detections = {
                name: detector.detect(frame)
                for name, detector in self.detectors.items()
            }
            try:
                self.out_q.put_nowait((frame.copy(), detections))
            except queue.Full:
                _ = self.out_q.get_nowait()
                self.out_q.put_nowait((frame.copy(), detections))
            self.event.set()


class StateMachine:
    def __init__(self, cfg, frame_capture, yolo_detectors, monodepth_estimator,
                 slot_allocator, path_planner, controller, pan_tilt_controller, user_io):

        self.cfg = cfg
        self.capture = frame_capture
        self.detectors = yolo_detectors
        self.depth_estimator = monodepth_estimator
        self.allocator = slot_allocator
        self.planner = path_planner
        self.ctrl = controller
        self.pan_tilt = pan_tilt_controller
        self.ui = user_io

        self.state = State.SEARCH
        self.goal_slot = None
        self.path = []
        self.current_pos = None
        self.logger = logging.getLogger(__name__)

        self.det_q = queue.Queue(maxsize=1)
        self.new_det_event = threading.Event()
        self.det_worker = DetectionWorker(self.capture, self.detectors, self.det_q, self.new_det_event)
        self.det_worker.start()

    def run(self):
        self.ui.prompt_start()
        print("STARTED LOOP")
        prev_state = None
        while True:
            if self.ui.wait_cancel(timeout=0):
                self.logger.info("User cancelled operation")
                break
            if not self.new_det_event.wait(timeout=0.1):
                continue
            self.new_det_event.clear()
            frame, detections = self.det_q.get()

            self.ui.show_status(f"State: {self.state.name}")
            for model_dets in detections.values():
                for det in model_dets:
                    x1, y1, x2, y2 = det['bbox']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.logger.info("User pressed 'q' to cancel parking")
                break

            if self.state == State.SEARCH:
                self._search_step(frame)
            elif self.state == State.NAVIGATE:
                self._navigate_step(detections)
            elif self.state == State.OBSTACLE_AVOID:
                self._avoid_step()
            elif self.state == State.WAIT:
                self._wait_step()
            elif self.state == State.FINAL_APPROACH:
                self._final_approach_step(detections)
            elif self.state == State.COMPLETE:
                self._complete_step()
                break

        cv2.destroyAllWindows()
        self.ui.notify_complete()

    def _search_step(self, frame):
        lines, _ = detect_parking_lines(frame)
        slot_center = find_parking_slot_center(lines, frame.shape)
        if slot_center:
            self.current_pos = self.allocator.p2w(*slot_center)
            self.goal_slot = slot_center
            self.state = State.NAVIGATE
            print("[SEARCH] 슬롯 중심 추정 완료 → NAVIGATE 전환")
        else:
            print("[SEARCH] 슬롯 라인 부족 → 계속 탐색")

    def _navigate_step(self, detections):
        if self.current_pos is None or self.goal_slot is None:
            self.state = State.SEARCH
            return

        threshold_depth = self.cfg.get("obstacle_distance_threshold", 0.5)
        height_ratio_threshold = self.cfg.get("obstacle_height_ratio_threshold", 0.5)
        frame_height = self.capture.height if hasattr(self.capture, 'height') else 480

        for det in detections.get("custom", []):
            depth = self.depth_estimator.estimate_depth(det["bbox"])
            if depth is not None and depth < threshold_depth:
                self.ctrl.stop()
                self.logger.info(f"[Obstacle-Custom] depth={depth:.2f}m")
                self.state = State.WAIT
                return

        for det in detections.get("coco", []):
            x1, y1, x2, y2 = det["bbox"]
            box_height = y2 - y1
            if box_height / frame_height > height_ratio_threshold:
                self.ctrl.stop()
                self.logger.info(f"[Obstacle-COCO] bbox height ratio={box_height/frame_height:.2f}")
                self.state = State.WAIT
                return

        if not self.path:
            self.path = self.planner.plan(self.current_pos, self.goal_slot)
        next_wp = self.path[0]
        self.ctrl.navigate_to(self.current_pos, next_wp)
        self.pan_tilt.set_tilt(0)

        if euclidean_distance(*self.current_pos, *next_wp) < self.cfg.get('waypoint_tolerance', 0.2):
            self.current_pos = next_wp
            self.path.pop(0)

        if not self.path:
            self.state = State.FINAL_APPROACH

    def _wait_step(self):
        print("[WAIT] 장애물 감지로 정지. 2초 대기 후 상태 재확인...")
        time.sleep(2)
        ret, frame = self.capture.read()
        if not ret or frame is None:
            print("[WAIT] 프레임 없음. 다시 대기.")
            return
        detections = {
            name: detector.detect(frame)
            for name, detector in self.detectors.items()
        }
        threshold_depth = self.cfg.get("obstacle_distance_threshold", 0.5)
        height_ratio_threshold = self.cfg.get("obstacle_height_ratio_threshold", 0.5)
        frame_height = frame.shape[0]
        for det in detections.get("custom", []):
            depth = self.depth_estimator.estimate_depth(det["bbox"])
            if depth is not None and depth < threshold_depth:
                print("[WAIT] 여전히 가까운 custom 객체 있음 → 계속 WAIT")
                return
        for det in detections.get("coco", []):
            x1, y1, x2, y2 = det["bbox"]
            if (y2 - y1) / frame_height > height_ratio_threshold:
                print("[WAIT] 여전히 큰 COCO 객체 있음 → 계속 WAIT")
                return
        print("[WAIT] 장애물 사라짐 → NAVIGATE 복귀")
        self.state = State.NAVIGATE

    def _avoid_step(self):
        dets = self.det_q.queue[0][1]['custom']
        if not dets:
            return
        x1, y1, x2, y2 = dets[0]['bbox']
        obs_px = ((x1 + x2) / 2, (y1 + y2) / 2)
        obs_world = self.allocator.p2w(*obs_px)
        bounds = np.array(self.allocator.area_world)
        self.path = self.planner.replan_around(
            self.current_pos, self.goal_slot, obs_world,
            self.cfg['clearance'], bounds
        )
        self.state = State.NAVIGATE

    def _final_approach_step(self, detections):
        for det in detections['coco']:
            x1, y1, x2, y2 = det['bbox']
            area_ratio = ((x2 - x1) * (y2 - y1)) / (640 * 480)
            if area_ratio > 0.4:
                self.ctrl.stop()
                self.logger.info("[FINAL_APPROACH] Stop due to coco object area ratio > 0.4")
                self.state = State.COMPLETE
                return
        if detections['custom']:
            dist = self.depth_estimator.estimate_depth(detections['custom'][0]['bbox'])
            if dist and dist < self.cfg['final_approach']['threshold']:
                self.ctrl.stop()
                self.logger.info("[FINAL_APPROACH] Stop due to custom object being close")
                self.state = State.COMPLETE
                return
        if euclidean_distance(*self.current_pos, *self.goal_slot) < 0.05:
            self.ctrl.stop()
            self.logger.info("[FINAL_APPROACH] 위치상 주차 완료로 판단 → COMPLETE 전환")
            self.state = State.COMPLETE
            return
        self.pan_tilt.set_tilt(self.cfg['pan_tilt']['final_tilt_angle'])
        dx = self.goal_slot[0] - self.current_pos[0]
        dy = self.goal_slot[1] - self.current_pos[1]
        norm = (dx**2 + dy**2)**0.5

        if norm < 1e-6:
            print("[Navigate] 현재 위치와 목표 위치가 동일. 조향 생략")
            return

        unit_vec = (dx / norm, dy / norm)
        target_pos = (
            self.current_pos[0] + unit_vec[0] * 1.0,
            self.current_pos[1] + unit_vec[1] * 1.0
        )
        self.ctrl.navigate_to(self.current_pos, target_pos)

    def _complete_step(self):
        self.logger.info(f"Parked at slot {self.goal_slot}")
        self.state = State.SEARCH
        self.goal_slot = None
        self.path = []
        self.current_pos = None