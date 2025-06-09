import cv2
import time
import logging
import threading
import math
import queue
import numpy as np
from enum import Enum
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility.distance import euclidean_distance
from vision.slot_geometry import find_black_rect_and_distance, find_left_reference, front_reference_gone
from utility.transformations import pixel_to_world

class State(Enum):
    SEARCH = 1
    NAVIGATE = 2
    OBSTACLE_AVOID = 3
    FINAL_APPROACH = 4
    COMPLETE = 5
    ERROR = 6
    WAIT = 7

class DetectionWorker(threading.Thread):
    def __init__(self, capture, detectors: dict, out_q, event):
        super().__init__(daemon=True)
        self.capture = capture
        self.detectors = list(detectors.items())
        self.out_q = out_q
        self.event = event
        self.failure_count = 0
        self.det_index = 0
        self.logger = logging.getLogger(__name__)

        self.prev_results = {}
        self.last_frame_time = time.time()
        self.frame_interval = 1 / 15

    def run(self):
        while True:
            current_time = time.time()
            if current_time - self.last_frame_time >= self.frame_interval:
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
                name, detector = self.detectors[self.det_index]
                det_result = detector.detect(frame)
                self.prev_results[name] = det_result
                self.det_index = (self.det_index + 1) % len(self.detectors)
                try:
                    self.out_q.put_nowait((frame.copy(), self.prev_results.copy()))
                except queue.Full:
                    _ = self.out_q.get_nowait()
                    self.out_q.put_nowait((frame.copy(), self.prev_results.copy()))
                self.event.set()
                self.last_frame_time = current_time
            time.sleep(0.01)

class StateMachine:
    def __init__(self, cfg, frame_capture, yolo_detectors, monodepth_estimator,
                 path_planner, controller, pan_tilt_controller, user_io):
        self.cfg = cfg
        self.capture = frame_capture
        self.detectors = yolo_detectors
        self.depth_estimator = monodepth_estimator
        self.planner = path_planner
        self.ctrl = controller
        self.ctrl.set_angle(65)
        self.pan_tilt = pan_tilt_controller
        self.ui = user_io

        self.state = State.SEARCH
        self.goal_slot = None
        self.current_pos = None
        self._avoid_started = False
        self._fa_initialized = False
        self.logger = logging.getLogger(__name__)
        self.wait_count = 0

        self.det_q = queue.Queue(maxsize=1)
        self.new_det_event = threading.Event()
        self.det_worker = DetectionWorker(self.capture, self.detectors, self.det_q, self.new_det_event)
        self.det_worker.start()
        homography_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'config', 'camera_params.npz'
        )
        if os.path.exists(homography_path):
            data = np.load(homography_path)
            self.homography_matrix = data['homography_matrix']
        else:
            logging.warning(f"'{homography_path}' not found")
            self.homography_matrix = np.eye(3)

        self.p2w = lambda x, y: pixel_to_world(x, y, self.homography_matrix)

    def run(self):
        while True:
            if self.state == State.SEARCH:
                ret, frame = self.capture.read()
                if not ret:
                    continue
                self._search_step(frame)

            elif self.state == State.NAVIGATE:
                ret, frame = self.capture.read()
                if not ret:
                    continue
                detections = {name: det.detect(frame) for name, det in self.detectors.items()}
                self._navigate_step(frame, detections)

            elif self.state == State.WAIT:
                self._wait_step()

            elif self.state == State.FINAL_APPROACH:
                detections = {name: det.detect(self.capture.read()[1]) for name, det in self.detectors.items()}
                self._final_approach_step(detections)

            elif self.state == State.COMPLETE:
                break

            elif self.state == State.ERROR:
                self.ctrl.stop()
                self.logger.error("Entered ERROR state.")
                break

        cv2.destroyAllWindows()
        self.ui.notify_complete()

    def _search_step(self, frame):
        self.wait_count = 0
        top_left, corners, slot_dist, _ = find_black_rect_and_distance(frame, debug=True)

        if top_left is not None and top_left[0] >= 20 and top_left[1] >= 20:
            self.area_px = corners
            self.area_world = [self.p2w(x, y) for x, y in corners]

            # 왼쪽 라인: top-left (0), bottom-left (3)
            left_top_world = self.p2w(*corners[0])
            left_bottom_world = self.p2w(*corners[3])

            # 슬롯 좌측 라인 벡터
            line_vec = np.array([
                left_bottom_world[0] - left_top_world[0],
                left_bottom_world[1] - left_top_world[1]
            ])
            line_unit = line_vec / (np.linalg.norm(line_vec) + 1e-6)

            # 슬롯 수직 방향 벡터 (슬롯의 위쪽 방향)
            up_vec = np.array([-line_unit[1], line_unit[0]])

            # 킥보드 위치 보정
            offset_along_line = -0.30  # 30cm 뒤 (길이)
            offset_perpendicular = 0.05  # 5cm 우측 (너비 절반)

            goal = np.array(left_top_world) \
                + offset_along_line * line_unit \
                + offset_perpendicular * up_vec

            self.goal_slot = tuple(goal)
            self.goal_distance = slot_dist
            self.current_pos = self.depth_estimator.estimate_current_position_from_y2(corners[0][1])

            self.logger.info(f"[SEARCH] Parking slot at {top_left}, dist: {slot_dist:.2f}m")
            self.logger.info(f"[SEARCH] Adjusted goal at {self.goal_slot}, current pos: {self.current_pos}")

            # 장애물 등록
            for det in self.detectors["custom"].detect(frame):
                ox, oy = det["bbox"][0], det["bbox"][1]
                self.planner.set_obstacle(self.p2w(ox, oy))

            # 경로 생성
            self.path_queue = self.planner.plan(self.current_pos, self.goal_slot)
            self.state = State.NAVIGATE

        else:
            # Fallback: scooter 기반
            scooter_dets = self.detectors["custom"].detect(frame)
            if not scooter_dets:
                return
            scooter_dets.sort(key=lambda d: (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]), reverse=True)
            bbox = scooter_dets[0]['bbox']
            scooter_dist = self.depth_estimator.estimate_depth(bbox)
            if scooter_dist is None:
                return
            self.goal_distance = scooter_dist
            self.current_pos = self.depth_estimator.estimate_current_position_from_y2(bbox[3])
            self.logger.info(f"[SEARCH] Scooter fallback. dist={scooter_dist:.2f}")
            self.state = State.NAVIGATE

    def _navigate_step(self, frame, detections):
        if self.goal_slot is None or self.current_pos is None:
            self.state = State.SEARCH
            return

        if not self.path_queue:
            self.logger.info("[NAVIGATE] No more path. Transitioning to FINAL_APPROACH")
            self.state = State.FINAL_APPROACH
            return

        next_pos = self.path_queue.pop(0)
        dx = next_pos[0] - self.current_pos[0]
        dy = next_pos[1] - self.current_pos[1]

        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        angle_deg = max(-45, min(45, angle_deg))

        servo_ang = self.ctrl.map_physical_angle_to_servo(angle_deg)
        self.ctrl.set_angle(servo_ang)
        self.ctrl.set_speed(self.cfg.get("navigate_speed", 30))

        timeout = time.time() + 5.0
        while not self.ctrl.update_navigation():
            if time.time() > timeout:
                self.logger.error("[NAVIGATE] Timeout during movement")
                self.state = State.ERROR
                return
            time.sleep(0.01)
        self.ctrl.stop()

        self.current_pos = next_pos
        dist = math.hypot(self.goal_slot[0] - self.current_pos[0], self.goal_slot[1] - self.current_pos[1])
        if dist <= self.cfg.get("final_approach_threshold", 0.5):
            self.logger.info("[NAVIGATE] Close enough. Switching to FINAL_APPROACH")
            self.state = State.FINAL_APPROACH

    def _obstacle_avoid_step(self):
        self.logger.info("[AVOID] Replanning after obstacle detection")
        self.path_queue = self.planner.a_star(self.current_pos, self.goal_slot)
        self.state = State.NAVIGATE

    def _wait_step(self):
        now = time.time()
        if not hasattr(self, "wait_start_time"):
            self.wait_start_time = now
            self.ctrl.stop()
            return

        if now - self.wait_start_time < 1.0:
            return

        ret, frame = self.capture.read()
        if not ret or frame is None:
            self.wait_start_time = now
            return

        detections = {name: detector.detect(frame) for name, detector in self.detectors.items()}

        obstacle_detected = False
        frame_height = frame.shape[0]

        for det in detections.get("custom", []):
            depth = self.depth_estimator.estimate_depth(det["bbox"])
            if depth and depth < self.cfg.get("obstacle_distance_threshold", 0.5):
                self.wait_count += 1
                obstacle_detected = True
                break

        if not obstacle_detected:
            for det in detections.get("coco", []):
                x1, y1, x2, y2 = det["bbox"]
                if (y2 - y1) / frame_height > self.cfg.get("obstacle_height_ratio_threshold", 0.7):
                    self.wait_count += 1
                    obstacle_detected = True
                    break

        if not obstacle_detected:
            if self.wait_count >= 5:
                self.state = State.NAVIGATE
                del self.wait_start_time
            else:
                self.wait_start_time = now
        else:
            self.wait_start_time = now

    def _final_approach_step(self, detections):
        if not hasattr(self, '_fa_inited'):
            self.pan_tilt.final_step_tilt_down()
            self._fa_inited = True
            self.logger.info("[FINAL] Final tilt 30")

        ret, frame = self.capture.read()
        if not ret:
            return

        line, _ = find_left_reference(frame)
        if line is not None:
            servo = self.ctrl.steering_and_offset(line, self.ctrl)
            self.ctrl.set_angle(servo, delay=0)

        self.ctrl.set_speed(30, reverse=False)
        time.sleep(1)

        if front_reference_gone(frame):
            self.ctrl.stop()
            self.logger.info("[FINAL] 기준선 사라짐, 주차 완료")
            time.sleep(1)
            self.state = State.COMPLETE
            del self._fa_inited

    def _complete_step(self):
        self.goal_slot = None
        self.current_pos = None
        self.state = State.SEARCH
        self.ctrl.stop()
        self.pan_tilt.reset()
        self.ui.notify_complete()
        self.cleanup()

    def cleanup(self):
        self.ctrl.cleanup()
        self.pan_tilt.release()
        self.capture.release()