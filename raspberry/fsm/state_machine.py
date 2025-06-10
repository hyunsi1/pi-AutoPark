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
from vision.approach_util import find_front_reference, find_side_reference,side_reference_gone,front_reference_gone


class State(Enum):
    SEARCH = 1
    NAVIGATE = 2
    AVOID = 3
    WAIT = 4
    FINAL_APPROACH = 5
    COMPLETE = 6
    ERROR = 7

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
        self.frame_interval = 1 / 10

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
        self.ctrl.set_angle(65)  # 기본 직진 각도 설정
        self.pan_tilt = pan_tilt_controller
        self.pan_tilt.reset()
        self.ui = user_io

        self.state = State.SEARCH
        self.goal_slot = None
        self.current_pos = None
        self.path_queue = []
        self._fa_initialized = False
        self.wait_count = 0
        self.logger = logging.getLogger(__name__)

        '''ret, frame = self.capture.read()
        if not ret or frame is None:
            raise RuntimeError("Cannot read first frame to get resolution")
        h, w = frame.shape[:2]'''
        
        # YOLO 멀티스레드 감지 워커
        self.det_q = queue.Queue(maxsize=1)
        self.new_det_event = threading.Event()
        self.det_worker = DetectionWorker(self.capture, self.detectors, self.det_q, self.new_det_event)
        self.det_worker.start()

        # 카메라 보정 매트릭스 로드
        homography_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'config', 'camera_params.npz'
        )
        if os.path.exists(homography_path):
            data = np.load(homography_path)
            self.homography_matrix = data['homography_matrix']
        else:
            logging.warning(f"[INIT] '{homography_path}' not found, using identity matrix")
            self.homography_matrix = np.eye(3)

        # 픽셀 → 월드 좌표 변환 함수
        self.p2w = lambda x, y: pixel_to_world(x, y, self.homography_matrix)

    def run(self):
        self.logger.info("[FSM] State machine started.")
        while True:
            try:
                if self.state == State.SEARCH:
                    ret, frame = self.capture.read()
                    if not ret:
                        self.logger.warning("[SEARCH] Failed to capture frame.")
                        continue
                    self._search_step(frame)

                elif self.state == State.NAVIGATE:
                    self._navigate_step()

                elif self.state == State.WAIT:
                    self._wait_step()

                elif self.state == State.AVOID:
                    self._avoid_step()

                elif self.state == State.FINAL_APPROACH:
                    detections = {name: det.detect(self.capture.read()[1])
                                for name, det in self.detectors.items()}
                    self._final_approach_step(detections)

                elif self.state == State.COMPLETE:
                    self._complete_step()
                    break

                elif self.state == State.ERROR:
                    self.ctrl.stop()
                    self.logger.error("[FSM] Entered ERROR state.")
                    break

                else:
                    self.logger.error(f"[FSM] Unknown state: {self.state}")
                    self.state = State.ERROR
                    break

            except Exception as e:
                self.logger.exception(f"[FSM] Exception occurred in state {self.state.name}: {e}")
                self.state = State.ERROR

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
            raw_path = self.planner.plan(self.current_pos, self.goal_slot)
            self.path_queue = self.planner.annotate_path_with_angles(raw_path)

            self.state = State.NAVIGATE

        else:
            # Fallback: 킥보드 탐지 기반으로 목표 설정
            scooter_dets = self.detectors["custom"].detect(frame)
            if not scooter_dets:
                return
            scooter_dets.sort(key=lambda d: (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]), reverse=True)
            bbox = scooter_dets[0]['bbox']  # 가장 큰 bbox 사용
            x1, y1, x2, y2 = bbox

            # 깊이 추정
            scooter_dist = self.depth_estimator.estimate_depth(bbox)
            if scooter_dist is None:
                return
            self.goal_distance = scooter_dist
            self.current_pos = self.depth_estimator.estimate_current_position_from_y2(y2)

            # bbox 중심 기준 world 좌표 계산
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            center_world = self.p2w(mid_x, mid_y)

            # 가정: 왼쪽 방향으로 40cm 오프셋하여 주차 목표 설정
            # 카메라가 전방을 바라보고 있다면 -x 방향이 킥보드 왼쪽이 됨
            offset_angle_rad = math.radians(-90)  # 왼쪽 방향
            offset_dist = 0.4  # 40cm

            goal_x = center_world[0] + offset_dist * math.cos(offset_angle_rad)
            goal_y = center_world[1] + offset_dist * math.sin(offset_angle_rad)
            self.goal_slot = (goal_x, goal_y)

            self.logger.info(f"[SEARCH] Scooter fallback. dist={scooter_dist:.2f}")
            self.logger.info(f"[SEARCH] Estimated fallback goal at {self.goal_slot}")

            # 장애물 등록
            for det in self.detectors["custom"].detect(frame):
                ox, oy = det["bbox"][0], det["bbox"][1]
                self.planner.set_obstacle(self.p2w(ox, oy))

            # 경로 생성
            raw_path = self.planner.plan(self.current_pos, self.goal_slot)
            self.path_queue = self.planner.annotate_path_with_angles(raw_path)
            self.state = State.NAVIGATE

    def _navigate_step(self):
        # 1) 준비 체크
        if self.current_pos is None or self.goal_slot is None:
            self.logger.warning("[NAVIGATE] Missing current_pos or goal_slot → SEARCH")
            self.state = State.SEARCH
            return

        # 2) 최신 detection
        self.new_det_event.wait(timeout=1.0)
        try:
            _, detections = self.det_q.get_nowait()
        except queue.Empty:
            self.logger.warning("[NAVIGATE] No detection data available")
            return

        # 3) 장애물 유무 먼저 체크
        if self.planner.obstacle_detector(
            detections.get("custom", []),
            danger_classes=self.cfg.get("obstacle_classes", [0]),
            danger_distance=self.cfg.get("obstacle_distance_threshold", 1.0)
        ):
            self.logger.info("[NAVIGATE] Obstacle detected → WAIT")
            self.state = State.WAIT
            return

        # 4) 매 프레임마다 항상 새로 경로 계산
        try:
            waypoints = self.planner.plan(
                start=self.current_pos,
                goal=self.goal_slot,
                grid_size=self.cfg.get("grid_size", 0.2)
            )
        except Exception as e:
            self.logger.warning(f"[NAVIGATE] plan() failed: {e} → direct fallback")
            dx = self.goal_slot[0] - self.current_pos[0]
            dy = self.goal_slot[1] - self.current_pos[1]
            waypoints = [{
                "pos": (self.goal_slot[0], self.goal_slot[1]),
                "angle": 0.0,
                "distance": math.hypot(dx, dy)
            }]

        # 5) 계산된 웨이포인트 순회
        for idx, wp in enumerate(waypoints):
            # 안전하게 튜플 언패킹
            try:
                x = float(wp["pos"][0])
                y = float(wp["pos"][1])
            except Exception as ex:
                self.logger.error(f"[NAVIGATE] Waypoint #{idx} bad pos={wp!r}: {ex}")
                self.state = State.ERROR
                return

            x0, y0 = self.current_pos
            x1 = float(wp["pos"][0])
            y1 = float(wp["pos"][1])
            line = (x0, y0, x1, y1)

            servo_ang = self.ctrl.steering_and_offset(line)
            self.ctrl.set_angle(servo_ang)
            self.ctrl.set_speed(self.cfg.get("navigate_speed", 30), reverse=False)

            distance = float(wp.get("distance", math.hypot(x1 - x0, y1 - y0)))

            # 5-2) 주행
            world_dist = distance * self.planner.pixel_to_meter
            travel_time = world_dist / self.planner.speed_mps
            print(f"[navigate_segment] distance={distance:.2f}m, speed={self.planner.speed_mps:.2f}m/s, travel_time={travel_time:.2f}s")
            t0 = time.time()
            while time.time() - t0 < travel_time:
                # 이동 중에도 장애물 재검사
                self.new_det_event.wait(timeout=1.0)
                try:
                    _, det2 = self.det_q.get_nowait()
                except queue.Empty:
                    det2 = detections
                if self.planner.obstacle_detector(
                    det2.get("custom", []),
                    danger_classes=self.cfg.get("obstacle_classes", [0]),
                    danger_distance=self.cfg.get("obstacle_distance_threshold", 1.0)
                ):
                    self.ctrl.stop()
                    self.logger.info("[NAVIGATE] Obstacle during move → WAIT")
                    self.state = State.WAIT
                    return
                time.sleep(0.1)

            # 5-3) 스텝 완료
            self.ctrl.stop()
            self.current_pos = (x, y)

        # 6) 모든 웨이포인트 완료 시
        self.logger.info("[NAVIGATE] Path complete → FINAL_APPROACH")
        self.state = State.FINAL_APPROACH


    def _wait_step(self):
        if not hasattr(self, "wait_start_time"):
            self.wait_start_time = time.time()
            self.wait_failures = getattr(self, "wait_failures", 0)
            self.ctrl.stop()
            self.logger.info("[WAIT] Dynamic obstacle detected. Pausing 1s.")
            return

        if time.time() - self.wait_start_time < 1.0:
            return

        # 프레임 읽기 및 동적 장애물 확인
        ret, frame = self.capture.read()
        if not ret or frame is None:
            self.wait_start_time = time.time()
            return

        detections = {name: detector.detect(frame) for name, detector in self.detectors.items()}
        frame_height = frame.shape[0]
        obstacle_detected = False

        for det in detections.get("coco", []):
            x1, y1, x2, y2 = det["bbox"]
            height_ratio = (y2 - y1) / frame_height
            if height_ratio > self.cfg.get("obstacle_height_ratio_threshold", 0.7):
                obstacle_detected = True
                self.logger.debug(f"[WAIT] Dynamic object detected (height_ratio={height_ratio:.2f})")
                break

        if obstacle_detected:
            self.wait_start_time = time.time()
            self.wait_failures += 1
            if self.wait_failures >= self.cfg.get("max_wait_retries", 5):
                self.logger.warning("[WAIT] Too many retries. Triggering AVOID replanning.")
                del self.wait_start_time
                self.state = State.AVOID
        else:
            self.logger.info("[WAIT] Obstacle cleared. Resuming navigation.")
            del self.wait_start_time
            self.wait_failures = 0
            self.state = State.NAVIGATE

    def _avoid_step(self):
        if self.current_pos is None or self.goal_slot is None:
            self.logger.error("[AVOID] Missing current_pos or goal_slot, reverting to SEARCH.")
            self.state = State.SEARCH
            return

        try:
            self.logger.info("[AVOID] Replanning path using A* due to persistent obstacle.")
            raw_path = self.planner.plan(self.current_pos, self.goal_slot)
            self.path_queue = self.planner.annotate_path_with_angles(raw_path)


            if not self.path_queue:
                self.logger.error("[AVOID] Failed to find new path. Staying in ERROR.")
                self.state = State.ERROR
                return

            self.logger.info(f"[AVOID] New path calculated with {len(self.path_queue)} waypoints.")
            self.state = State.NAVIGATE

        except Exception as e:
            self.logger.exception(f"[AVOID] Exception during path replanning: {e}")
            self.state = State.ERROR

    def _final_approach_step(self, detections):
        """
        Final approach: repeat small reverse + steering steps to align
        with front (��) or side (��) reference lines.
        """
        # 1) Initialization: tilt camera down once
        if not hasattr(self, "_fa_inited"):
            self.pan_tilt.final_step_tilt_down()
            self._fa_inited = True
            self.logger.info("[FINAL] Tilt down")

        # 2) Load timing and speed parameters
        step_time    = self.cfg.get("fa_step_time",    0.1)   # seconds per step
        final_speed  = self.cfg.get("final_speed",    20)    # PWM duty or m/s
        max_cycles   = self.cfg.get("fa_max_cycles", 100)    # safety cap

        for cycle in range(max_cycles):
            # 3) grab frame
            ret, frame = self.capture.read()
            if not ret or frame is None:
                break

            # 4) detect reference lines
            front_line,_ = find_front_reference(frame)  # returns ((x1,y1),(x2,y2)) or None
            if front_line is not None:
                # target heading = line_angle + 90�� for perpendicular
                (x1,y1),(x2,y2) = front_line
                raw_line_angle = math.degrees(math.atan2(y2-y1, x2-x1))
                target_heading = raw_line_angle + 90.0
            else:
                side_line,_ = find_side_reference(frame)
                if side_line is not None:
                    (x1,y1),(x2,y2) = side_line
                    raw_line_angle = math.degrees(math.atan2(y2-y1, x2-x1))
                    target_heading = raw_line_angle       # parallel
                else:
                    # no reference seen �� done
                    break

            # 5) clamp heading into [-45,45] via steering_and_offset
            #    build a dummy line vector in world coords
            rad = math.radians(target_heading)
            dx = math.cos(rad)
            dy = math.sin(rad)
            servo_cmd = self.ctrl.steering_and_offset((0.0, 0.0, dx, dy))
            self.ctrl.set_angle(servo_cmd, delay=0)

            # 6) perform small reverse step
            self.ctrl.set_speed(final_speed, reverse=True)
            time.sleep(step_time)
            self.ctrl.stop()

            # 7) check if both references are gone
            ret2, frame2 = self.capture.read()
            if ret2 and frame2 is not None:
                if front_reference_gone(frame2) and side_reference_gone(frame2):
                    self.logger.info("[FINAL] References gone �� COMPLETE")
                    self.state = State.COMPLETE
                    del self._fa_inited
                    return

        # fallback: safety exit
        self.logger.warning("[FINAL] max cycles reached �� COMPLETE")
        self.state = State.COMPLETE
        if hasattr(self, "_fa_inited"):
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
        try:
            self.ctrl.cleanup()
        except Exception as e:
            self.logger.warning(f"[CLEANUP] Controller cleanup failed: {e}")
        try:
            self.pan_tilt.release()
        except Exception as e:
            self.logger.warning(f"[CLEANUP] PanTilt release failed: {e}")
        try:
            self.capture.release()
        except Exception as e:
            self.logger.warning(f"[CLEANUP] Camera release failed: {e}")