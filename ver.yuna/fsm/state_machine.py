import cv2
import time
import logging
import threading
import math
import queue
import numpy as np
from enum import Enum, auto
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility.distance import euclidean_distance
from vision.slot_geometry import find_black_rect_and_distance
from utility.transformations import pixel_to_world

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
        self.detectors = list(detectors.items())  # [('coco', model), ('custom', model)]
        self.out_q = out_q
        self.event = event
        self.failure_count = 0
        self.det_index = 0  # alternating index
        self.logger = logging.getLogger(__name__)

        self.prev_results = {}  # coco and custom results

        # Time-based logic to avoid blocking
        self.last_frame_time = time.time()  # Used to ensure non-blocking frame capture
        self.frame_interval = 1 / 15  # Capture a frame every 1/15 seconds (~15 FPS)

    def run(self):
        while True:
            current_time = time.time()

            # Check if enough time has passed to capture a new frame
            if current_time - self.last_frame_time >= self.frame_interval:
                try:
                    ret, frame = self.capture.read()
                except Exception as e:
                    self.logger.error(f"Frame capture exception: {e}")
                    time.sleep(0.5)  # Sleep for a short time to avoid high CPU usage
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
                self.prev_results[name] = det_result  # store results
                self.det_index = (self.det_index + 1) % len(self.detectors)

                try:
                    self.out_q.put_nowait((frame.copy(), self.prev_results.copy()))
                except queue.Full:
                    _ = self.out_q.get_nowait()
                    self.out_q.put_nowait((frame.copy(), self.prev_results.copy()))
                self.event.set()

                self.last_frame_time = current_time  # Update the last frame time

            # Allow other tasks to run while waiting for the next frame
            time.sleep(0.01)

class StateMachine:
    def __init__(self, cfg, frame_capture, yolo_detectors, monodepth_estimator,
                 path_planner, controller, pan_tilt_controller, user_io, detect_parking_slot_by_contour):
        self.cfg = cfg
        self.capture = frame_capture
        self.detectors = yolo_detectors
        self.depth_estimator = monodepth_estimator
        self.planner = path_planner
        self.ctrl = controller
        self.pan_tilt = pan_tilt_controller
        self.ui = user_io
        self.detect_parking_slot_by_contour = detect_parking_slot_by_contour

        self.state = State.SEARCH
        self.goal_slot = None
        self.current_pos = None
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
            logging.warning(f"'{homography_path}' 없음. 단위 행렬 사용.")
            self.homography_matrix = np.eye(3)

        # 픽셀→월드 변환 함수
        self.p2w = lambda x, y: pixel_to_world(x, y, self.homography_matrix)

    def run(self):
        self.ui.prompt_start()
        while True:
            if self.ui.wait_cancel(timeout=0):
                self.logger.info("User cancelled operation")
                break

            if not self.new_det_event.wait(timeout=0.1):  # Adjusting detection frequency
                continue
            self.new_det_event.clear()

            frame, detections = self.det_q.get()

            cv2.imshow("Live Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.logger.info("User pressed q → exiting")
                break

            # 상태에 따라 다른 처리를 담당하는 메서드 호출
            if self.state == State.SEARCH:
                self._search_step(frame)
            elif self.state == State.NAVIGATE:
                self._navigate_step(frame, detections)
            elif self.state == State.OBSTACLE_AVOID:
                self._avoid_step(frame, detections)
            elif self.state == State.WAIT:
                self._wait_step()
            elif self.state == State.FINAL_APPROACH:
                self._final_approach_step(frame, detections)
            elif self.state == State.COMPLETE:
                self._complete_step()
                break

        cv2.destroyAllWindows()
        self.ui.notify_complete()

    def _search_step(self, frame):
        self.wait_count = 0

        # 1) 슬롯 검출 + 거리 추정
        top_left, corners, slot_dist, annotated_frame = find_black_rect_and_distance(frame, debug=True)
        if top_left is not None and top_left[0] >= 20 and top_left[1] >= 20:
            # world 좌표 및 목표 거리
            self.goal_slot     = self.p2w(*top_left)
            self.goal_distance = slot_dist
            self.area_px    = corners                                # [(x,y),... 순서: TL,TR,BR,BL]
            self.area_world = [ self.p2w(x,y) for x,y in self.area_px ]
            # current_pos 추정 (y2 = top-left y)
            self.current_pos   = self.depth_estimator.estimate_current_position_from_y2(top_left[1])
            self.logger.info(f"[SEARCH] Slot at {top_left}, distance: {slot_dist:.2f}m, pos: {self.current_pos}")
        else:
            # 2) 슬롯 실패 → custom 모델(킥보드)만 사용
            scooter_dets = self.detectors["SCOOTER"].detect(frame)
            if not scooter_dets:
                self.logger.info("[SEARCH] No slot & no scooter detected, skipping.")
                return

            # 킥보드 중 가장 큰 bbox 선택
            scooter_dets.sort(key=lambda d: (d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]), reverse=True)
            bbox = scooter_dets[0]['bbox']  # (x1,y1,x2,y2)
            scooter_dist = self.depth_estimator.estimate_depth(bbox)
            if scooter_dist is None:
                self.logger.info("[SEARCH] Invalid scooter bbox, skipping.")
                return

            # 목표 거리 & current_pos
            self.goal_distance = scooter_dist
            self.current_pos   = self.depth_estimator.estimate_current_position_from_y2(bbox[3])
            self.logger.info(f"[SEARCH] Scooter fallback distance: {scooter_dist:.2f}m, pos: {self.current_pos}")

            # 킥보드에만 depth 주석
            annotated_frame = self.depth_estimator.annotate_detections(frame.copy(), scooter_dets)

        # 3) 결과 표시 & 상태 전환
        cv2.imshow("SEARCH Result", annotated_frame)
        self.state = State.NAVIGATE
        self.logger.info("[SEARCH] Transitioning to NAVIGATE")

    def _navigate_step(self, frame, detections):
        # 1) 장애물 체크
        for det in detections.get("custom", []):
            if self._is_obstacle_too_close(det):
                self.state = State.OBSTACLE_AVOID
                return

        # 2) 방향 결정 (필요하다면 방향 정보 사용)
        frame_h   = frame.shape[0]
        direction = self.planner.plan(self.current_pos[0], self.current_pos[1],
                                    frame_height=frame_h)
        if direction == "stop":
            self.state = State.OBSTACLE_AVOID
            return

        self._perform_navigation()

        custom = detections.get("custom", [])
        if custom:
            best     = max(custom, key=lambda d: (d['bbox'][2]-d['bbox'][0]) *
                                        (d['bbox'][3]-d['bbox'][1]))
            _, _, _, y2 = best['bbox']
            self.current_pos = self.depth_estimator.estimate_current_position_from_y2(y2)
        else:
            prev   = self.current_pos
            target = self.planner.pid_step(prev, self.goal_slot)
            dx, dy = target[0]-prev[0], target[1]-prev[1]
            dist   = math.hypot(dx, dy)
            if dist > 1e-6:
                ux, uy = dx/dist, dy/dist
                step   = self.cfg.get("movement_step", 0.1)
                self.current_pos = (prev[0]+ux*step, prev[1]+uy*step)

        # 5) 최종 접근 판정
        dxg, dyg = self.goal_slot[0]-self.current_pos[0], self.goal_slot[1]-self.current_pos[1]
        if math.hypot(dxg, dyg) < self.cfg.get("final_approach_threshold", 0.3):
            self.state = State.FINAL_APPROACH
            self.logger.info("[NAVIGATE] → FINAL_APPROACH")

    def _is_obstacle_too_close(self, detection):
        """Check if detected object is too close."""
        # 객체 탐지 후, depth 추정
        depth = self.depth_estimator.estimate_depth(detection["bbox"])
        
        if depth is None:
            return False  # 객체의 깊이를 추정할 수 없는 경우
        
        # 설정된 장애물 거리 임계값보다 가까운 경우 장애물로 간주
        obstacle_distance_threshold = self.cfg.get("obstacle_distance_threshold", 0.5)
        
        if depth < obstacle_distance_threshold:
            self.logger.info(f"[Obstacle] Object too close: depth={depth:.2f}m")
            return True  # 장애물이 너무 가까움
        
        return False  # 장애물 없음

    def _perform_navigation(self):
        """Actual navigation logic to move towards goal."""
        # 목표 위치와 현재 위치 차이 계산
        dx = self.goal_slot[0] - self.current_pos[0]
        dy = self.goal_slot[1] - self.current_pos[1]
        dist = math.hypot(dx, dy)

        # 설정된 최종 접근 거리 임계값보다 가까워졌을 때, final_approach 상태로 전환
        final_approach_threshold = self.cfg.get("final_approach_threshold", 0.3)
        if dist < final_approach_threshold:
            self.state = State.FINAL_APPROACH
            self.logger.info(f"[NAVIGATE] Within threshold distance: {dist:.2f}m, transitioning to FINAL_APPROACH")
            return

        # 경로 계획: PID 제어를 통해 목표 위치로 이동
        prev = self.current_pos
        target = self.planner.pid_step(prev, self.goal_slot)

        # 이동 명령
        self.ctrl.navigate_to(prev, target)
        self.logger.info(f"[NAVIGATE] Moving towards target: {target}")

    def _avoid_step(self, frame, detections):
        scan_cfg = self.cfg.get('avoid', {})
        scan_angles   = scan_cfg.get(
            'steer_scan_angles',
            [self.ctrl.ANGLE_LEFT, self.ctrl.ANGLE_RIGHT]
        )
        scan_delay    = scan_cfg.get('steer_scan_delay', 0.2)
        center_angle  = scan_cfg.get('steer_center_angle', self.ctrl.ANGLE_FORWARD)

        for ang in scan_angles:
            self.ctrl.set_angle(ang)
            time.sleep(scan_delay)

        self.ctrl.set_angle(center_angle)

        try:
            frame, detections = self.det_q.get(timeout=1.0)
        except queue.Empty:
            self.logger.warning("[AVOID] No detection data, returning to NAVIGATE.")
            self.state = State.NAVIGATE
            return

        custom_dets = detections.get('custom', [])
        if not custom_dets:
            self.logger.info("[AVOID] No obstacles detected, returning to NAVIGATE.")
            self.state = State.NAVIGATE
            return

        x1, y1, x2, y2 = custom_dets[0]['bbox']
        obs_px   = ((x1 + x2) / 2, (y1 + y2) / 2)
        obs_world = self.p2w(*obs_px)[:2]

        clearance = scan_cfg.get('clearance_pixels', 50)
        bounds    = np.array(self.area_world)
        waypoints = self.planner.replan_around(
            self.current_pos,
            self.goal_slot,
            obs_world,
            clearance,
            bounds
        )
        self.logger.info(f"[AVOID] New path: {waypoints}")

        for wp in waypoints:
            target = wp[:2]
            self.ctrl.navigate_to(self.current_pos, target)
            # 이동 완료 대기
            while not self.ctrl.update_navigation():
                time.sleep(0.01)
            self.current_pos = target

        self.ctrl.set_angle(center_angle)
        self.state = State.NAVIGATE
        self.logger.info("[AVOID] Path avoidance complete → NAVIGATE")

    def _wait_step(self):
        """Handle WAIT state when an obstacle is detected."""
        now = time.time()

        # 대기 시작 시간 기록
        if not hasattr(self, "wait_start_time"):
            self.wait_start_time = now
            self.logger.info("[WAIT] Obstacle detected, starting wait...")
            self.ctrl.stop()  # 모터 정지
            return

        # 대기 시간 1초가 지나면 프레임 처리
        if now - self.wait_start_time < 1.0:
            return

        ret, frame = self.capture.read()
        if not ret or frame is None:
            self.logger.warning("[WAIT] No frame detected, restarting wait...")
            self.wait_start_time = now
            return

        detections = {
            name: detector.detect(frame)
            for name, detector in self.detectors.items()
        }

        # 장애물 감지 여부 확인
        obstacle_detected = False
        frame_height = frame.shape[0]

        # custom 객체 및 coco 객체에 대해 장애물 감지
        for det in detections.get("custom", []):
            depth = self.depth_estimator.estimate_depth(det["bbox"])
            if depth and depth < self.cfg.get("obstacle_distance_threshold", 0.5):
                self.wait_count += 1
                self.logger.info(f"[WAIT] Custom obstacle detected. Wait count: {self.wait_count}")
                obstacle_detected = True
                break

        # coco 객체에 대해서도 동일한 감지
        if not obstacle_detected:
            for det in detections.get("coco", []):
                x1, y1, x2, y2 = det["bbox"]
                if (y2 - y1) / frame_height > self.cfg.get("obstacle_height_ratio_threshold", 0.7):
                    self.wait_count += 1
                    self.logger.info(f"[WAIT] Large COCO object detected. Wait count: {self.wait_count}")
                    obstacle_detected = True
                    break

        # 장애물이 사라지면 NAVIGATE로 돌아가기
        if not obstacle_detected:
            if self.wait_count >= 5:
                self.logger.info("[WAIT] No obstacles detected for 5 counts. Transitioning to NAVIGATE.")
                self.state = State.NAVIGATE
                del self.wait_start_time  # 초기화
            else:
                self.wait_start_time = now  # 다시 대기 시작
        else:
            self.logger.info("[WAIT] Obstacle detected. Continuing to wait.")

    def _final_approach_step(self, frame, detections):
        """Handle final approach for precise parking."""
        self.wait_count = 0

        # 목표와의 거리 계산
        dx = self.goal_slot[0] - self.current_pos[0]
        dy = self.goal_slot[1] - self.current_pos[1]
        dist = math.hypot(dx, dy)

        # tilt down을 한번만 호출하도록 처리
        if not hasattr(self, '_tilted_down') or not self._tilted_down:
            final_angle = self.cfg['pan_tilt'].get('final_tilt_angle', 10)
            self.pan_tilt.set_tilt(final_angle)
            self._tilted_down = True
            self.logger.info(f"[FINAL_APPROACH] Tilting down to {final_angle}°")

        # 목표 위치에 도달 시: 주차 완료
        tol = self.cfg.get('park_tolerance', 0.05)
        if dist < tol:
            stop_angle = self.cfg.get('stop_steering_angle', 90)
            self.ctrl.start_steering(stop_angle)
            self.ctrl.stop()
            self.pan_tilt.reset()

            self.logger.info(f"[FINAL_APPROACH] Parking completed within {tol}m")
            self.state = State.COMPLETE
            return

        # 최종 접근: 세밀한 이동
        step = min(self.cfg.get('final_step_size', 0.1), dist)
        ux, uy = dx / dist, dy / dist
        target_pos = (
            self.current_pos[0] + ux * step,
            self.current_pos[1] + uy * step
        )

        self.ctrl.navigate_to(self.current_pos, target_pos)
        self.current_pos = target_pos

        self.logger.info(f"[FINAL_APPROACH] Moving towards {target_pos} with step size {step:.3f}m")

    def _complete_step(self):
        """Handle completion of the parking task."""
        # 주차 완료 메시지 출력
        self.logger.info(f"Parked at slot {self.goal_slot}")

        # 주차 완료 후 목표 슬롯과 현재 위치를 초기화
        self.goal_slot = None
        self.current_pos = None
        
        # 시스템을 초기 상태로 복귀
        self.state = State.SEARCH

        # 차량의 속도 및 방향 제어를 멈추고, 카메라를 원위치로 복귀
        self.ctrl.stop()  # 모터 및 조향 제어를 멈추기
        self.pan_tilt.reset()  # 카메라 tilt 원위치 복귀

        # 필요하다면 UI나 외부 시스템에 완료 메시지 전달
        self.ui.notify_complete()  # UI에 주차 완료 알림

        # 로그 기록
        self.logger.info("[COMPLETE] Parking successfully completed. Returning to SEARCH state.")

        # 추가적으로 시스템 종료 또는 초기화가 필요한 경우 이를 처리하는 코드 추가 가능
        # 예: 자원을 해제하거나, 결과를 저장하는 등의 작업
        self.cleanup()

    def cleanup(self):
        """Clean up any resources and prepare for the next cycle."""
        # GPIO 및 하드웨어 자원 정리
        self.ctrl.cleanup()  # 모터 및 조향 제어 정리
        self.pan_tilt.release()  # 서보 모터 리소스 해제
        self.capture.release()  # 웹캠 리소스 해제
        self.logger.info("[CLEANUP] Resources cleaned up successfully.")
