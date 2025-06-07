import cv2
import time
import logging
import threading
import queue
import numpy as np
from enum import Enum, auto
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility.distance import euclidean_distance
from vision.slot_geometry import detect_parking_slot_by_contour 

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

        self.prev_results = {}  # coco와 custom 결과 유지

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
            name, detector = self.detectors[self.det_index]
            det_result = detector.detect(frame)
            self.prev_results[name] = det_result  # 결과 저장

            self.det_index = (self.det_index + 1) % len(self.detectors)
            try:
                self.out_q.put_nowait((frame.copy(), self.prev_results.copy()))
            except queue.Full:
                _ = self.out_q.get_nowait()
                self.out_q.put_nowait((frame.copy(), self.prev_results.copy()))
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
        self.current_pos = None
        self.logger = logging.getLogger(__name__)
        self.wait_count = 0

        self.det_q = queue.Queue(maxsize=1)
        self.new_det_event = threading.Event()
        self.det_worker = DetectionWorker(self.capture, self.detectors, self.det_q, self.new_det_event)
        self.det_worker.start()

    def run(self):
        self.ui.prompt_start()
        while True:
            if self.ui.wait_cancel(timeout=0):
                self.logger.info("User cancelled operation")
                break
            if not self.new_det_event.wait(timeout=0.1):
                continue
            self.new_det_event.clear()
            frame, detections = self.det_q.get()

            #self.ui.show_status(f"State: {self.state.name}")
            for name, det_list in detections.items():
                color = (255, 0, 0) if name == "coco" else (0, 255, 0)
                for det in det_list:
                    x1, y1, x2, y2 = det["bbox"]
                    conf = det["confidence"]
                    label = name.upper()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label}:{conf:.2f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.imshow("Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.logger.info("User pressed 'q' to cancel parking")
                break

            new_pos = self.ctrl.update_navigation()
            if new_pos is not None:
                self.current_pos = new_pos
            self.ctrl.update_steering()

            if self.ctrl.is_busy:
                if new_pos is not None:
                    self.current_pos = new_pos 
                continue

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
        self.wait_count = 0

        slot_center, annotated_frame = detect_parking_slot_by_contour(frame)
        if not slot_center or slot_center[0] < 20 or slot_center[1] < 20:
            self.logger.info(f"[SEARCH] 비정상적인 슬롯 중심 좌표 {slot_center} → 무효 처리")
            return

        world_slot = self.allocator.p2w(*slot_center)
        self.goal_slot = world_slot

        y2 = slot_center[1]
        self.current_pos = self.depth_estimator.estimate_current_position_from_y2(y2)
        self.logger.info(f"[SEARCH] 슬롯 중심 기반 현재 위치 추정 완료: {self.current_pos}")

        self.state = State.NAVIGATE
        self.logger.info("[SEARCH] 슬롯 중심 추정 완료 → NAVIGATE 전환")

    def _navigate_step(self, detections):
        if self.goal_slot is None or self.current_pos is None:
            self.logger.error("[NAVIGATE] goal_slot 또는 current_pos가 None입니다. NAVIGATE 중단")
            self.state = State.SEARCH
            return

        self.wait_count = 0

        # 장애물 체크
        for det in detections.get("custom", []):
            depth = self.depth_estimator.estimate_depth(det["bbox"])
            if depth is not None and depth < self.cfg.get("obstacle_distance_threshold", 0.5):
                self.ctrl.stop()
                self.logger.info(f"[Obstacle-Custom] depth={depth:.2f}m → WAIT 상태")
                self.state = State.OBSTACLE_AVOID
                return

        frame_height = self.capture.height if hasattr(self.capture, 'height') else 480
        for det in detections.get("coco", []):
            x1, y1, x2, y2 = det["bbox"]
            box_height = y2 - y1
            if box_height / frame_height > self.cfg.get("obstacle_height_ratio_threshold", 0.5):
                self.ctrl.stop()
                self.logger.info(f"[Obstacle-COCO] bbox height ratio={box_height/frame_height:.2f} → WAIT 상태")
                self.state = State.WAIT
                return

        dx = self.goal_slot[0] - self.current_pos[0]
        dy = self.goal_slot[1] - self.current_pos[1]
        dist = (dx**2 + dy**2)**0.5
        print(f"[NAVIGATE] 현재 위치: {self.current_pos} / 목표 위치: {self.goal_slot} / 거리: {dist:.3f}m")

        if dist < self.cfg.get("final_approach_threshold", 0.3):
            self.logger.info(f"[NAVIGATE] FINAL_APPROACH 거리 도달 (dist={dist:.2f}) → 상태 전환")
            self.state = State.FINAL_APPROACH
            return

        prev_pos = self.current_pos
        target_pos = self.planner.pid_step(prev_pos, self.goal_slot)

        dx_cmd = target_pos[0] - prev_pos[0]
        dy_cmd = target_pos[1] - prev_pos[1]
        dist_cmd = (dx_cmd**2 + dy_cmd**2)**0.5
        MIN_MOVEMENT_THRESHOLD = 0.01  # 1cm

        if dist_cmd < MIN_MOVEMENT_THRESHOLD:
            self.logger.warning("[NAVIGATE] 이동 명령 너무 작음 → 명령 생략")
            return

        if hasattr(self, "last_pos"):
            dir_prev = np.array(prev_pos) - np.array(self.last_pos)
            dir_now = np.array(target_pos) - np.array(prev_pos)
            dot = np.dot(dir_prev, dir_now)
            if dot < -0.001:  # 음의 내적이면 반대방향
                self.logger.warning("[NAVIGATE] 왕복 반복 감지 → 명령 무시")
                return

        # 이동 명령
        self.ctrl.navigate_to(prev_pos, target_pos) 
        self.last_nav_target = target_pos
        print(f"[NAVIGATE] 위치 갱신: {self.current_pos} / 거리: {dist:.2f}m")


    def _wait_step(self):
        now = time.time()

        # 처음 진입했을 때 시간 저장
        if not hasattr(self, "wait_start_time"):
            self.wait_start_time = now
            print("[WAIT] 장애물 감지로 정지. 2초 대기 시작...")
            self.ctrl.stop()  # 모터 정지
            return
        
        if now - self.wait_start_time < 1.0:
            return
        
        ret, frame = self.capture.read()
        if not ret or frame is None:
            print("[WAIT] 프레임 없음. 다시 대기.")
            self.wait_start_time = now
            return

        detections = {
            name: detector.detect(frame)
            for name, detector in self.detectors.items()
        }

        frame_height = frame.shape[0]
        obstacle_detected = False

        for det in detections.get("custom", []):
            depth = self.depth_estimator.estimate_depth(det["bbox"])
            print(f"[DEBUG] bbox={det['bbox']}, depth={depth}")
            if depth is not None and depth < self.cfg.get("obstacle_distance_threshold", 0.5):
                self.wait_count += 1
                print(f"[WAIT] 가까운 custom 객체 존재 → 대기 누적 {self.wait_count}회")
                obstacle_detected = True
                break

        if not obstacle_detected:
            for det in detections.get("coco", []):
                x1, y1, x2, y2 = det["bbox"]
                if (y2 - y1) / frame_height > self.cfg.get("obstacle_height_ratio_threshold", 0.7):
                    self.wait_count += 1
                    print(f"[WAIT] 큰 COCO 객체 존재 → 대기 누적 {self.wait_count}회")
                    obstacle_detected = True
                    break

        if obstacle_detected:
            if self.wait_count >= 5:
                print("[WAIT] 장애물 고정으로 판단 → OBSTACLE_AVOID 전환")
                self.state = State.OBSTACLE_AVOID
                del self.wait_start_time  # 초기화
            else:
                self.wait_start_time = now  # 다시 대기 시작
        else:
            print("[WAIT] 장애물 사라짐 → NAVIGATE 복귀")
            self.wait_count = 0
            self.state = State.NAVIGATE
            del self.wait_start_time  # 초기화

    def _avoid_step(self):
        try:
            frame, detections = self.det_q.get(timeout=1.0)
            dets = detections.get('custom', [])

            if not dets:
                self.logger.info("[AVOID] 감지된 custom 객체 없음 → 회피 생략")
                self.state = State.NAVIGATE
                return

            x1, y1, x2, y2 = dets[0]['bbox']
            obs_px = ((x1 + x2) / 2, (y1 + y2) / 2)
            obs_world = self.allocator.p2w(*obs_px)
            if len(obs_world) > 2:
                obs_world = obs_world[:2]

            bounds = np.array(self.allocator.area_world)
            self.path = self.planner.replan_around(
                self.current_pos, self.goal_slot, obs_world,
                self.cfg['clearance'], bounds
            )

            print("[AVOID] 회피 경로: ", self.path)

            for waypoint in self.path:
                wp = waypoint[:2] if len(waypoint) > 2 else waypoint
                self.ctrl.navigate_to(self.current_pos[:2], wp)
                print(f"[AVOID] Waypoint 이동: {wp}")
                self.current_pos = wp  # 위치 갱신

            print("[AVOID] 경로 이동 완료 → NAVIGATE 복귀")
            self.state = State.NAVIGATE

        except queue.Empty:
            self.logger.warning("[AVOID] 감지 큐가 비어 있음. 회피 생략")
        except Exception as e:
            self.logger.error(f"[AVOID] 예외 발생: {e}")
            self.state = State.NAVIGATE

    def _final_approach_step(self, detections):
        self.wait_count = 0

        # 현재 위치와 목표 주차 슬롯 간의 거리 계산
        if euclidean_distance(*self.current_pos, *self.goal_slot) < 0.05:
            self.ctrl.stop()
            self.logger.info("[FINAL_APPROACH] 위치상 주차 완료로 판단 → COMPLETE 전환")
            self.state = State.COMPLETE
            return

        # 틸트 각도 설정
        final_angle = self.cfg['pan_tilt'].get('final_tilt_angle', 10)
        self.pan_tilt.set_tilt(final_angle)  # set_tilt 함수 사용

        # 목표 방향으로 이동 벡터 계산
        dx = self.goal_slot[0] - self.current_pos[0]
        dy = self.goal_slot[1] - self.current_pos[1]
        norm = (dx**2 + dy**2)**0.5
        unit_vec = (dx / norm, dy / norm)

        # 목표 위치로 0.5m 전진
        target_pos = (
            self.current_pos[0] + unit_vec[0] * 0.5,
            self.current_pos[1] + unit_vec[1] * 0.5
        )

        self.ctrl.navigate_to(self.current_pos, target_pos)
        self.current_pos = target_pos

    def _complete_step(self):
        self.logger.info(f"Parked at slot {self.goal_slot}")
        self.state = State.SEARCH
        self.goal_slot = None
        self.current_pos = None
