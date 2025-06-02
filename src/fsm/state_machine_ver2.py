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
from vision.slot_geometry import find_parking_slot_center, detect_parking_lines

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
                self._search_step()
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

    def _search_step(self):
        self.wait_count = 0

        # 최신 프레임 받아오기
        try:
            frame, _ = self.det_q.get(timeout=1.0)
        except queue.Empty:
            self.logger.warning("[SEARCH] 감지 큐가 비어 있음 → 재시도")
            return

        # Hough Line을 통한 라인 검출
        lines, _ = detect_parking_lines(frame)

        # slot_center 계산 시 라인이 적으면 무시
        if lines is None or len(lines) < 4:
            self.logger.info(f"[SEARCH] 검출된 라인 부족 (n={len(lines) if lines else 0}) → 슬롯 검출 불가")
            # 추가: 장애물로 인해 시야가 가려진 경우 → WAIT 상태로 전환
            detections = {
                name: detector.detect(frame)
                for name, detector in self.detectors.items()
            }
            frame_height = frame.shape[0]
            for det in detections.get("coco", []):
                x1, y1, x2, y2 = det["bbox"]
                if (y2 - y1) / frame_height > self.cfg.get("obstacle_height_ratio_threshold", 0.5):
                    self.logger.info("[SEARCH] 시야를 가리는 장애물 존재 → WAIT 전환")
                    self.ctrl.stop()
                    self.state = State.WAIT
                    return
            return  # 장애물은 없지만 라인 부족 → 다음 루프로 탐색 유지

        # 슬롯 중심 계산
        slot_center = find_parking_slot_center(lines, frame.shape)
        print(f"[DEBUG] slot_center = {slot_center}")

        # 슬롯 중심 유효성 검증
        if not slot_center or slot_center[0] < 20 or slot_center[1] < 20:
            self.logger.info(f"[SEARCH] 비정상적인 슬롯 중심 좌표 {slot_center} → 무효 처리")
            return

        # 픽셀 좌표 → 월드 좌표
        world_slot = self.allocator.p2w(*slot_center)

        # 주차 슬롯 중심 기준 0.5m 뒤에서 접근 시작
        self.goal_slot = world_slot
        self.current_pos = (
            world_slot[0] - 0.5,
            world_slot[1]
        )

        self.state = State.NAVIGATE
        self.logger.info("[SEARCH] 슬롯 중심 추정 완료 → NAVIGATE 전환")


    def _navigate_step(self, detections):
        if self.goal_slot is None or self.current_pos is None:
            self.logger.error("[NAVIGATE] goal_slot 또는 current_pos가 None입니다. NAVIGATE 중단")
            self.state = State.SEARCH  # 안전하게 초기화 상태로 돌리기
            return
        self.wait_count = 0
        frame_height = self.capture.height if hasattr(self.capture, 'height') else 480

        # custom: Monodepth로 거리 기준 판단
        for det in detections.get("custom", []):
            depth = self.depth_estimator.estimate_depth(det["bbox"])
            if depth is not None and depth < self.cfg.get("obstacle_distance_threshold", 0.5):
                self.ctrl.stop()
                self.logger.info(f"[Obstacle-Custom] depth={depth:.2f}m → WAIT 상태")
                self.state = State.WAIT
                return

        # coco: 화면 내 비율 기준 판단
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
        dist = (dx**2 + dy**2) ** 0.5

        if dist < self.cfg.get("final_approach_threshold", 0.3):
            self.state = State.FINAL_APPROACH
            return

        norm = (dx**2 + dy**2)**0.5
        if norm < 0.5:
            self.logger.info("[NAVIGATE] 현재 위치와 목표 위치가 동일. 조향 생략")
            return

        if norm <= 1.0:
            self.logger.info("[NAVIGATE] 목표와 너무 가까움 → current_pos를 goal_slot로 강제 설정")
            self.ctrl.navigate_to(self.current_pos, self.goal_slot)
            self.current_pos = self.goal_slot
            self.state = State.FINAL_APPROACH
            return
        
        unit_vec = (dx / norm, dy / norm)
        target_pos = (
            self.current_pos[0] + unit_vec[0] * 1.0,
            self.current_pos[1] + unit_vec[1] * 1.0
        )

        self.ctrl.navigate_to(self.current_pos, target_pos)
        self.pan_tilt.set_tilt(0)
        self.current_pos = target_pos

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

        frame_height = frame.shape[0]

        for det in detections.get("custom", []):
            depth = self.depth_estimator.estimate_depth(det["bbox"])
            if depth is not None and depth < self.cfg.get("obstacle_distance_threshold", 0.5):
                self.wait_count += 1
                print(f"[WAIT] 가까운 custom 객체 존재 → 대기 누적 {self.wait_count}회")
                if self.wait_count >= 5:
                    print("[WAIT] 장애물 고정으로 판단 → OBSTACLE_AVOID 전환")
                    self.state = State.OBSTACLE_AVOID
                return

        for det in detections.get("coco", []):
            x1, y1, x2, y2 = det["bbox"]
            if (y2 - y1) / frame_height > self.cfg.get("obstacle_height_ratio_threshold", 0.5):
                self.wait_count += 1
                print(f"[WAIT] 큰 COCO 객체 존재 → 대기 누적 {self.wait_count}회")
                if self.wait_count >= 5:
                    print("[WAIT] 장애물 고정으로 판단 → OBSTACLE_AVOID 전환")
                    self.state = State.OBSTACLE_AVOID
                return

        print("[WAIT] 장애물 사라짐 → NAVIGATE 복귀 및 대기 회수 초기화")
        self.wait_count = 0
        self.state = State.NAVIGATE

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
                obs_world = obs_world[:2]  # ⚠️ 2D로 변환

            bounds = np.array(self.allocator.area_world)
            self.path = self.planner.replan_around(
                self.current_pos, self.goal_slot, obs_world,
                self.cfg['clearance'], bounds
            )

            self.logger.info("[AVOID] 회피 경로 재계획 완료 → NAVIGATE 전환")
            self.state = State.NAVIGATE

        except queue.Empty:
            self.logger.warning("[AVOID] 감지 큐가 비어 있음. 회피 생략")
        except Exception as e:
            self.logger.error(f"[AVOID] 예외 발생: {e}")
            self.state = State.NAVIGATE

    def _final_approach_step(self, detections):
        self.wait_count =0
        if euclidean_distance(*self.current_pos, *self.goal_slot) < 0.05:
            self.ctrl.stop()
            self.logger.info("[FINAL_APPROACH] 위치상 주차 완료로 판단 → COMPLETE 전환")
            self.state = State.COMPLETE
            return
        self.pan_tilt.set_tilt(self.cfg['pan_tilt']['final_tilt_angle'])
        dx = self.goal_slot[0] - self.current_pos[0]
        dy = self.goal_slot[1] - self.current_pos[1]
        norm = (dx**2 + dy**2)**0.5
        unit_vec = (dx / norm, dy / norm)
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
