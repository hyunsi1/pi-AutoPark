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
from vision.goal_setter import GoalSetter
from vision.final_approach_helper import find_left_reference, steering, count_front_lines

class State(Enum):
    SEARCH = auto()
    REPOSITION = auto()  
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
                 goal_setter, path_planner, controller, pan_tilt_controller, user_io):

        self.cfg = cfg
        self.capture = frame_capture
        self.detectors = yolo_detectors
        self.depth_estimator = monodepth_estimator
        self.goal_setter = goal_setter
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

            # new_pos = self.ctrl.update_navigation()
            # if new_pos is not None:
            #     self.current_pos = new_pos
            # self.ctrl.update_steering()

            # if self.ctrl.is_busy:
            #     if new_pos is not None:
            #         self.current_pos = new_pos 
            #     continue

            if self.state == State.SEARCH:
                self._search_step(frame, detections)
            elif self.state == State.REPOSITION:
                self._reposition_step()
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

    def _search_step(self, frame, detections):
        self.wait_count = 0

        boxes = [d["bbox"] for d in detections.get("custom", [])]

        if boxes:
            ar_vals = [(y2 - y1) / max(1, x2 - x1) for x1, y1, x2, y2 in boxes]
            side_view = np.mean(ar_vals) < 1.4
        else:
            side_view = False

        empty_view = (len(boxes) == 0)

        if side_view or empty_view:
            self.logger.info("[SEARCH] 측면 시야 또는 빈 화면 → REPOSITION")
            self._side_view_flag = True
            self.state = State.REPOSITION
            return

        goal, _, mode = self.goal_setter.get_goal_point(frame, boxes)
        if goal is None:
            self.logger.info("[SEARCH] 목표점 없음")
            return

        self.parking_mode = mode
        self.goal_slot = goal
        self.current_pos = self.depth_estimator.estimate_current_position_world()

        if mode == "parking":
            self.pan_tilt.reset()

        self.state = State.NAVIGATE
        self.logger.info(f"[SEARCH] mode={mode}, goal={self.goal_slot} → NAVIGATE")

    def _reposition_step(self):
        SIDE_BACK_TIME = 1.2       

        if not hasattr(self, "_repo_inited"):
            # 항상 ‘왼쪽 뒤’ 대각 후진
            self.ctrl.set_angle(100)
            self.ctrl.set_speed(50, reverse=True, time_duration=0.5)
            self._repo_dur   = SIDE_BACK_TIME
            self._repo_start = time.time()
            self._repo_inited = True
            self.logger.info("[REPO] 후진 시작")

        # 지정 시간 경과 → 정지 & NAVIGATE 복귀
        if time.time() - self._repo_start > self._repo_dur:
            self.ctrl.stop()
            self.ctrl.set_angle(self.ctrl.ANGLE_FORWARD)  # 핸들 중앙
            del self._repo_inited, self._side_view_flag   # 플래그 초기화
            self.state = State.SEARCH
            self.logger.info("[REPO] 완료 → 다시 SEARCH")


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
            if self.wait_count >= 1:
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
            obs_world = self.goal_setter.pixel_to_world(obs_px)  # 픽셀→월드 변환!
            self.path = self.planner.replan_around(
                self.current_pos, self.goal_slot, obs_world,
                self.cfg['clearance'], frame.shape[:2]
            )
            for wp in self.path:
                self.ctrl.navigate_to(self.current_pos[:2], wp)
                self.current_pos = wp
            self.state = State.NAVIGATE
        except queue.Empty:
            self.logger.warning("[AVOID] 감지 큐가 비어 있음. 회피 생략")
        except Exception as e:
            self.logger.error(f"[AVOID] 예외 발생: {e}")
            self.state = State.NAVIGATE

    def _final_approach_step(self, detections):
        """최종 접근: 왼쪽 기준선 맞추며 직진/후진 반복 + 앞 기준선 1개 되면 멈춤"""
        # 초기 틸트
        self.pan_tilt.tilt(45)
        self.logger.info("[FINAL] tilt 45°로 초기화 완료")
        time.sleep(1)

        # 첫 번째 왼쪽 기준선 검출
        ret, frame = self.capture.read()
        if not ret:
            self.logger.error("[FINAL] 초기 프레임 읽기 실패!")
            return

        slope = find_left_reference(frame, min_length=500, slope_thresh=1.0)
        if slope is not None:
            servo = steering(slope)
            self.ctrl.set_angle(servo)
            self.logger.info(f"[FINAL] 초기 왼쪽 기준선 slope={slope:.2f}, servo={servo}")

        # 첫 전진/후진
        self.ctrl.set_speed(20, reverse=False, sleep_duration=0.3)
        self.logger.info("[FINAL] 첫 번째 전진")
        time.sleep(0.3)
        self.ctrl.stop()
        time.sleep(0.1)

        self.ctrl.set_speed(20, reverse=True, sleep_duration=0.3)
        self.logger.info("[FINAL] 첫 번째 후진")
        time.sleep(0.3)
        self.ctrl.stop()
        time.sleep(0.1)

        # 두 번째 왼쪽 기준선 검출
        ret, frame = self.capture.read()
        if not ret:
            self.logger.error("[FINAL] 두 번째 프레임 읽기 실패!")
            return

        slope = find_left_reference(frame, min_length=500, slope_thresh=1.0)
        if slope is not None:
            servo = steering(slope)
            self.ctrl.set_angle(servo)
            self.logger.info(f"[FINAL] 두 번째 왼쪽 기준선 slope={slope:.2f}, servo={servo}")

        # 두 번째 전진/후진
        self.ctrl.set_speed(20, reverse=False, sleep_duration=0.3)
        self.logger.info("[FINAL] 두 번째 전진")
        time.sleep(0.3)
        self.ctrl.stop()
        time.sleep(0.1)

        self.ctrl.set_speed(20, reverse=True, sleep_duration=0.3)
        self.logger.info("[FINAL] 두 번째 후진")
        time.sleep(0.3)
        self.ctrl.stop()
        time.sleep(0.1)

        # 틸트 60도로 올리고 조금씩 전진하며 기준선 개수 확인
        self.pan_tilt.tilt(60)
        self.logger.info("[FINAL] tilt 60°로 전환")
        time.sleep(1)

        while True:
            # 조금씩 전진
            self.ctrl.set_speed(20, reverse=False, sleep_duration=0.1)
            self.logger.info("[FINAL] 루프: 전진 중 (0.1s)")
            time.sleep(0.5)
            self.ctrl.stop()
            time.sleep(0.1)

            # 앞 기준선 개수 확인
            ret, frame = self.capture.read()
            if not ret:
                self.logger.error("[FINAL] 루프 프레임 읽기 실패!")
                continue

            count_lines = count_front_lines(frame)
            self.logger.info(f"[FINAL] 루프: 앞 기준선 개수 = {count_lines}")

            # 앞 기준선이 1개가 되면 멈추기
            if count_lines == 1:
                self.ctrl.stop()
                self.logger.info("[FINAL] 앞 기준선 1개 됨 → 멈춤!")
                break

        # 마지막으로 살짝 앞으로 전진
        self.ctrl.set_speed(20, reverse=False, sleep_duration=0.2)
        self.logger.info("[FINAL] 마지막 전진")
        time.sleep(0.5)
        self.ctrl.stop()
        time.sleep(0.1)

        self.logger.info("[FINAL] 최종 접근 완료. 상태 COMPLETE로 전환")
        self.state = State.COMPLETE

    def _complete_step(self):
        self.logger.info(f"Parked at slot {self.goal_slot}")
        self.state = State.SEARCH
        self.goal_slot = None
        self.current_pos = None
