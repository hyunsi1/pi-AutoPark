import math
import time
import logging

class Controller:
    """
    PID 기반 속도 및 조향 제어 스텁

    - navigate_to(): 지정한 waypoint로 이동
    - stop(): 즉시 정지
    """
    def __init__(self, max_speed: float = 1.0, turn_speed: float = 0.5):
        # max_speed: 전진 최대 속도 (m/s)
        # turn_speed: 회전 속도 비율
        self.max_speed = max_speed
        self.turn_speed = turn_speed
        logging.info(f"Controller initialized (max_speed={max_speed} m/s)")

    def navigate_to(self, current: tuple, target: tuple):
        """
        현재 위치에서 목표 좌표로 전진 및 조향만으로 주행

        Args:
            current: (x, y) 현재 위치
            target:  (x, y) 목표 위치
        """
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        distance = math.hypot(dx, dy)
        if distance < 0.01:
            logging.info("Already at waypoint")
            return

        # 목표 각도 계산
        desired_heading = math.degrees(math.atan2(dy, dx))
        logging.info(f"Navigating: distance={distance:.2f}m, heading={desired_heading:.1f}°")

        # 스티어링만으로 유턴·회전 구현
        # 실제 로봇: set_steering(desired_heading)
        turn_time = abs(desired_heading) / (self.turn_speed * 360)  # 비율에 따른 회전 시간 계산
        logging.debug(f"Turning toward heading: {desired_heading:.1f}° (turn_time={turn_time:.2f}s)")
        time.sleep(turn_time)

        # 전진
        # 실제 로봇: set_throttle(self.max_speed)
        move_time = distance / self.max_speed
        logging.debug(f"Driving forward: {distance:.2f}m (move_time={move_time:.2f}s)")
        time.sleep(move_time)
        logging.info("Reached waypoint")

    def stop(self):
        """
        차를 즉시 정지시키는 메서드
        """
        logging.info("Controller: stop called. Cutting throttle.")
        # 실제 로봇: set_throttle(0)
