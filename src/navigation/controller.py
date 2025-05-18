import math
import time
import logging
import os
try:
    import RPi.GPIO as GPIO
except (ImportError, RuntimeError):
    # 개발 PC(윈도우 등)에서는 더미 모듈 정의
    from unittest import mock
    GPIO = mock.MagicMock()


SIMULATE_FAST = os.getenv("SIMULATE_FAST", "0") == "1"

class Controller:
    """
    PID 기반 속도 및 조향 제어 스텁

    - navigate_to(): 지정한 waypoint로 이동
    - stop(): 즉시 정지
    """
    def __init__(self, pwm_pin=18, freq=50):
        """
        pwm_pin: PWM 제어용 GPIO 핀 번호 (BCM 기준)
        freq: PWM 신호 주파수 (서보는 일반적으로 50Hz)
        """
        self.pwm_pin = pwm_pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pwm_pin, GPIO.OUT)

        self.pwm = GPIO.PWM(self.pwm_pin, freq)
        self.pwm.start(0)  # 초기에는 0% 듀티사이클
        self.cur_angle = 90  # 초기 각도 (중립)
        self.set_angle(self.cur_angle)
        print(f"[Init] PWM started on GPIO{pwm_pin} at {freq}Hz")

    def set_angle(self, angle):
        """
        주어진 각도로 서보 모터 조향
        angle: 0 ~ 180도
        """
        angle = max(0, min(180, angle))  # 안전 범위 제한
        duty = 2 + (angle / 18)  # 듀티 변환 공식: 0도 → 2%, 180도 → 12%
        self.pwm.ChangeDutyCycle(duty)
        time.sleep(0.3)  # 서보 이동 시간
        self.pwm.ChangeDutyCycle(0)  # 떨림 방지 (서보 전류 차단)
        self.cur_angle = angle
        print(f"[SetAngle] angle={angle}°, duty={duty:.2f}%")

    def navigate_to(self, cur_pos, target_pos):
        """
        현재 위치에서 목표 위치로 향하는 방향 각도를 계산하여 조향
        """
        dx = target_pos[0] - cur_pos[0]
        dy = target_pos[1] - cur_pos[1]

        if dx == 0 and dy == 0:
            print("[Navigate] 현재 위치와 목표 위치가 동일. 조향 생략")
            return

        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        # -90도 ~ +90도 범위를 0도 ~ 180도로 맵핑
        steering_angle = 90 + angle_deg
        steering_angle = max(0, min(180, steering_angle))

        self.set_angle(steering_angle)
        print(f"[Navigate] {cur_pos} → {target_pos} → θ = {steering_angle:.1f}°")

    def stop(self):
        """
        PWM 출력을 0으로 설정해 모터를 정지시킴
        """
        self.pwm.ChangeDutyCycle(0)
        print("[Stop] PWM 출력 중지")

    def cleanup(self):
        """
        GPIO 자원 해제
        """
        self.stop()
        self.pwm.stop()
        GPIO.cleanup()
        print("[Cleanup] GPIO 해제 완료")
