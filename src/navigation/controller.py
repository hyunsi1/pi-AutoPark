import math
import time
import logging
import os
try:
    import RPi.GPIO as GPIO
except (ImportError, RuntimeError):
    from unittest import mock
    GPIO = mock.MagicMock()

SIMULATE_FAST = os.getenv("SIMULATE_FAST", "0") == "1"

class Controller:
    def __init__(self, steer_pin=18, motor_pin=19, freq=50):
        """
        steer_pin: 조향용 서보 PWM 핀 (BCM 번호)
        motor_pin: 구동용 DC 모터 PWM 핀
        freq: PWM 주파수 (보통 50Hz)
        """
        self.steer_pin = steer_pin
        self.motor_pin = motor_pin
        GPIO.setmode(GPIO.BCM)

        # 조향 PWM (서보 모터)
        GPIO.setup(self.steer_pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self.steer_pin, freq)
        self.pwm.start(0)

        # 구동 PWM (DC 모터)
        GPIO.setup(self.motor_pin, GPIO.OUT)
        self.motor_pwm = GPIO.PWM(self.motor_pin, freq)
        self.motor_pwm.start(0)

        self.cur_angle = 90
        self.set_angle(self.cur_angle)
        print(f"[Init] PWM started on steer={steer_pin}, motor={motor_pin} at {freq}Hz")

    def set_angle(self, angle):
        angle = max(0, min(180, angle))
        duty = 2 + (angle / 18)  # 보통 서보 모터용
        self.pwm.ChangeDutyCycle(duty)
        time.sleep(0.3)
        self.pwm.ChangeDutyCycle(0)
        self.cur_angle = angle
        print(f"[SetAngle] angle={angle}°, duty={duty:.2f}%")

    def set_speed(self, duty):
        duty = max(0, min(100, duty))
        self.motor_pwm.ChangeDutyCycle(duty)
        print(f"[SetSpeed] duty={duty:.1f}%")

    def navigate_to(self, cur_pos, target_pos):
        dx = target_pos[0] - cur_pos[0]
        dy = target_pos[1] - cur_pos[1]

        if dx == 0 and dy == 0:
            print("[Navigate] 현재 위치와 목표 위치가 동일. 조향 생략")
            return

        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        steering_angle = 90 + angle_deg
        steering_angle = max(0, min(180, steering_angle))

        self.set_angle(steering_angle)
        self.set_speed(30)
        print(f"[Navigate] {cur_pos} → {target_pos} → θ = {steering_angle:.1f}°")

    def stop(self):
        self.pwm.ChangeDutyCycle(0)
        self.motor_pwm.ChangeDutyCycle(0)
        print("[Stop] PWM 출력 중지")

    def cleanup(self):
        self.stop()
        self.pwm.stop()
        self.motor_pwm.stop()
        GPIO.cleanup()
        print("[Cleanup] GPIO 해제 완료")
