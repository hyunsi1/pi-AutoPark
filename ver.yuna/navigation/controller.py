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
        self.steer_pin = steer_pin
        self.motor_pin = motor_pin
        GPIO.setmode(GPIO.BCM)

        # 조향 및 구동 PWM 설정
        GPIO.setup(self.steer_pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self.steer_pin, freq)
        self.pwm.start(0)

        GPIO.setup(self.motor_pin, GPIO.OUT)
        self.motor_pwm = GPIO.PWM(self.motor_pin, freq)
        self.motor_pwm.start(0)

        # 상태 변수 먼저 초기화
        self.is_steering = False
        self.steer_start_time = 0
        self.steer_duration = 1.3

        self.is_moving = False
        self.move_start_time = 0
        self.move_duration = 1.5

        # 초기값 설정 이후 호출
        self.cur_angle = 90
        self.set_speed(0)
        self.pending_speed = 0
        self.start_steering(self.cur_angle)
        self.last_target_pos = None 

        print(f"[Init] PWM started on steer={steer_pin}, motor={motor_pin} at {freq}Hz")

    @property
    def is_busy(self):
        return self.is_steering or self.is_moving

    def start_steering(self, angle):
        if self.is_steering:
            return
        angle = max(0, min(180, angle))
        duty = 2 + (angle / 18)
        self.pwm.ChangeDutyCycle(duty)
        self.cur_angle = angle
        self.steer_start_time = time.time()
        self.is_steering = True
        logging.debug(f"[StartSteering] angle={angle}°, duty={duty:.2f}%")

    def update_steering(self):
        if self.is_steering and time.time() - self.steer_start_time >= self.steer_duration:
            self.pwm.ChangeDutyCycle(0)
            self.is_steering = False
            logging.debug("[UpdateSteering] completed")
            return True
        return False

    def set_speed(self, duty):
        duty = max(0, min(100, duty))
        self.motor_pwm.ChangeDutyCycle(duty)
        logging.debug(f"[SetSpeed] {duty:.1f}%")

    def navigate_to(self, cur_pos, target_pos):
        dx = target_pos[0] - cur_pos[0]
        dy = target_pos[1] - cur_pos[1]
        if abs(dx) < 1e-4 and abs(dy) < 1e-4:
            return False
        steering_angle = max(0, min(180, 90 + math.degrees(math.atan2(dy, dx))))
        self.start_steering(steering_angle)
        self.pending_speed = 30
        self.last_target_pos = target_pos
        logging.debug(f"[Navigate] {cur_pos} → {target_pos} → θ={steering_angle:.1f}°")
        return True

    def update_navigation(self):
        if self.is_steering:
            if self.update_steering():
                self.set_speed(self.pending_speed)
                self.move_start_time = time.time()
                self.is_moving = True
        elif self.is_moving:
            if time.time() - self.move_start_time >= self.move_duration:
                self.stop()
                self.is_moving = False
                logging.debug("[UpdateNavigation] movement complete")
                return self.last_target_pos
        return None

    def stop(self):
        self.pwm.ChangeDutyCycle(0)
        self.motor_pwm.ChangeDutyCycle(0)
        logging.debug("[Stop] all PWM off")

    def cleanup(self):
        self.stop()
        self.pwm.stop()
        self.motor_pwm.stop()
        self.GPIO.cleanup()
        logging.info("[Cleanup] GPIO cleaned up")