# for BTS7960 DC 모터 드라이버
import time
import logging
import RPi.GPIO as GPIO
from DFRobot_Expansion_Board_IIC import DFRobot_Expansion_Board_IIC as Board

class KickboardController:
    def __init__(self, steer_channel=0, in1_channel=1, in2_channel=2, ena_gpio=16, i2c_bus=1, addr=0x10):
        self.steer_ch = steer_channel      # 서보 채널
        self.in1_ch = in1_channel          # DC모터 IN1 채널
        self.in2_ch = in2_channel          # DC모터 IN2 채널
        self.ena_gpio = ena_gpio           # Enable 핀

        # 각도 정의
        self.ANGLE_LEFT = 120
        self.ANGLE_RIGHT = 30
        self.ANGLE_FORWARD = 75

        # 보드 초기화
        self.board = Board(i2c_bus, addr)
        while self.board.begin() != self.board.STA_OK:
            print("[Init] 보드 초기화 실패, 재시도 중...")
            time.sleep(1)
        print("[Init] 보드 초기화 성공")

        # PWM 설정
        self.board.set_pwm_enable()
        self.board.set_pwm_frequency(1000)

        # GPIO ENA 핀 초기화
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.ena_gpio, GPIO.OUT)
        GPIO.output(self.ena_gpio, GPIO.HIGH)  # 항상 enable

        # 초기 상태
        self.set_angle(self.ANGLE_FORWARD)
        self.stop()
        print(f"[Init] steer={steer_channel}, in1={in1_channel}, in2={in2_channel}, ena={ena_gpio}")

    def angle_to_duty(self, angle):
        return 2.5 + (angle / 180.0) * 10.0

    def set_angle(self, angle):
        angle = max(0, min(180, angle))
        duty = self.angle_to_duty(angle)
        self.board.set_pwm_duty(self.steer_ch, duty)
        print(f"[Steering] angle={angle}°, duty={duty:.1f}%")

    def set_speed(self, speed_percent, reverse=False):
        """
        speed_percent: 0~100
        reverse: True면 후진, False면 전진
        """
        duty = max(0, min(100, speed_percent))
        if reverse:
            self.board.set_pwm_duty([self.in1_ch], 0)
            self.board.set_pwm_duty([self.in2_ch], duty)
            print(f"[Motor] ← 후진, IN1=0%, IN2={duty:.1f}%")
        else:
            self.board.set_pwm_duty([self.in1_ch], duty)
            self.board.set_pwm_duty([self.in2_ch], 0)
            print(f"[Motor] → 전진, IN1={duty:.1f}%, IN2=0%")

    def stop(self):
        self.board.set_pwm_duty([self.in1_ch], 0)
        self.board.set_pwm_duty([self.in2_ch], 0)
        print("[Stop] 모터 정지")

    def navigate(self, direction: str, speed=30):
        direction = direction.lower()
        if direction == "left":
            self.set_angle(self.ANGLE_LEFT)
            self.set_speed(speed, reverse=False)
            print("[Navigate] 좌회전")
        elif direction == "right":
            self.set_angle(self.ANGLE_RIGHT)
            self.set_speed(speed, reverse=False)
            print("[Navigate] 우회전")
        elif direction == "forward":
            self.set_angle(self.ANGLE_FORWARD)
            self.set_speed(speed, reverse=False)
            print("[Navigate] 직진")
        elif direction == "backward":
            self.set_angle(self.ANGLE_FORWARD)
            self.set_speed(speed, reverse=True)
            print("[Navigate] 후진")
        elif direction == "stop":
            self.stop()
            print("[Navigate] 정지")
        else:
            print(f"[Navigate] 잘못된 입력: {direction}")

    def cleanup(self):
        self.stop()
        GPIO.cleanup()
        print("[Cleanup] GPIO 정리 완료")

'''
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
'''
