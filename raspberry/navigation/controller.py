import time
import logging
import math
import RPi.GPIO as GPIO
from DFRobot_RaspberryPi_Expansion_Board import DFRobot_Expansion_Board_IIC as Board
from DFRobot_RaspberryPi_Expansion_Board import DFRobot_Expansion_Board_Servo as Servo

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
        self.start_steering(self.cur_angle)  # 이 메서드는 위 상태 변수들이 초기화되어야 안전

        print(f"[Init] PWM started on steer={steer_pin}, motor={motor_pin} at {freq}Hz")


    @property
    def is_busy(self):
        return self.is_steering or self.is_moving

    # --- 조향 제어 ---
    def start_steering(self, angle):
        if self.is_steering:
            return 
        angle = max(0, min(180, angle))
        duty = 2 + (angle / 18)
        self.pwm.ChangeDutyCycle(duty)
        self.cur_angle = angle
        self.steer_start_time = time.time()
        self.is_steering = True
        print(f"[StartSteering] angle={angle}°, duty={duty:.2f}%")

    def update_steering(self):
        if self.is_steering and time.time() - self.steer_start_time >= self.steer_duration:
            self.pwm.ChangeDutyCycle(0)  # 서보 모터 끄기
            self.is_steering = False
            print("[UpdateSteering] 조향 완료")
            return True
        return False

    # --- 속도 제어 ---
    def set_speed(self, duty):
        duty = max(0, min(100, duty))
        self.motor_pwm.ChangeDutyCycle(duty)
        print(f"[SetSpeed] duty={duty:.1f}%")

    # --- 목표 위치로 이동 시작 ---
    def navigate_to(self, cur_pos, target_pos):
        dx = target_pos[0] - cur_pos[0]
        dy = target_pos[1] - cur_pos[1]

        if dx == 0 and dy == 0:
            print("[Navigate] 현재 위치와 목표 위치가 동일. 조향 생략")
            return False

        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        steering_angle = 90 + angle_deg
        steering_angle = max(0, min(180, steering_angle))

        self.start_steering(steering_angle)
        self.pending_speed = 30  # 조향 완료 후 속도 적용 예정
        print(f"[Navigate] {cur_pos} → {target_pos} → θ = {steering_angle:.1f}°")
        return True

    # --- FSM에서 주기적으로 호출 ---
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
                print("[UpdateNavigation] 이동 완료")
                return True  # 이동 완료
        return False  # 아직 진행 중

    # --- 방향 키워드로 이동 (vision 기반) ---
    def navigate_direction(self, direction: str):
        if direction == "left":
            self.start_steering(60)
        elif direction == "right":
            self.start_steering(120)
        elif direction == "forward":
            self.start_steering(90)
        elif direction == "stop":
            self.stop()
            return

        self.set_speed(30)
        self.move_start_time = time.time()
        self.is_moving = True
        print(f"[NavigateDir] {direction.capitalize()} started")

    def update_direction(self):
        return self.update_navigation()

    # --- 정지 ---
    def stop(self):
        self.pwm.ChangeDutyCycle(0)
        self.motor_pwm.ChangeDutyCycle(0)
        print("[Stop] PWM 출력 중지")

    # --- 종료 시 정리 ---
    def cleanup(self):
        self.stop()
        self.pwm.stop()
        self.motor_pwm.stop()
        GPIO.cleanup()
        print("[Cleanup] GPIO 해제 완료")
