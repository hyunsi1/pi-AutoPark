import time
import logging
import math
import RPi.GPIO as GPIO
from DFRobot_RaspberryPi_Expansion_Board import DFRobot_Expansion_Board_IIC as Board
from DFRobot_RaspberryPi_Expansion_Board import DFRobot_Expansion_Board_Servo as Servo

class Controller:
    def __init__(self, steer_channel=0, in1_channel=1, in2_channel=2, ena_gpio=16, i2c_bus=1, addr=0x10):
        self.steer_ch = steer_channel
        self.in1_ch = in1_channel
        self.in2_ch = in2_channel
        self.ena_gpio = ena_gpio

        # 각도 정의
        self.ANGLE_LEFT = 100
        self.ANGLE_RIGHT = 30
        self.ANGLE_FORWARD = 60

        # I2C 보드 초기화
        self.board = Board(i2c_bus, addr)
        while self.board.begin() != self.board.STA_OK:
            print("[Init] I2C 보드 초기화 실패, 재시도 중...")
            time.sleep(1)
        print("[Init] I2C 보드 초기화 성공")

        # Servo 객체 초기화 (PWM 50Hz로 시작)
        self.servo = Servo(self.board)
        self.servo.begin()  # Servo 초기화 및 50Hz 설정됨

        # ENA 핀 초기화
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.ena_gpio, GPIO.OUT)
        GPIO.output(self.ena_gpio, GPIO.HIGH)

        # 초기 상태
        self.set_angle(self.ANGLE_FORWARD)
        self.stop()
        print(f"[Init] steer={steer_channel}, in1={in1_channel}, in2={in2_channel}, ena={ena_gpio}")

    def set_angle(self, angle, delay=0.3):
        """서보 각도 설정 + PWM 주파수 전환"""
        angle = max(0, min(180, angle))

        # 서보용 주파수 전환 및 구동
        self.board.set_pwm_frequency(50)
        self.servo.move(self.steer_ch, angle)
        print(f"[Steering] angle={angle}°")
        time.sleep(delay)


    def set_speed(self, speed_percent, reverse=False):
        """DC 모터 속도/방향 설정"""

        # DC 모터용 주파수로 설정정
        self.board.set_pwm_frequency(1000)

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
        """모터 정지"""
        self.board.set_pwm_frequency(1000)
        self.board.set_pwm_duty([self.in1_ch], 0)
        self.board.set_pwm_duty([self.in2_ch], 0)
        print("[Stop] 모터 정지")

    def navigate(self, direction: str, speed=30):
        """방향 명령 처리"""
        direction = direction.lower()

        if direction == "left":
            self.set_angle(self.ANGLE_LEFT)
            self.set_speed(speed)
            print("[Navigate] 좌회전")

        elif direction == "right":
            self.set_angle(self.ANGLE_RIGHT)
            self.set_speed(speed)
            print("[Navigate] 우회전")

        elif direction == "forward":
            self.set_angle(self.ANGLE_FORWARD)
            self.set_speed(speed)
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

    def map_physical_angle_to_servo(self, physical_angle_deg):
        """-45~+45도 물리 각도를 30~100의 서보각으로 변환"""
        servo_angle = (physical_angle_deg + 45) * (100 - 30) / 90 + 30
        return servo_angle

    def navigate_to(self, cur_pos, target_pos):
        """현재 위치에서 목표 위치로 조향"""
        dx = target_pos[0] - cur_pos[0]
        dy = target_pos[1] - cur_pos[1]

        if dx == 0 and dy == 0:
            self.stop()
            return

        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        angle_deg = max(-45, min(45, angle_deg))

        steering_angle = self.map_physical_angle_to_servo(angle_deg)
        steering_angle = max(0, min(180, steering_angle))

        self.set_angle(steering_angle)
        self.set_speed(30)
        print(f"[NavigateTo] {cur_pos} -> {target_pos} : steering_angle={steering_angle:.1f}")

    def cleanup(self):
        """GPIO 정리"""
        self.stop()
        GPIO.cleanup()
        print("[Cleanup] GPIO 정리 완료")
