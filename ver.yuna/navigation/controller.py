import time
import logging
import math
import RPi.GPIO as GPIO
from DFRobot_RaspberryPi_Expansion_Board import DFRobot_Expansion_Board_IIC as Board
from DFRobot_RaspberryPi_Expansion_Board import DFRobot_Expansion_Board_Servo as Servo
from path_planner import PathPlanner 

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

        # PathPlanner 초기화
        self.planner = PathPlanner()

        # 주행 상태
        self.is_steering = False
        self.is_moving = False
        self.last_target_pos = None
        self.steer_start_time = time.time()  # 서보 모터 조향 시작 시간
        self.move_start_time = time.time()  # 모터 이동 시작 시간
        self.pending_speed = 0  # 대기 속도
        self.move_duration = 2.0  # 모터 이동 대기 시간

    @property
    def is_busy(self):
        """현재 하드웨어가 동작 중일 때 True 반환"""
        return self.is_steering or self.is_moving

    def set_angle(self, angle, delay=0.3):
        """서보 각도 설정 + PWM 주파수 전환"""
        angle = max(0, min(180, angle))

        # 서보용 주파수 전환 및 구동
        self.board.set_pwm_frequency(50)
        self.servo.move(self.steer_ch, angle)
        print(f"[Steering] angle={angle}°")

        self.steer_start_time = time.time()  # 각도 변경 후 시간 초기화
        self.is_steering = True

    def set_speed(self, speed_percent, reverse=False):
        """DC 모터 속도/방향 설정"""
        # DC 모터용 주파수로 설정
        self.board.set_pwm_frequency(1000)

        duty = max(0, min(100, speed_percent))
        if reverse:
            self.board.set_pwm_duty([self.in1_ch], 0)
            self.board.set_pwm_duty([self.in2_ch], duty)
            time.sleep(0.5)
            print(f"[Motor] ← 후진, IN1=0%, IN2={duty:.1f}%")
        else:
            self.board.set_pwm_duty([self.in1_ch], duty)
            self.board.set_pwm_duty([self.in2_ch], 0)
            time.sleep(0.5)
            print(f"[Motor] → 전진, IN1={duty:.1f}%, IN2=0%")

        self.is_moving = True

    def stop(self):
        """모터 정지"""
        self.board.set_pwm_frequency(1000)
        self.board.set_pwm_duty([self.in1_ch], 0)
        self.board.set_pwm_duty([self.in2_ch], 0)
        time.sleep(0.5)
        print("[Stop] 모터 정지")
        self.is_moving = False

    def navigate(self, direction: str, speed=30):
        """방향 명령 처리"""
        if self.is_busy:
            print("[Navigate] 하드웨어 동작 중 - 상태 전환 불가")
            return  # 하드웨어가 동작 중일 때는 상태 전환을 막음

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

    def navigate_using_planner(self, current_pos, goal_pos):
        """PathPlanner 사용하여 목표 위치로 정확한 경로 계획 후 이동"""
        # 목표 지점까지의 경로를 PathPlanner로 계산
        target_pos = self.planner.pid_step(current_pos, goal_pos)

        # 목표 지점까지의 경로를 이동
        self.navigate_to(current_pos, target_pos)

    def update_movement(self, speed, direction, move_duration=None):
        """주행 시간 및 상태 업데이트"""
        if move_duration is None:
            move_duration = self.move_duration  # 기본 이동 시간 설정

        if time.time() - self.move_start_time >= move_duration:
            self.is_moving = False
            self.stop()
            print(f"[Update Movement] 목표 위치 도달, 이동 종료")
        else:
            self.is_moving = True
            self.navigate(direction, speed)
            print(f"[Update Movement] 이동 중... {direction} 진행")

    def update_steering(self, angle, steer_duration=None):
        """조향 상태 업데이트"""
        if steer_duration is None:
            steer_duration = 2.0  # 기본 조향 시간 설정

        if time.time() - self.steer_start_time >= steer_duration:
            self.is_steering = False
            print(f"[Update Steering] 조향 완료")
        else:
            self.is_steering = True
            self.set_angle(angle)
            print(f"[Update Steering] 조향 중... {angle}°로 조향")


'''
import time
import logging
import math
import RPi.GPIO as GPIO
from DFRobot_RaspberryPi_Expansion_Board import DFRobot_Expansion_Board_IIC as Board
from DFRobot_RaspberryPi_Expansion_Board import DFRobot_Expansion_Board_Servo as Servo
from navigation.path_planner import PathPlanner

class Controller:
    def __init__(
        self,
        steer_channel=0,
        in1_gpio=20,
        in2_gpio=21,
        ena_gpio=16,
        i2c_bus=1,
        addr=0x10
    ):
        # --- steering servo init ---
        self.steer_ch = steer_channel
        self.ANGLE_LEFT    = 100
        self.ANGLE_RIGHT   = 30
        self.ANGLE_FORWARD = 60

        # I2C expansion board for servo
        self.board = Board(i2c_bus, addr)
        while self.board.begin() != self.board.STA_OK:
            print("[Init] I2C init failed, retrying...")
            time.sleep(1)
        print("[Init] I2C initialized")

        self.servo = Servo(self.board)
        self.servo.begin()  # 50Hz

        # --- DC motor GPIO init ---
        self.in1_gpio = in1_gpio
        self.in2_gpio = in2_gpio
        self.ena_gpio = ena_gpio

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.in1_gpio, GPIO.OUT)
        GPIO.setup(self.in2_gpio, GPIO.OUT)
        GPIO.setup(self.ena_gpio, GPIO.OUT)

        # ENA PWM at 1kHz
        self.pwm = GPIO.PWM(self.ena_gpio, 1000)
        self.pwm.start(0)

        # reset steering & motor
        self.set_angle(self.ANGLE_FORWARD)
        self.stop()
        print(f"[Init] steer_ch={steer_channel}, IN1_GPIO={in1_gpio}, IN2_GPIO={in2_gpio}, ENA_GPIO={ena_gpio}")

        # path planner
        self.planner = PathPlanner()

        # busy flags
        self.is_steering = False
        self.is_moving   = False
        self.steer_start_time = time.time()
        self.move_start_time  = time.time()
        self.move_duration    = 2.0

    @property
    def is_busy(self):
        return self.is_steering or self.is_moving

    def set_angle(self, angle):
        """Set steering servo angle (0–180)."""
        angle = max(0, min(180, angle))
        self.board.set_pwm_frequency(50)
        self.servo.move(self.steer_ch, angle)
        print(f"[Steering] angle={angle}°")
        self.steer_start_time = time.time()
        self.is_steering = True

    def set_speed(self, speed_percent, reverse=False):
        """Drive DC motor with PWM on ENA and direction on IN1/IN2."""
        speed = max(0, min(100, speed_percent))
        if reverse:
            GPIO.output(self.in1_gpio, GPIO.LOW)
            GPIO.output(self.in2_gpio, GPIO.HIGH)
        else:
            GPIO.output(self.in1_gpio, GPIO.HIGH)
            GPIO.output(self.in2_gpio, GPIO.LOW)
        self.pwm.ChangeDutyCycle(speed)
        print(f"[Motor] speed={'-' if reverse else ''}{speed}%")
        self.is_moving = True
        self.move_start_time = time.time()

    def stop(self):
        """Stop DC motor."""
        GPIO.output(self.in1_gpio, GPIO.LOW)
        GPIO.output(self.in2_gpio, GPIO.LOW)
        self.pwm.ChangeDutyCycle(0)
        time.sleep(0.1)
        print("[Stop] motor stopped")
        self.is_moving = False

    def navigate(self, direction: str, speed=30):
        """Simple directional drive."""
        if self.is_busy:
            print("[Navigate] Busy, command ignored")
            return

        d = direction.lower()
        if d == "left":
            self.set_angle(self.ANGLE_LEFT)
            self.set_speed(speed)
        elif d == "right":
            self.set_angle(self.ANGLE_RIGHT)
            self.set_speed(speed)
        elif d == "forward":
            self.set_angle(self.ANGLE_FORWARD)
            self.set_speed(speed)
        elif d == "backward":
            self.set_angle(self.ANGLE_FORWARD)
            self.set_speed(speed, reverse=True)
        elif d == "stop":
            self.stop()
        else:
            print(f"[Navigate] Unknown direction: {direction}")

    def map_physical_angle_to_servo(self, physical_angle_deg):
        """Map -45~+45° to servo 30~100° range."""
        return (physical_angle_deg + 45) * (100 - 30) / 90 + 30

    def navigate_to(self, cur_pos, target_pos):
        """Steer and drive toward target pose."""
        dx = target_pos[0] - cur_pos[0]
        dy = target_pos[1] - cur_pos[1]
        if dx == 0 and dy == 0:
            self.stop()
            return

        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        angle_deg = max(-45, min(45, angle_deg))
        servo_ang = self.map_physical_angle_to_servo(angle_deg)
        servo_ang = max(0, min(180, servo_ang))

        self.set_angle(servo_ang)
        self.set_speed(30)
        print(f"[NavigateTo] {cur_pos} -> {target_pos} : steering={servo_ang:.1f}°")

    def cleanup(self):
        """Release resources."""
        self.stop()
        self.pwm.stop()
        GPIO.cleanup()
        print("[Cleanup] GPIO cleaned up")

    def navigate_using_planner(self, current_pos, goal_pos):
        """PID‐based planner step."""
        target = self.planner.pid_step(current_pos, goal_pos)
        self.navigate_to(current_pos, target)

    def update_movement(self, speed, direction, move_duration=None):
        """Periodic call to maintain movement."""
        if move_duration is None:
            move_duration = self.move_duration
        if time.time() - self.move_start_time >= move_duration:
            self.stop()
            print("[Update Movement] Movement duration expired")
        else:
            self.navigate(direction, speed)

    def update_steering(self, angle, steer_duration=2.0):
        """Periodic call to maintain steering."""
        if time.time() - self.steer_start_time >= steer_duration:
            self.is_steering = False
            print("[Update Steering] Steering complete")
        else:
            self.set_angle(angle)
            print(f"[Update Steering] Maintaining angle {angle}°")

# Stand‐alone test
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    ctrl = Controller()

    print("\n-- Motor forward 50% for 2s --")
    ctrl.set_speed(50, reverse=False)
    time.sleep(2)
    ctrl.stop()

    print("\n-- Motor backward 50% for 2s --")
    ctrl.set_speed(50, reverse=True)
    time.sleep(2)
    ctrl.stop()

    print("\n-- Steering test --")
    for ang in [ctrl.ANGLE_LEFT, ctrl.ANGLE_FORWARD, ctrl.ANGLE_RIGHT]:
        ctrl.set_angle(ang)
        time.sleep(1)

    ctrl.cleanup()
    sys.exit(0)
'''
