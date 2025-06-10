import time
import logging

# ========== DFRobot HAT 모듈 임포트 ==========
try:
    from DFRobot_RaspberryPi_Expansion_Board import DFRobot_Expansion_Board_IIC as Board
    from DFRobot_RaspberryPi_Expansion_Board import DFRobot_Expansion_Board_Servo as Servo
except ImportError:
    raise ImportError("DFRobot_RaspberryPi_Expansion_Board 라이브러리를 찾을 수 없습니다.")

class PanTiltController:
    def __init__(self, board_addr=0x10, tilt_channel=3, i2c_bus=1):
        self.tilt_channel = tilt_channel  # PWM 채널 번호

        self.board = Board(i2c_bus, board_addr)
        while self.board.begin() != self.board.STA_OK:
            print("[Init] I2C 보드 초기화 실패, 재시도 중...")
            time.sleep(1)
        print("[Init] I2C 보드 초기화 성공")

        # Servo 객체 초기화
        self.servo = Servo(self.board)
        self.servo.begin()

        self.reset()

    def reset(self):
        self.board.set_pwm_frequency(50)
        self.servo.move(self.tilt_channel, 0)  
        print("[Tilt] 0° 복귀")

    def final_step_tilt_down(self):
        self.board.set_pwm_frequency(50)
        self.servo.move(self.tilt_channel, 30)
        print("[Tilt] Final Step → 30° 내림")
    
    def tilt(self, angle):
        self.board.set_pwm_frequency(50)
        self.servo.move(self.tilt_channel, angle)
        print(f"[Tilt] {angle}° 내림")

    def release(self):
        self.board.set_pwm_disable()
        print("[Tilt] 보드 PWM OFF")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ctl = PanTiltController(tilt_channel=3)  # 채널 번호를 보드 연결에 맞게 조정

    try:
        ctl.reset()
        time.sleep(1)
        ctl.final_step_tilt_down()
        time.sleep(1)
        ctl.reset()
    finally:
        ctl.release()
