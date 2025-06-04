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
        self.servo.move(self.tilt_channel, 0)  
        print("[Tilt] 0도 복귀")

    def final_step_tilt_down(self):
        self.servo.move(self.tilt_channel, 30)
        print("[Tilt] Final Step → 30도 내려감")

    def release(self):
        self.board.set_pwm_disable()
        print("[Tilt] 보드 PWM OFF")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ctl = PanTiltController(tilt_channel=0)  # 채널 번호를 보드 연결에 맞게 조정

    try:
        for ang in [-30, 0, 30, 60, -60]:
            pwm_angle = ang + 90
            ctl.servo.move(ctl.tilt_channel, pwm_angle)
            print(f"Moved to {ang}° → PWM {pwm_angle}°")
            time.sleep(1)
        ctl.reset()
        time.sleep(1)
        ctl.final_step_tilt_down()
        time.sleep(1)
        ctl.reset()
    finally:
        ctl.release()
