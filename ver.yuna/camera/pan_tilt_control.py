import time
import logging

# ========== DFRobot HAT 모듈 임포트 ========== 
try:
    from DFRobot_RaspberryPi_Expansion_Board import DFRobot_Expansion_Board_IIC as Board
    from DFRobot_RaspberryPi_Expansion_Board import DFRobot_Expansion_Board_Servo as Servo
except ImportError:
    raise ImportError("DFRobot_RaspberryPi_Expansion_Board 라이브러리를 찾을 수 없습니다.")

class PanTiltController:
    def __init__(self, board_addr=0x10, tilt_channel=3, i2c_bus=1, reset_delay=1.0):
        self.tilt_channel = tilt_channel  # PWM 채널 번호
        self.reset_delay = reset_delay    # 초기화 지연 시간

        # I2C 보드 초기화
        self.board = Board(i2c_bus, board_addr)
        while self.board.begin() != self.board.STA_OK:
            print("[Init] I2C 보드 초기화 실패, 재시도 중...")
            time.sleep(1)
        print("[Init] I2C 보드 초기화 성공")

        # Servo 객체 초기화
        self.servo = Servo(self.board)
        self.servo.begin()

        # 초기 상태 설정
        self.last_move_time = time.time()  # 마지막 동작 시간 기록
        self.reset()  # 초기화

    def reset(self):
        """서보 각도 초기화 (0도 복귀)"""
        self.servo.move(self.tilt_channel, 0)
        print("[Tilt] 0도 복귀")
        self.last_move_time = time.time()  # 시간 갱신

    def final_step_tilt_down(self):
        """최종 단계로 서보를 30도 내림"""
        self.servo.move(self.tilt_channel, 30)
        print("[Tilt] Final Step → 30도 내려감")
        self.last_move_time = time.time()  # 시간 갱신

    def release(self):
        """PWM 해제 (서보 모터를 멈추고 리소스를 반환)"""
        self.board.set_pwm_disable()
        print("[Tilt] 보드 PWM OFF")

    def is_tilt_done(self, duration=1.0):
        """서보가 특정 시간 동안 움직였는지 확인 (비동기적 제어)"""
        # 마지막 동작 이후 시간이 충분히 경과했으면 동작 완료
        return (time.time() - self.last_move_time) >= duration

    def perform_tilt_motion(self, angle, duration=1.0):
        """서보 모터로 특정 각도 이동 후 대기 (비동기적 제어)"""
        if self.is_tilt_done(duration):
            self.servo.move(self.tilt_channel, angle)
            print(f"[Tilt] 이동 완료: {angle}°")
            self.last_move_time = time.time()
        else:
            print("[Tilt] 이전 동작 완료 전 이동 불가")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ctl = PanTiltController(tilt_channel=0)  # 채널 번호를 보드 연결에 맞게 조정

    try:
        # 직진 후 서보 모터를 이용한 동작 테스트
        ctl.perform_tilt_motion(-30, duration=1.0)  # -30도 이동
        time.sleep(1)
        ctl.perform_tilt_motion(0, duration=1.0)   # 0도로 복귀
        time.sleep(1)
        ctl.final_step_tilt_down()                 # 최종적으로 30도로 내림
        time.sleep(1)
        ctl.reset()                               # 다시 0도로 복귀
    finally:
        ctl.release()  # 최종 정리
