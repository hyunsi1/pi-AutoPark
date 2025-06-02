import time
import logging
############ for DFR0604 HAT ##############
try:
    from DFRobot_RaspberryPi_Expansion_Board import DFRobot_Expansion_Board_IIC as Board
except ImportError:
    try:
        import RPi.GPIO as GPIO
        REAL_GPIO = True
    except (ImportError, RuntimeError):
        import logging
        from unittest import mock
        GPIO = mock.MagicMock()
        REAL_GPIO = False
        logging.warning("GPIO 환경 확인: GPIO 제어 불가능, 가상 모드 활성화")

class PanTiltController:
    """
    I2C 확장 보드를 통한 Tilt 제어 서보 컨트롤러
    - 기본 각도: 0°
    - final_step 도달 시: 30°
    """
    def __init__(self, board_addr=0x10, tilt_channel=3, i2c_bus=1):
        self.tilt_channel = tilt_channel  # PWM 채널
        self.current_tilt = 0.0

        self.board = Board(i2c_bus, board_addr)
        while self.board.begin() != self.board.STA_OK:
            print("[Init] I2C 보드 초기화 실패, 재시도 중...")
            time.sleep(1)
        print("[Init] I2C 보드 초기화 성공")

        self.board.set_pwm_enable()
        self.board.set_pwm_frequency(50)  # 서보 PWM = 50Hz

        self.set_tilt(0.0)  # 초기 상태 0도

    def _angle_to_duty(self, angle: float) -> float:
        """
        -90° ~ +90° → duty 2.5% ~ 12.5%
        (0° 기준: 7.5%)
        """
        angle = max(min(angle, 90.0), -90.0)
        return 7.5 + (angle / 18.0)

    def set_tilt(self, angle: float):
        """틸트 각도 설정"""
        self.current_tilt = angle
        duty = self._angle_to_duty(angle)
        self.board.set_pwm_duty(self.tilt_channel, duty)
        print(f"[Tilt] 각도={angle:.1f}°, PWM duty={duty:.2f}%")
        time.sleep(0.3)

    def reset(self):
        """틸트 0° 복귀"""
        self.set_tilt(0.0)
        print("[Tilt] 0도 복귀")

    def final_step_tilt_down(self):
        """파이널 스텝 도달 시 틸트 아래로 (30도)"""
        self.set_tilt(30.0)
        print("[Tilt] Final Step → 30도 내려감")

    def release(self):
        """PWM OFF (안정화용)"""
        self.board.set_pwm_duty(self.tilt_channel, 0)
        print("[Tilt] PWM OFF")
'''
try:
    import RPi.GPIO as GPIO
    REAL_GPIO = True
except (ImportError, RuntimeError):
    from unittest import mock
    GPIO = mock.MagicMock()
    REAL_GPIO = False
    logging.warning("GPIO 환경 확인: GPIO 제어 불가능, 가상 모드 활성화")

class PanTiltController:
    """
    카메라 Tilt 제어기 (GPIO PWM 방식)
    - tilt: 상하 회전 (-90° ~ +90°)
    - 제어 핀: GPIO PWM 핀 (기본: GPIO 12 또는 18)
    """
    def __init__(self, tilt_pin: int = 12, frequency: int = 50):
        self.REAL_GPIO = REAL_GPIO
        self.tilt_pin = tilt_pin
        self.frequency = frequency
        self.current_tilt = 0.0

        if self.REAL_GPIO:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.tilt_pin, GPIO.OUT)
            self.pwm = GPIO.PWM(self.tilt_pin, self.frequency)
            self.pwm.start(7.5)  # 중립 위치 (0도)
            logging.info(f"TiltController 초기화: GPIO {tilt_pin}, 주파수 {frequency}Hz")
        else:
            logging.info("TiltController 초기화 (Virtual Mode)")

    def _angle_to_duty_cycle(self, angle: float) -> float:
        """
        -90° ~ +90° 범위를 2.5% ~ 12.5% PWM duty로 매핑
        """
        angle = max(min(angle, 90.0), -90.0)
        return 7.5 + (angle / 18.0)

    def set_tilt(self, angle: float):
        """카메라 상하 각도 설정"""
        self.current_tilt = angle
        duty = self._angle_to_duty_cycle(angle)

        if self.REAL_GPIO:
            self.pwm.ChangeDutyCycle(duty)
            logging.info(f"[GPIO PWM] Tilt → {angle:.1f}° (duty={duty:.2f}%)")
        else:
            logging.info(f"[Virtual] Tilt → {angle:.1f}°")

        time.sleep(0.3)

    def reset(self):
        """Tilt를 0°로 복귀"""
        self.set_tilt(0.0)
        logging.info("TiltController → 0° 초기화")

    def release(self):
        """GPIO 자원 해제"""
        if self.REAL_GPIO:
            self.pwm.stop()
            GPIO.cleanup()
            logging.info("GPIO 해제 완료")
'''
# ========== 실행 예제 ==========
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ctl = PanTiltController(tilt_pin=12)  # GPIO 12 (핀 32 사용)

    try:
        for ang in [-30, 0, 30, 60, -60]:
            ctl.set_tilt(ang)
            time.sleep(1)
        ctl.reset()
    finally:
        ctl.release()