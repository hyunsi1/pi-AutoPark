import time
import logging

try:
    import busio
    import board
    from adafruit_pca9685 import PCA9685
    REAL_GPIO = True
    logging.info("GPIO 환경 확인: 라즈베리파이 GPIO 사용 가능")
except (ImportError, NotImplementedError):
    REAL_GPIO = False
    logging.warning("GPIO 환경 확인: GPIO 제어 불가능, 가상 모드 활성화")

class PanTiltController:
    """
    카메라 Tilt(상하) 제어기 (Adafruit PCA9685 기반)
    - tilt: 상하 회전 (-90°~+90°)
    """
    def __init__(self, tilt_channel: int = 1, frequency: int = 50):
        self.current_tilt = 0.0
        self.REAL_GPIO = REAL_GPIO

        if self.REAL_GPIO:
            # I2C 버스 초기화
            self.i2c = busio.I2C(board.SCL, board.SDA)
            self.pca = PCA9685(self.i2c)
            self.pca.frequency = frequency
            self.tilt_channel = tilt_channel
            logging.info(
                f"TiltController 초기화 (Real GPIO) tilt_channel={tilt_channel}, freq={frequency}Hz"
            )
        else:
            logging.info("TiltController 초기화 (Virtual Mode)")

    def _angle_to_duty_cycle(self, angle: float) -> int:
        """
        서보 각도(-90~+90)를 16비트 duty_cycle 값으로 변환
        -90° -> 2.5% , 0° -> 7.5% , +90° ->12.5%
        """
        angle = max(min(angle, 90.0), -90.0)
        percent = 7.5 + (angle / 18.0)
        return int(percent / 100.0 * 0xFFFF)

    def set_tilt(self, angle: float):
        """Tilt(상하) 서보 각도 설정"""
        self.current_tilt = angle
        if not self.REAL_GPIO:
            logging.info(f"[Virtual] Tilt -> {angle}°")
            time.sleep(0.1)
            return

        duty = self._angle_to_duty_cycle(angle)
        self.pca.channels[self.tilt_channel].duty_cycle = duty
        logging.info(f"Tilt -> {angle}° (duty_cycle={duty})")
        time.sleep(0.1)

    def reset(self):
        """Tilt를 기본 위치(0°)로 복귀"""
        self.set_tilt(0.0)
        logging.info("TiltController reset to 0°")

    def release(self):
        """하드웨어 자원 해제"""
        if self.REAL_GPIO:
            self.pca.deinit()
            logging.info("PCA9685 deinitialized")

# 간단 사용 예제
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ctl = PanTiltController(tilt_channel=1)
    try:
        for ang in [-30, 0, 30, -30]:
            ctl.set_tilt(ang)
            time.sleep(1)
        ctl.reset()
    finally:
        ctl.release()
