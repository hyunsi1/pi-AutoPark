import time
import logging

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
