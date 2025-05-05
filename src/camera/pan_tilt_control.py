import time
import logging

try:
    import busio
    import board
    from adafruit_pca9685 import PCA9685
    REAL_GPIO = True
    logging.info("GPIO 환경 확인: 라즈베리파이 GPIO 사용 가능")
except (NotImplementedError, ImportError):
    REAL_GPIO = False
    logging.warning("GPIO 환경 확인: GPIO 제어 불가능, 가상 모드 활성화")

class PanTiltController:
    """
    카메라 Pan/Tilt 제어기 (Adafruit PCA9685 기반)
    - pan: 좌우 회전 (-90°~+90°)
    - tilt: 상하 회전 (-90°~+90°)
    - 상황별 포즈 메모/리콜 기능 제공

    Usage:
        controller = PanTiltController(pan_channel=0, tilt_channel=1)
        controller.goto(30, -15)
        controller.memorize_position('overview')
        controller.reset()
        controller.recall_position('overview')
        controller.release()
    """
    def __init__(self, pan_channel: int = 0, tilt_channel: int = 1, i2c_bus=None, frequency: int = 50):
        # 현재 상태
        self.current_pan = 0.0
        self.current_tilt = 0.0
        self.positions = {}
        self.REAL_GPIO = REAL_GPIO

        if self.REAL_GPIO:
            # 실제 GPIO 환경 초기화
            self.i2c = i2c_bus or busio.I2C(board.SCL, board.SDA)
            self.pca = PCA9685(self.i2c)
            self.pca.frequency = frequency
            self.pan_channel = pan_channel
            self.tilt_channel = tilt_channel

            logging.info(f"PanTiltController initialized (Real GPIO) pan={pan_channel}, tilt={tilt_channel}, freq={frequency}Hz")
        else:
            # GPIO 없는 환경에서 가상 모드 초기화
            logging.info("PanTiltController initialized (Virtual Mode)")

    def _angle_to_duty_cycle(self, angle: float) -> int:
        """
        서보 각도(-90~+90)를 PCA9685 16비트 duty_cycle 값으로 변환
        -90° -> 2.5% (min), 0° -> 7.5% (mid), +90° -> 12.5% (max)
        """
        # 제한
        angle = max(min(angle, 90.0), -90.0)
        # 2.5 ~ 12.5% 사이로 맵핑
        percent = 7.5 + (angle / 18.0)
        # 16-bit 범위 (0x0000 ~ 0xFFFF)
        return int(percent / 100.0 * 0xFFFF)

    def set_pan(self, angle: float):
        """pan 채널 서보 각도 설정"""
        self.current_pan = angle
        duty = self._angle_to_duty_cycle(angle)
        self.pca.channels[self.pan_channel].duty_cycle = duty
        logging.info(f"Pan -> {angle}° (duty_cycle={duty})")
        time.sleep(0.1)

    def set_tilt(self, angle: float):
        """tilt 채널 서보 각도 설정"""
        self.current_tilt = angle
        duty = self._angle_to_duty_cycle(angle)
        self.pca.channels[self.tilt_channel].duty_cycle = duty
        logging.info(f"Tilt -> {angle}° (duty_cycle={duty})")
        time.sleep(0.1)

    def goto(self, pan_angle: float, tilt_angle: float):
        """지정된 pan/tilt 포즈로 이동"""
        self.set_pan(pan_angle)
        self.set_tilt(tilt_angle)

    def pan_left(self, step: float = 5.0):
        """현재 위치에서 왼쪽으로 이동"""
        self.set_pan(self.current_pan - step)

    def pan_right(self, step: float = 5.0):
        """현재 위치에서 오른쪽으로 이동"""
        self.set_pan(self.current_pan + step)

    def tilt_up(self, step: float = 5.0):
        """현재 위치에서 위로 이동"""
        self.set_tilt(self.current_tilt + step)

    def tilt_down(self, step: float = 5.0):
        """현재 위치에서 아래로 이동"""
        self.set_tilt(self.current_tilt - step)

    def memorize_position(self, name: str):
        """현재 pan/tilt 좌표를 이름으로 저장"""
        self.positions[name] = (self.current_pan, self.current_tilt)
        logging.info(f"Memorized '{name}': pan={self.current_pan}, tilt={self.current_tilt}")

    def recall_position(self, name: str):
        """저장된 좌표로 이동"""
        if name not in self.positions:
            raise KeyError(f"Unknown position '{name}'")
        pan, tilt = self.positions[name]
        self.goto(pan, tilt)
        logging.info(f"Recalled '{name}' -> pan={pan}, tilt={tilt}")

    def reset(self):
        """pan/tilt 초기 위치(0,0)로 복귀"""
        self.goto(0.0, 0.0)
        logging.info("Reset to (0°,0°)")

    def release(self):
        """하드웨어 리소스 해제"""
        self.pca.deinit()
        logging.info("PCA9685 deinitialized, resources released")
