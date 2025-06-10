import time
import math
import RPi.GPIO as GPIO
from DFRobot_RaspberryPi_Expansion_Board import DFRobot_Expansion_Board_IIC as Board
from DFRobot_RaspberryPi_Expansion_Board import DFRobot_Expansion_Board_Servo as Servo

class Controller:
    def __init__(
        self,
        steer_channel=0,
        in1_channel=1,
        in2_channel=2,
        ena_gpio=16,
        i2c_bus=1,
        addr=0x10,
        angle_delay=0.2,
        speed_stabilize=0.1
    ):
        # GPIO & I2C setup
        self.steer_ch = steer_channel
        self.in1_ch = in1_channel
        self.in2_ch = in2_channel
        self.ena_gpio = ena_gpio
        self.angle_delay = angle_delay
        self.speed_stabilize = speed_stabilize

        self.board = Board(i2c_bus, addr)
        while self.board.begin() != self.board.STA_OK:
            print("[Init] I2C init failed, retrying...")
            time.sleep(1)
        print("[Init] I2C initialized")

        self.servo = Servo(self.board)
        self.servo.begin()

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.ena_gpio, GPIO.OUT)
        GPIO.output(self.ena_gpio, GPIO.HIGH)

        # initial state
        self.is_steering = False
        self.is_moving = False

        # center forward
        self.center_steer()
        self.stop()
        print(f"[Init] steer={steer_channel}, in1={in1_channel}, in2={in2_channel}, ena={ena_gpio}")

    def center_steer(self):
        # map 0 offset to servo midpoint
        self.set_angle(65)

    def set_angle(self, angle, delay=None):
        """
        Move steering servo to 'angle' (0-180°) and hold for 'delay' seconds.
        """
        if delay is None:
            delay = self.angle_delay

        angle = max(0, min(180, angle))
        self.board.set_pwm_frequency(50)
        self.servo.move(self.steer_ch, angle)
        self.is_steering = True
        print(f"[Steering] angle={angle}°")

        time.sleep(delay)
        self.is_steering = False

    def set_speed(self, duty, reverse=False, stabilize=None):
        """
        Start motor at 'duty'% speed. Optionally reverse.
        """
        if stabilize is None:
            stabilize = self.speed_stabilize

        duty = max(0, min(100, duty))
        self.board.set_pwm_frequency(1000)
        if reverse:
            self.board.set_pwm_duty([self.in1_ch], 0)
            self.board.set_pwm_duty([self.in2_ch], duty)
            print(f"[Motor] reverse, IN1=0%, IN2={duty}%")
        else:
            self.board.set_pwm_duty([self.in1_ch], duty)
            self.board.set_pwm_duty([self.in2_ch], 0)
            print(f"[Motor] forward, IN1={duty}%, IN2=0%")

        self.is_moving = True
        time.sleep(stabilize)

    def stop(self, stabilize=None):
        """
        Stop motor and optionally wait for hardware stabilization.
        """
        if stabilize is None:
            stabilize = self.speed_stabilize

        self.board.set_pwm_frequency(1000)
        self.board.set_pwm_duty([self.in1_ch], 0)
        self.board.set_pwm_duty([self.in2_ch], 0)
        print("[Stop] motor stopped")

        self.is_moving = False
        time.sleep(stabilize)

    def emergency_stop(self):
        """
        Immediate full stop without stabilization delay.
        """
        self.board.set_pwm_duty([self.in1_ch], 0)
        self.board.set_pwm_duty([self.in2_ch], 0)
        self.is_moving = False
        print("[Emergency Stop]")

    def map_physical_angle_to_servo(self, physical_angle_deg, min_servo=30, max_servo=100):
        """
        Map physical angle (-45° to +45°) to servo PWM angle range.
        """
        clamped = max(-45, min(45, physical_angle_deg))
        # linear map from [-45,45] to [min_servo,max_servo]
        return (clamped + 45) * (max_servo - min_servo) / 90 + min_servo

    @property
    def is_busy(self):
        return self.is_steering or self.is_moving

    def navigate_segment(self, planner, desired_angle, distance, speed_pct=30, reverse=False, obstacle_detector=None):
        """
        Steering + move segment: steer, drive for distance at speed_pct, then stop.
        If obstacle_detector returns True, triggers emergency_stop.
        """
        servo_angle = self.map_physical_angle_to_servo(desired_angle)
        self.set_angle(servo_angle)
        self.set_speed(speed_pct, reverse)

        # drive with obstacle check
        t_move = distance / planner.speed_mps
        t0 = time.time()
        while time.time() - t0 < t_move:
            if obstacle_detector and obstacle_detector():
                self.emergency_stop()
                return False
            time.sleep(0.05)

        self.stop()
        return True

    def cleanup(self):
        self.stop()
        GPIO.cleanup()
        print("[Cleanup] GPIO cleaned up")


# ---------------- Test Main ----------------
if __name__ == "__main__":
    from path_planner import PathPlanner

    # Initialize planner and controller
    planner = PathPlanner(pixel_to_meter=0.1, speed_mps=0.8)
    controller = Controller()

    # Define start and goal positions
    start = (0.0, 0.0)
    goal = (1.0, 0.0)

    print("[Main] Planning path from {} to {}".format(start, goal))
    waypoints = planner.plan(start, goal)
    print("[Main] Waypoints:", waypoints)

    # Example obstacle detector function
    def dummy_detector():
        # Simulate an obstacle after 0.5s
        return time.time() - t_nav_start > 0.5

    print("[Main] Navigating through waypoints with emergency_stop on obstacle...")
    for wp in waypoints:
        angle = wp["angle"]
        dist = wp["distance"]
        t_nav_start = time.time()
        success = controller.navigate_segment(
            planner,
            desired_angle=angle,
            distance=dist,
            speed_pct=50,
            obstacle_detector=dummy_detector
        )
        if not success:
            print("[Main] Navigation aborted by emergency_stop")
            break

    # Manual emergency stop example
    print("[Main] Testing manual emergency stop...")
    controller.set_speed(50)
    time.sleep(1)
    controller.emergency_stop()

    # Cleanup GPIO and exit
    controller.cleanup()
