import time
import logging
import yaml

from enum import Enum, auto
from camera.capture import FrameCapture
from camera.pan_tilt_control import PanTiltController
from camera.calibration import load_camera_parameters
from vision.yolo_detector import YOLODetector
from vision.monodepth_estimator import MonoDepthEstimator
from vision.slot_allocator import SlotAllocator
from navigation.path_planner import PathPlanner
from navigation.controller import Controller
from utility.transformations import pixel_to_world
from utility.distance import euclidean_distance


def load_config(path="config/config.yaml"):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class State(Enum):
    SEARCH = auto()
    NAVIGATE = auto()
    OBSTACLE_AVOID = auto()
    FINAL_APPROACH = auto()
    COMPLETE = auto()


class StateMachine:
    def __init__(self, config_path="config/config.yaml"):
        # Load configuration
        self.cfg = load_config(config_path)
        # Camera parameters
        self.cam_mtx, self.dist_coefs = load_camera_parameters()
        # Components
        self.detector = YOLODetector(weights_path=self.cfg['yolo']['weights_path'])
        self.estimator = MonoDepthEstimator(config_path=config_path)
        area_px = self.cfg['slot_area_coords']
        self.allocator = SlotAllocator(area_coords_px=area_px, config_path=config_path)
        self.planner = PathPlanner(num_segments=self.cfg.get('path_segments', 5))
        self.controller = Controller(max_speed=self.cfg.get('max_speed',1.0),
                                     turn_speed=self.cfg.get('turn_speed',0.5))
        self.pan_tilt = PanTiltController(
            pan_channel=self.cfg['pan_tilt']['pan_channel'],
            tilt_channel=self.cfg['pan_tilt']['tilt_channel']
        )
        # State
        self.state = State.SEARCH
        self.front_obstacle = None
        logging.info("StateMachine initialized")

    def run(self):
        while True:
            if self.state == State.SEARCH:
                self._search()
            elif self.state == State.NAVIGATE:
                self._navigate()
            elif self.state == State.OBSTACLE_AVOID:
                self._avoid()
            elif self.state == State.FINAL_APPROACH:
                self._final_approach()
            elif self.state == State.COMPLETE:
                self._complete()
                break
            time.sleep(0.1)

    def _search(self):
        logging.info("[SEARCH] state")
        with FrameCapture(source=self.cfg['camera']['source'],
                          image_folder=self.cfg['camera'].get('image_folder'),
                          loop=self.cfg['camera'].get('loop', False)) as cap:
            for ret, frame in iter(lambda: cap.read(), (False, None)):
                # detect scooters
                dets = self.detector.detect(frame)
                # estimate depth and map to world
                for d in dets:
                    d['depth'] = self.estimator.estimate_depth(d['bbox'])
                target = self.allocator.allocate(dets)
                if target:
                    self.target_slot = target
                    logging.info(f"Assigned slot: {target}")
                    self.state = State.NAVIGATE
                    return

    def _navigate(self):
        logging.info("[NAVIGATE] state")
        # Use last detection for current position
        last = self.detector.last_frame
        if isinstance(last, list) and len(last) > 0:
            last_det = last[0]
            curr_px = ((last_det['bbox'][0] + last_det['bbox'][2]) // 2, last_det['bbox'][3])
        elif isinstance(last, dict):
            curr_px = ((last['bbox'][0] + last['bbox'][2]) // 2, last['bbox'][3])
        else:
            curr_px = (0, 0) 
        homography = self.allocator.homography_matrix  # 반드시 존재해야 함
        curr_world = pixel_to_world(curr_px[0], curr_px[1], homography)
        # Plan path
        waypoints = self.planner.plan(curr_world, self.target_slot)
        safe_dist = self.cfg.get('safe_dist', 0.5)
        for wp in waypoints:
            # check obstacles
            dets = [self.detector.last_frame] if self.detector.last_frame else []
            pts = []
            for d in dets:
                if isinstance(d, dict) and 'bbox' in d:
                    x1, y1, x2, y2 = d['bbox']
                    px = ((x1 + x2) // 2, y2)
                    pts.append(pixel_to_world(px[0], px[1], self.allocator.homography_matrix))

            front = [p for p in pts if euclidean_distance(curr_world,p) < safe_dist]
            if front:
                self.front_obstacle = front[0]
                logging.info(f"Obstacle detected at {self.front_obstacle}")
                self.state = State.OBSTACLE_AVOID
                return
            self.controller.navigate_to(curr_world, wp)
            curr_world = wp
        self.state = State.FINAL_APPROACH

    def _avoid(self):
        logging.info("[OBSTACLE_AVOID] state")
        self.controller.stop()
        # generate detour
        clearance = self.cfg.get('clearance', 0.5)
        curr = pixel_to_world(((self.front_obstacle[0],self.front_obstacle[1])))
        detour = self.planner.replan_around(curr, self.target_slot, self.front_obstacle, clearance, area_bounds=self.allocator.area_world)
        for wp in detour[1:]:
            self.controller.navigate_to(curr, wp)
            curr = wp
        logging.info("Detour complete, resuming NAVIGATE")
        self.state = State.NAVIGATE

    def _final_approach(self):
        logging.info("[FINAL_APPROACH] state")
        self.pan_tilt.set_tilt(self.cfg['pan_tilt']['final_tilt_angle'])
        while True:
            dets = self.detector.detect(self.pan_tilt.capture_frame())
            if not dets: continue
            d = self.estimator.estimate_depth(dets[0]['bbox'])
            if d <= self.cfg['final_approach']['threshold']:
                break
        self.state = State.COMPLETE

    def _complete(self):
        logging.info("[COMPLETE] state")
        self.pan_tilt.reset()
        logging.info("Parking complete.")
        self.state = State.SEARCH

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sm = StateMachine()
    sm.run()
