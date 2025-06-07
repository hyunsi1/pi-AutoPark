import os, yaml, logging, numpy as np
from camera.capture           import FrameCapture
from vision.yolo_detector     import YOLODetector
from vision.monodepth_estimator import MonoDepthEstimator
from vision.goal_setter       import GoalSetter
from navigation.path_planner  import PathPlanner
from navigation.controller    import Controller
from camera.pan_tilt_control  import PanTiltController
from interface.user_io        import UserIO
from fsm.state_machine_hyunsu import StateMachine


def load_config(path='config/config.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger('matplotlib').setLevel(logging.ERROR)

def main() -> None:
    cfg = load_config()
    setup_logging()
    log = logging.getLogger(__name__)
    log.info('=== PI-AutoPark 시작 ===')

    # ―――― Vision ――――
    cap = FrameCapture(**cfg['camera'])
    yolo_custom = YOLODetector(weights_path=cfg['yolo']['custom_weights'],
                               conf_thres=cfg['yolo']['conf_thres'],
                               iou_thres=cfg['yolo']['iou_thres'])
    yolo_coco   = YOLODetector(weights_path=cfg['yolo']['coco_weights'],
                               conf_thres=cfg['yolo']['conf_thres'],
                               iou_thres=cfg['yolo']['iou_thres'])
    detectors = {'custom': yolo_custom, 'coco': yolo_coco}
    depth     = MonoDepthEstimator()

    # ―――― Homography (픽셀→월드) ――――
    H = np.load('config/camera_params.npz')['homography_matrix'] \
        if os.path.exists('config/camera_params.npz') else np.eye(3)
    goal_setter = GoalSetter(min_gap=80, homography=H)

    # ―――― Navigation ――――
    planner    = PathPlanner(step_size=0.15)
    controller = Controller(**cfg['controller'])
    pan_tilt   = PanTiltController(**cfg['pan_tilt'])
    ui         = UserIO()

    sm = StateMachine(cfg, cap, detectors, depth,
                      goal_setter, planner, controller, pan_tilt, ui)
    sm.run()

if __name__ == '__main__':
    main()