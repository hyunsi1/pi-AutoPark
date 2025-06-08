import cv2
from pathlib import Path
import itertools

class FrameCapture:
    """
    Webcam or image folder frame provider.

    Args:
        source (int or str): webcam index or video file path.
        image_folder (str): path to a folder of image files. If provided, frames are read from images.
        loop (bool): if True and image_folder is set, cycle through images indefinitely.
    """
    def __init__(self, source=0, image_folder: str = None, loop: bool = False):
        self.loop = loop
        if image_folder:
            self.image_folder = Path(image_folder)
            self.paths = sorted(self.image_folder.glob('*.*'))
            self.iterator = itertools.cycle(self.paths) if loop else iter(self.paths)
            self.cap = None
        else:
            self.cap = cv2.VideoCapture(source)
            self.paths = None

    def read(self):
        """
        Returns:
            ret (bool): True if frame is returned, False otherwise.
            frame (ndarray): the BGR image frame.
        """
        if self.cap:
            ret, frame = self.cap.read()
            if not ret:
                return ret, frame
            # 추가: 왜곡 제거
            if self.camera_matrix is not None and self.dist_coefs is not None:
                frame = cv2.undistort(frame, self.camera_matrix, self.dist_coefs, None, self.camera_matrix)
            return ret, frame
        else:
            try:
                path = next(self.iterator)
                frame = cv2.imread(str(path))
                if frame is None:
                    return False, None
                # 추가: 왜곡 제거
                if self.camera_matrix is not None and self.dist_coefs is not None:
                    frame = cv2.undistort(frame, self.camera_matrix, self.dist_coefs, None, self.camera_matrix)
                return True, frame
            except StopIteration:
                return False, None

    def release(self):
        """
        Release video capture if using webcam or video file.
        """
        if self.cap:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
