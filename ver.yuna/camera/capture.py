import cv2
import time
import logging
from pathlib import Path
import itertools

class FrameCapture:
    """
    Webcam or image folder frame provider.

    Args:
        source (int or str): Webcam index (for webcam) or video file path.
        image_folder (str): Path to a folder of image files. If provided, frames are read from images.
        loop (bool): If True and image_folder is set, cycle through images indefinitely.
        fps (float): Desired frame rate. Used to adjust frame capture interval.
    """
    def __init__(self, source=0, image_folder: str = None, loop: bool = False, fps: float = 15.0):
        self.loop = loop
        self.fps = fps
        self.last_frame_time = time.time()  # To control frame capture rate (non-blocking)
        
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
            frame (ndarray): The BGR image frame.
        """
        # 비차단 방식으로 웹캡처 또는 이미지 폴더에서 프레임을 처리
        current_time = time.time()
        
        # 현재 시간과 이전 캡처 시간을 비교하여 프레임을 처리할지 결정
        if current_time - self.last_frame_time >= 1 / self.fps:
            self.last_frame_time = current_time  # Update last capture time
            
            if self.cap:  # 웹캡 캡처인 경우
                ret, frame = self.cap.read()
                if ret:
                    return ret, frame
            else:  # 이미지 폴더에서 프레임 읽기
                try:
                    path = next(self.iterator)
                    frame = cv2.imread(str(path))
                    return True, frame
                except StopIteration:
                    return False, None

        # 시간이 충분히 경과하지 않으면 이전 프레임을 그대로 반환
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
