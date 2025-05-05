import os
import glob
import cv2
import numpy as np
import logging

def calibrate_camera(
    image_dir: str,
    pattern_size: tuple = (9, 6),
    square_size: float = 0.025,
    save_path: str = None
):
    """
    체커보드 패턴 이미지로 카메라 내부 파라미터와 왜곡 계수 계산

    Args:
        image_dir: 체커보드 이미지들이 저장된 디렉터리 경로
        pattern_size: 체커보드 코너 개수 (가로, 세로)
        square_size: 체커보드 각 칸의 실제 크기 (미터)
        save_path: 결과를 저장할 .npz 파일 경로 (optional)

    Returns:
        camera_matrix: 3x3 내부 파라미터 행렬
        dist_coefs: 왜곡 계수 벡터
        rvecs, tvecs: 각 이미지별 회전·병진 벡터 리스트
    """
    # 1) 객체 좌표 준비 (체커보드 패턴)
    objp = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []  # 3D 점
    imgpoints = []  # 2D 점

    # 2) 이미지들 로드 및 코너 검출
    images = glob.glob(os.path.join(image_dir, '*.jpg'))
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if not ret:
            continue
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)

    if not objpoints:
        raise RuntimeError(f"No chessboard corners found in '{image_dir}'")

    # 3) 칼리브레이션 수행
    ret, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    # 4) 결과 저장
    if save_path:
        np.savez(save_path, camera_matrix=camera_matrix, dist_coefs=dist_coefs)

    return camera_matrix, dist_coefs, rvecs, tvecs


def load_camera_parameters(path='../../config/camera_params.npz'):
    """
    저장된 calibration 결과(.npz) 파일에서 파라미터 로드

    Args:
        path: .npz 파일 경로 (없으면 config/camera_params.npz)
    Returns:
        (camera_matrix, dist_coefs)
    """
    if not os.path.exists(path):
        logging.warning(f"'{path}' 파일이 없습니다. 기본값을 사용합니다.")
        # 임시 기본값 리턴 (테스트용)
        cam_mtx = np.eye(3)
        dist_coefs = np.zeros(5)
        return cam_mtx, dist_coefs

    data = np.load(path)
    cam_mtx = data['camera_matrix']
    dist_coefs = data['distortion_coefficients']
    return cam_mtx, dist_coefs