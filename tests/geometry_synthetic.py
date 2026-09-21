"""測試用：合成雙目標定結果與3D點投影，供三角測量驗證用。

與tests/synthetic.py不同——那份是homography-based，假設標的物是平面(Z=0)，
三角測量要驗證的是任意非共平面的3D點，因此這裡直接用cv2.projectPoints合成。
"""
from __future__ import annotations

import cv2
import numpy as np

from calibration.stereo_calibration import StereoCalibrationResult


def make_synthetic_stereo_calibration(
    K: np.ndarray, baseline_mm: float, image_size: tuple[int, int] = (640, 480)
) -> StereoCalibrationResult:
    """零畸變、純X軸平移基線，第二台相機位於第一台右側baseline_mm處。

    T是「把第一台相機座標系的點轉到第二台」的平移。第二台在右側b mm時，
    第一台的原點在它眼中落在x=-b，所以T_x是負的——這與OpenCV的
    stereoCalibrate在真實左右並排模組上算出來的正負號一致。
    """
    dist = np.zeros(5)
    R = np.eye(3)
    T = np.array([[-baseline_mm], [0.0], [0.0]])
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K, dist, K, dist, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    return StereoCalibrationResult(
        camera_matrix_left=K,
        dist_coeffs_left=dist,
        camera_matrix_right=K,
        dist_coeffs_right=dist,
        R=R,
        T=T,
        E=np.eye(3),
        F=np.eye(3),
        rms_error=0.0,
        image_size=image_size,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
    )


def project_point(K: np.ndarray, R: np.ndarray, T: np.ndarray, point_3d: np.ndarray) -> np.ndarray:
    """將世界座標點投影到指定相機的像素座標(零畸變)，回傳shape(2,)。"""
    rvec, _ = cv2.Rodrigues(R)
    img_pts, _ = cv2.projectPoints(
        np.asarray(point_3d, dtype=np.float64).reshape(1, 1, 3), rvec, T, K, np.zeros(5)
    )
    return img_pts.reshape(2)
