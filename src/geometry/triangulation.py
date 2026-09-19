"""雙目3D三角測量。

正確流程：先用undistortPoints把原始像素點去畸變＋套用stereoRectify的校正轉換
（R1/R2/P1/P2定義在校正後的座標系，不能把原始像素直接輸入triangulatePoints），
再用triangulatePoints三角測量，最後除以齊次座標第4維還原3D點。
"""
from __future__ import annotations

import cv2
import numpy as np

from calibration.stereo_calibration import StereoCalibrationResult


def triangulate_points(
    calib: StereoCalibrationResult, points_left: np.ndarray, points_right: np.ndarray
) -> np.ndarray:
    """points_left/points_right shape(N,2)，可含NaN列(未偵測到)。

    回傳shape(N,3)，單位與標定時square_size_mm一致(mm)；NaN列的輸出也是NaN。
    """
    points_left = np.asarray(points_left, dtype=np.float64)
    points_right = np.asarray(points_right, dtype=np.float64)
    n = len(points_left)
    result = np.full((n, 3), np.nan, dtype=np.float64)

    valid = ~np.isnan(points_left).any(axis=1) & ~np.isnan(points_right).any(axis=1)
    if not valid.any():
        return result

    pts_l = points_left[valid].reshape(-1, 1, 2)
    pts_r = points_right[valid].reshape(-1, 1, 2)

    rect_l = cv2.undistortPoints(
        pts_l, calib.camera_matrix_left, calib.dist_coeffs_left, R=calib.R1, P=calib.P1
    )
    rect_r = cv2.undistortPoints(
        pts_r, calib.camera_matrix_right, calib.dist_coeffs_right, R=calib.R2, P=calib.P2
    )

    points_4d = cv2.triangulatePoints(
        calib.P1, calib.P2, rect_l.reshape(-1, 2).T, rect_r.reshape(-1, 2).T
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        points_3d = (points_4d[:3] / points_4d[3]).T

    # 齊次座標第4維趨近0代表兩條視線幾乎平行、交點在無窮遠（左右對應點幾乎重合時會這樣）。
    # 這種結果是inf或極大值，不處理會混入角度計算變成看似合理的數字，統一標記為NaN。
    points_3d[~np.isfinite(points_3d).all(axis=1)] = np.nan

    result[valid] = points_3d
    return result


def triangulate_point(
    calib: StereoCalibrationResult, point_left: np.ndarray, point_right: np.ndarray
) -> np.ndarray:
    """單點版本，回傳shape(3,)。"""
    return triangulate_points(calib, np.asarray(point_left).reshape(1, 2), np.asarray(point_right).reshape(1, 2))[0]
