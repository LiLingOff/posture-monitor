"""雙目3D三角測量。

正確流程：先用undistortPoints把原始像素點去畸變＋套用stereoRectify的校正轉換
（R1/R2/P1/P2定義在校正後的座標系，不能把原始像素直接輸入triangulatePoints），
再用triangulatePoints三角測量，最後除以齊次座標第4維還原3D點。
"""
from __future__ import annotations

import cv2
import numpy as np

from calibration.stereo_calibration import StereoCalibrationResult


def _paired_points(points_left, points_right) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """整理左右點並回傳哪些列兩邊都有限。形狀不對就明確報錯，不留給 OpenCV。

    空輸入（空 list）要當成零個點，不是形狀錯誤。inf 與 NaN 一樣當成沒偵測到：
    傳進 OpenCV 之後不同版本會回 NaN、inf 或直接 assertion failed。
    """
    left = np.asarray(points_left, dtype=np.float64)
    right = np.asarray(points_right, dtype=np.float64)
    left = left.reshape(0, 2) if left.size == 0 else left
    right = right.reshape(0, 2) if right.size == 0 else right
    for name, arr in (("points_left", left), ("points_right", right)):
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"{name} 形狀應為 (N, 2)，實際是 {arr.shape}")
    if len(left) != len(right):
        raise ValueError(f"左右點數不同：左 {len(left)}、右 {len(right)}")
    valid = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
    return left, right, valid


def triangulate_points(
    calib: StereoCalibrationResult, points_left: np.ndarray, points_right: np.ndarray
) -> np.ndarray:
    """points_left/points_right shape(N,2)，可含NaN列(未偵測到)。

    回傳shape(N,3)，單位與標定時square_size_mm一致(mm)；NaN列的輸出也是NaN。
    """
    points_left, points_right, valid = _paired_points(points_left, points_right)
    n = len(points_left)
    result = np.full((n, 3), np.nan, dtype=np.float64)
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
    # 這種結果是inf或極大值，不處理的話會混進角度計算，統一標記為NaN。
    points_3d[~np.isfinite(points_3d).all(axis=1)] = np.nan

    result[valid] = points_3d
    return result


def triangulate_point(
    calib: StereoCalibrationResult, point_left: np.ndarray, point_right: np.ndarray
) -> np.ndarray:
    """單點版本，回傳shape(3,)。"""
    return triangulate_points(calib, np.asarray(point_left).reshape(1, 2), np.asarray(point_right).reshape(1, 2))[0]


def rectified_vertical_disparity(
    calib: StereoCalibrationResult, points_left: np.ndarray, points_right: np.ndarray
) -> np.ndarray:
    """校正後左右對應點的垂直座標差（像素），shape(N,)。

    stereoRectify的目的就是讓對極線變成水平線，所以校正後同一個點在左右影像的
    y座標應該幾乎相同。這個差值偏大，代表標定不準或左右配對錯誤。
    而三角測量不會因此報錯，它只會把兩條不相交的視線取最近點，
    照樣給出一個數量級正常的3D座標。

    這是少數能用真實資料（不是標定板）檢驗標定品質的指標。
    """
    points_left, points_right, valid = _paired_points(points_left, points_right)
    n = len(points_left)
    result = np.full(n, np.nan, dtype=np.float64)
    if not valid.any():
        return result

    rect_l = cv2.undistortPoints(
        points_left[valid].reshape(-1, 1, 2),
        calib.camera_matrix_left, calib.dist_coeffs_left, R=calib.R1, P=calib.P1,
    ).reshape(-1, 2)
    rect_r = cv2.undistortPoints(
        points_right[valid].reshape(-1, 1, 2),
        calib.camera_matrix_right, calib.dist_coeffs_right, R=calib.R2, P=calib.P2,
    ).reshape(-1, 2)

    result[valid] = rect_l[:, 1] - rect_r[:, 1]
    return result
