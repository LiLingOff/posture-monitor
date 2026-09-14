"""測試用：合成ChArUco board的相機視角影像。

原理跟tests/synthetic.py相同（平面標的物的相機成像=正面圖的一次homography變換），
差別是frontal->object的mm對應關係改用實際偵測結果反推（cv2.findHomography），
不依賴假設board的內部座標系原點/方向，對OpenCV版本間的座標慣例差異更穩健。
"""
from __future__ import annotations

import cv2
import numpy as np

from calibration.charuco import CharucoBoardSpec


def make_charuco_frontal_image(board_spec: CharucoBoardSpec, pixels_per_square: int = 80) -> np.ndarray:
    board = board_spec.build_board()
    size_px = (board_spec.squares_x * pixels_per_square, board_spec.squares_y * pixels_per_square)
    return board.generateImage(size_px)


def frontal_to_object_homography(frontal_img: np.ndarray, board_spec: CharucoBoardSpec) -> np.ndarray:
    """在無形變的正面圖上偵測角點，用(像素座標, 已知mm座標)反推homography。"""
    board = board_spec.build_board()
    detector = cv2.aruco.CharucoDetector(board)
    corners, ids, _, _ = detector.detectBoard(frontal_img)
    obj_points_all = board.getChessboardCorners()
    obj_pts_2d = obj_points_all[ids.flatten()][:, :2].astype(np.float32)
    img_pts = corners.reshape(-1, 2).astype(np.float32)
    H, _ = cv2.findHomography(img_pts, obj_pts_2d)
    return H


def synthesize_charuco_view(
    frontal_img: np.ndarray,
    frontal_to_obj: np.ndarray,
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    dest_size: tuple[int, int],
) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    H_cam = K @ np.column_stack([R[:, 0], R[:, 1], tvec.reshape(3)])
    H = H_cam @ frontal_to_obj
    return cv2.warpPerspective(frontal_img, H, dest_size, flags=cv2.INTER_LINEAR, borderValue=200)
