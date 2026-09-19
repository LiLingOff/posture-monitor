"""測試用：以已知相機內參/外參合成棋盤格影像。

原理：棋盤格是平面標的物，一個無畸變針孔相機對平面的成像
等價於對正面平視影像做一次單應變換（homography）：
    H = K @ [r1 r2 t]
其中 r1, r2 為旋轉矩陣前兩欄、t 為平移向量。
因此對高解析度正面棋盤格圖套用不同姿態算出的 H 做 warpPerspective，
即可產生幾何上等價於真實相機在該姿態拍到的影像，
可用來驗證 calibrateCamera 與 stereoCalibrate 的正確性。
"""
from __future__ import annotations

import cv2
import numpy as np

from calibration.chessboard import ChessboardSpec

PX_PER_MM = 40.0 / 25.0  # 正面圖每mm對應的像素數（供frontal image生成用）


def make_frontal_chessboard_image(spec: ChessboardSpec, margin_squares: int = 2) -> np.ndarray:
    """產生一張正面平視的棋盤格灰階圖，方格邊長換算為 PX_PER_MM 像素/mm。

    棋盤格本體外圍留白邊界（quiet zone），只在留白內繪製棋盤格圖案，
    否則 findChessboardCorners 會因缺乏清楚邊界而偵測失敗。
    """
    square_px = int(round(spec.square_size_mm * PX_PER_MM))
    pattern_cols = spec.cols + 1
    pattern_rows = spec.rows + 1
    cols_squares = pattern_cols + 2 * margin_squares
    rows_squares = pattern_rows + 2 * margin_squares
    img = np.full((rows_squares * square_px, cols_squares * square_px), 255, dtype=np.uint8)

    for r in range(pattern_rows):
        for c in range(pattern_cols):
            if (r + c) % 2 == 0:
                y0 = (r + margin_squares) * square_px
                y1 = y0 + square_px
                x0 = (c + margin_squares) * square_px
                x1 = x0 + square_px
                img[y0:y1, x0:x1] = 0

    return img


def frontal_to_object_offset_squares(margin_squares: int) -> np.ndarray:
    """第一個內角點（物件座標原點）在正面圖中的位置，以方格數表示。"""
    return np.array([margin_squares, margin_squares], dtype=np.float64)


def synthesize_view(
    frontal_img: np.ndarray,
    spec: ChessboardSpec,
    margin_squares: int,
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    dest_size: tuple[int, int],
) -> np.ndarray:
    """依相機姿態，將正面棋盤格圖 warp 成該姿態下的合成相機影像。"""
    R, _ = cv2.Rodrigues(rvec)
    H_cam = K @ np.column_stack([R[:, 0], R[:, 1], tvec.reshape(3)])

    offset_squares = frontal_to_object_offset_squares(margin_squares)
    # frontal 像素 (u,v) -> 物件平面 mm 座標 (X,Y): X = u/PX_PER_MM - offset_x_mm
    H_frontal_to_obj = np.array(
        [
            [1.0 / PX_PER_MM, 0, -offset_squares[0] * spec.square_size_mm],
            [0, 1.0 / PX_PER_MM, -offset_squares[1] * spec.square_size_mm],
            [0, 0, 1.0],
        ]
    )

    H = H_cam @ H_frontal_to_obj
    return cv2.warpPerspective(frontal_img, H, dest_size, flags=cv2.INTER_LINEAR, borderValue=200)
