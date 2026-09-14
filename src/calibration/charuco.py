"""ChArUco板角點偵測。

跟chessboard.py的差異：每個角點有獨立ID（靠ArUco標記辨識），
影像不需要看到完整board、只要偵測到夠多角點就能用，
用來解決雙目模組兩顆鏡頭視野重疊區域小、無法同時拍到完整棋盤格的問題。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_ARUCO_DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
}


@dataclass(frozen=True)
class CharucoBoardSpec:
    squares_x: int
    squares_y: int
    square_size_mm: float
    marker_size_mm: float
    dictionary_name: str = "DICT_5X5_100"
    # 部分現成板子（例如AndyMark）用OpenCV 4.6以前的舊版ArUco標記排列方式，
    # 若偵測不到角點或角點對不上，改成True試試看。
    legacy_pattern: bool = False

    def build_board(self) -> cv2.aruco.CharucoBoard:
        if self.dictionary_name not in _ARUCO_DICTIONARIES:
            raise ValueError(
                f"不支援的dictionary_name={self.dictionary_name}，"
                f"可用：{sorted(_ARUCO_DICTIONARIES)}"
            )
        dictionary = cv2.aruco.getPredefinedDictionary(_ARUCO_DICTIONARIES[self.dictionary_name])
        board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y), self.square_size_mm, self.marker_size_mm, dictionary
        )
        board.setLegacyPattern(self.legacy_pattern)
        return board


def save_board_image(spec: CharucoBoardSpec, path: Path, pixels_per_square: int = 80) -> None:
    """產生board圖檔供列印。"""
    board = spec.build_board()
    size_px = (spec.squares_x * pixels_per_square, spec.squares_y * pixels_per_square)
    img = board.generateImage(size_px)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def detect_charuco(
    gray: np.ndarray, board: cv2.aruco.CharucoBoard, min_corners: int = 1
) -> tuple[np.ndarray, np.ndarray] | None:
    """偵測ChArUco角點，回傳(charucoCorners shape(N,1,2), charucoIds shape(N,1))。

    可能只偵測到board的一部分，N可以小於board實際角點總數。
    偵測到的角點數少於min_corners時回傳None。
    """
    detector = cv2.aruco.CharucoDetector(board)
    corners, ids, _, _ = detector.detectBoard(gray)
    if corners is None or ids is None or len(ids) < min_corners:
        return None
    return corners, ids
