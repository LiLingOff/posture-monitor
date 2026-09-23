from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class ChessboardSpec:
    cols: int
    rows: int
    square_size_mm: float

    def object_points(self) -> np.ndarray:
        objp = np.zeros((self.rows * self.cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0 : self.cols, 0 : self.rows].T.reshape(-1, 2)
        objp *= self.square_size_mm
        return objp


_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
_FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE


def find_corners(
    gray: np.ndarray, spec: ChessboardSpec, fast_check: bool = False
) -> np.ndarray | None:
    """偵測棋盤格內角點，回傳shape(N,1,2)或None。

    fast_check對應cv2.CALIB_CB_FAST_CHECK：它是為即時預覽設計的提早退出啟發式，
    畫面裡沒有棋盤格時可以大幅省時，代價是對比不足或輕微失焦的影像可能誤判成沒有。
    拍攝迴圈要即時回饋所以開啟；離線標定沒有速度壓力，關掉以免漏掉本來可用的影像。
    """
    flags = _FIND_FLAGS + (cv2.CALIB_CB_FAST_CHECK if fast_check else 0)
    found, corners = cv2.findChessboardCorners(gray, (spec.cols, spec.rows), flags)
    if not found:
        return None
    corners = corners.reshape(-1, 1, 2).astype(np.float32)
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _SUBPIX_CRITERIA)
    return corners


# 副檔名一律轉小寫比對：手機匯出的.JPG、.jpeg被略過的話，
# 使用者只會看到「影像過少」，很難聯想到是副檔名沒對上。
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


def load_gray_images(image_dir: Path) -> list[tuple[Path, np.ndarray]]:
    if not image_dir.is_dir():
        return []
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
    images = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        images.append((p, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
    return images


def ensure_uniform_size(images: Sequence[tuple[Path, np.ndarray]]) -> tuple[int, int]:
    """確認所有影像解析度一致並回傳(width, height)。

    內參fx/fy/cx/cy的單位是像素、綁定於當時的解析度。資料夾裡混到不同解析度的影像
    （換過拍攝模式、舊檔案沒清乾淨）時，擬合出來的參數沒有意義，
    而OpenCV只會照算不會抱怨——這正是最難察覺的一類錯誤。
    """
    if not images:
        raise ValueError("沒有可用的影像")
    sizes = {(gray.shape[1], gray.shape[0]) for _, gray in images}
    if len(sizes) > 1:
        detail = "、".join(f"{w}x{h}" for w, h in sorted(sizes))
        raise ValueError(
            f"資料夾內影像解析度不一致（{detail}）。內參綁定於解析度，"
            f"混用會得到無意義的結果；請只保留同一個解析度拍攝的影像"
        )
    return sizes.pop()
