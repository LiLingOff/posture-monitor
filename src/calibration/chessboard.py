from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
_FIND_FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    + cv2.CALIB_CB_NORMALIZE_IMAGE
    + cv2.CALIB_CB_FAST_CHECK
)


def find_corners(gray: np.ndarray, spec: ChessboardSpec) -> np.ndarray | None:
    found, corners = cv2.findChessboardCorners(
        gray, (spec.cols, spec.rows), _FIND_FLAGS
    )
    if not found:
        return None
    corners = corners.reshape(-1, 1, 2).astype(np.float32)
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _SUBPIX_CRITERIA)
    return corners


def load_gray_images(image_dir: Path) -> list[tuple[Path, np.ndarray]]:
    paths = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
    images = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        images.append((p, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
    return images
