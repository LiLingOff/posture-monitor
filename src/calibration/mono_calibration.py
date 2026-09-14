from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .chessboard import ChessboardSpec, find_corners, load_gray_images

_MIN_IMAGES = 10


@dataclass
class MonoCalibrationResult:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]
    per_view_errors: list[float]
    used_images: list[Path]

    @property
    def rms_reprojection_error(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.per_view_errors))))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            image_size=np.array(self.image_size),
            per_view_errors=np.array(self.per_view_errors),
        )

    @classmethod
    def load(cls, path: Path) -> "MonoCalibrationResult":
        data = np.load(path, allow_pickle=False)
        return cls(
            camera_matrix=data["camera_matrix"],
            dist_coeffs=data["dist_coeffs"],
            image_size=tuple(int(v) for v in data["image_size"]),
            per_view_errors=list(data["per_view_errors"]),
            used_images=[],
        )


def _per_view_reprojection_errors(
    obj_points: list[np.ndarray],
    img_points: list[np.ndarray],
    rvecs: tuple[np.ndarray, ...],
    tvecs: tuple[np.ndarray, ...],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[float]:
    errors = []
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected)
        errors.append(float(error))
    return errors


def calibrate_mono(
    image_dir: Path,
    spec: ChessboardSpec,
    target_error_px: float = 0.3,
) -> MonoCalibrationResult:
    images = load_gray_images(image_dir)
    if len(images) < _MIN_IMAGES:
        raise ValueError(f"標定影像過少({len(images)}張) 目前資料夾：{image_dir}")

    objp = spec.object_points()
    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    used_paths: list[Path] = []
    image_size: tuple[int, int] | None = None

    for path, gray in images:
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        corners = find_corners(gray, spec)
        if corners is None:
            continue
        obj_points.append(objp)
        img_points.append(corners)
        used_paths.append(path)

    if len(obj_points) < _MIN_IMAGES:
        raise ValueError(
            f"同步偵測到棋盤格的組數僅{len(obj_points)}張 需至少{_MIN_IMAGES}張"
        )

    assert image_size is not None
    _, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    per_view_errors = _per_view_reprojection_errors(
        obj_points, img_points, rvecs, tvecs, camera_matrix, dist_coeffs
    )

    result = MonoCalibrationResult(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,
        per_view_errors=per_view_errors,
        used_images=used_paths,
    )

    if result.rms_reprojection_error > target_error_px:
        print(
            f"RMS重投影誤差{result.rms_reprojection_error:.4f}px 超出目標{target_error_px}px"
        )

    return result
