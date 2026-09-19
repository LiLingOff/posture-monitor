from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .charuco import CharucoBoardSpec, detect_charuco
from .chessboard import ChessboardSpec, find_corners, load_gray_images

_MIN_IMAGES = 10
_MIN_CHARUCO_CORNERS_PER_VIEW = 6


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
        # cv2.norm回傳的是sqrt(所有點誤差平方和)，要除以sqrt(N)才是該張影像的RMS。
        # 除以N（OpenCV官方教學的寫法）會把誤差低估sqrt(N)倍，35個角點就差5.9倍，
        # 會讓0.3px的品質門檻失去作用。這樣算出來的總RMS會等於calibrateCamera的回傳值。
        error = cv2.norm(img_points[i], projected, cv2.NORM_L2) / np.sqrt(len(projected))
        errors.append(float(error))
    return errors


def _run_mono_calibration(
    obj_points: list[np.ndarray],
    img_points: list[np.ndarray],
    image_size: tuple[int, int],
    used_paths: list[Path],
    target_error_px: float,
) -> MonoCalibrationResult:
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


def calibrate_mono(
    image_dir: Path,
    spec: ChessboardSpec,
    target_error_px: float = 0.3,
) -> MonoCalibrationResult:
    images = load_gray_images(image_dir)
    if len(images) < _MIN_IMAGES:
        raise ValueError(f"標定影像過少（{len(images)}張），至少要{_MIN_IMAGES}張、建議30–40張。資料夾：{image_dir}")

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
            f"成功偵測到棋盤格的影像只有{len(obj_points)}張，需至少{_MIN_IMAGES}張。"
            f"棋盤格要完整入鏡且對焦清楚，或確認--cols/--rows與實際板子相符"
        )

    assert image_size is not None
    return _run_mono_calibration(obj_points, img_points, image_size, used_paths, target_error_px)


def calibrate_mono_charuco(
    image_dir: Path,
    board_spec: CharucoBoardSpec,
    target_error_px: float = 0.3,
    min_corners_per_view: int = _MIN_CHARUCO_CORNERS_PER_VIEW,
) -> MonoCalibrationResult:
    """ChArUco版單眼標定。每張影像只要偵測到足夠多的角點就能使用，不需要整塊board入鏡。"""
    board = board_spec.build_board()
    board_obj_points = board.getChessboardCorners()
    images = load_gray_images(image_dir)
    if len(images) < _MIN_IMAGES:
        raise ValueError(f"標定影像過少（{len(images)}張），至少要{_MIN_IMAGES}張、建議30–40張。資料夾：{image_dir}")

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    used_paths: list[Path] = []
    image_size: tuple[int, int] | None = None

    for path, gray in images:
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        detected = detect_charuco(gray, board, min_corners=min_corners_per_view)
        if detected is None:
            continue
        corners, ids = detected
        obj_points.append(board_obj_points[ids.flatten()].astype(np.float32))
        img_points.append(corners.reshape(-1, 1, 2).astype(np.float32))
        used_paths.append(path)

    if len(obj_points) < _MIN_IMAGES:
        raise ValueError(
            f"偵測到足夠角點的影像只有{len(obj_points)}張，需至少{_MIN_IMAGES}張。"
            f"確認--squares-x/--squares-y/--dictionary與實際板子相符，或加上--legacy-pattern再試"
        )

    assert image_size is not None
    return _run_mono_calibration(obj_points, img_points, image_size, used_paths, target_error_px)
