from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .chessboard import ChessboardSpec, find_corners, load_gray_images

_MIN_PAIRS = 10


@dataclass
class StereoCalibrationResult:
    camera_matrix_left: np.ndarray
    dist_coeffs_left: np.ndarray
    camera_matrix_right: np.ndarray
    dist_coeffs_right: np.ndarray
    R: np.ndarray  # 右相機相對左相機旋轉
    T: np.ndarray  # 右相機相對左相機平移
    E: np.ndarray
    F: np.ndarray
    rms_error: float
    image_size: tuple[int, int]

    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray

    @property
    def baseline_mm(self) -> float:
        return float(np.linalg.norm(self.T))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            camera_matrix_left=self.camera_matrix_left,
            dist_coeffs_left=self.dist_coeffs_left,
            camera_matrix_right=self.camera_matrix_right,
            dist_coeffs_right=self.dist_coeffs_right,
            R=self.R,
            T=self.T,
            E=self.E,
            F=self.F,
            rms_error=np.array(self.rms_error),
            image_size=np.array(self.image_size),
            R1=self.R1,
            R2=self.R2,
            P1=self.P1,
            P2=self.P2,
            Q=self.Q,
        )

    @classmethod
    def load(cls, path: Path) -> "StereoCalibrationResult":
        d = np.load(path, allow_pickle=False)
        return cls(
            camera_matrix_left=d["camera_matrix_left"],
            dist_coeffs_left=d["dist_coeffs_left"],
            camera_matrix_right=d["camera_matrix_right"],
            dist_coeffs_right=d["dist_coeffs_right"],
            R=d["R"],
            T=d["T"],
            E=d["E"],
            F=d["F"],
            rms_error=float(d["rms_error"]),
            image_size=tuple(int(v) for v in d["image_size"]),
            R1=d["R1"],
            R2=d["R2"],
            P1=d["P1"],
            P2=d["P2"],
            Q=d["Q"],
        )


def _find_matched_corners(
    left_dir: Path, right_dir: Path, spec: ChessboardSpec
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    left_images = {p.name: gray for p, gray in load_gray_images(left_dir)}
    right_images = {p.name: gray for p, gray in load_gray_images(right_dir)}
    common_names = sorted(set(left_images) & set(right_images))

    if not common_names:
        raise ValueError(f"（left={left_dir}, right={right_dir}）")

    objp = spec.object_points()
    obj_points: list[np.ndarray] = []
    left_points: list[np.ndarray] = []
    right_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    for name in common_names:
        gray_l, gray_r = left_images[name], right_images[name]
        if image_size is None:
            image_size = (gray_l.shape[1], gray_l.shape[0])
        corners_l = find_corners(gray_l, spec)
        corners_r = find_corners(gray_r, spec)
        if corners_l is None or corners_r is None:
            continue
        obj_points.append(objp)
        left_points.append(corners_l)
        right_points.append(corners_r)

    assert image_size is not None
    return obj_points, left_points, right_points, image_size


def calibrate_stereo(
    left_dir: Path,
    right_dir: Path,
    spec: ChessboardSpec,
    target_error_px: float = 0.5,
) -> StereoCalibrationResult:
    obj_points, left_points, right_points, image_size = _find_matched_corners(
        left_dir, right_dir, spec
    )

    if len(obj_points) < _MIN_PAIRS:
        raise ValueError(
            f"同步偵測到棋盤格的組數僅{len(obj_points)}組 需至少{_MIN_PAIRS}組"
        )

    _, cm_l0, dc_l0, _, _ = cv2.calibrateCamera(
        obj_points, left_points, image_size, None, None
    )
    _, cm_r0, dc_r0, _, _ = cv2.calibrateCamera(
        obj_points, right_points, image_size, None, None
    )

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    rms, cm_l, dc_l, cm_r, dc_r, R, T, E, F = cv2.stereoCalibrate(
        obj_points,
        left_points,
        right_points,
        cm_l0,
        dc_l0,
        cm_r0,
        dc_r0,
        image_size,
        criteria=criteria,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS,
    )

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        cm_l,
        dc_l,
        cm_r,
        dc_r,
        image_size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )

    result = StereoCalibrationResult(
        camera_matrix_left=cm_l,
        dist_coeffs_left=dc_l,
        camera_matrix_right=cm_r,
        dist_coeffs_right=dc_r,
        R=R,
        T=T,
        E=E,
        F=F,
        rms_error=float(rms),
        image_size=image_size,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
    )

    if result.rms_error > target_error_px:
        print(f"雙目標定RMS誤差{result.rms_error:.4f}px 超出目標{target_error_px}px")

    print(f"雙目基線長度：{result.baseline_mm:.2f} mm")

    return result
