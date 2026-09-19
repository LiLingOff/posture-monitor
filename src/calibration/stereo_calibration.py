from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .charuco import CharucoBoardSpec, detect_charuco
from .chessboard import ChessboardSpec, find_corners, load_gray_images

_MIN_PAIRS = 10
_MIN_SHARED_CHARUCO_CORNERS = 6


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
        raise ValueError(f"左右資料夾沒有同名檔案可配對，雙目標定需要同一時刻拍的成對影像。"
            f"（left={left_dir}, right={right_dir}）")

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


def _run_stereo_calibration(
    obj_points: list[np.ndarray],
    left_points: list[np.ndarray],
    right_points: list[np.ndarray],
    image_size: tuple[int, int],
    target_error_px: float,
) -> StereoCalibrationResult:
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
            f"左右同時偵測到完整棋盤格的只有{len(obj_points)}組，需至少{_MIN_PAIRS}組。"
            f"兩顆鏡頭視野重疊區過小時改用ChArUco板（--charuco），不需要整塊板子同時入鏡"
        )

    return _run_stereo_calibration(obj_points, left_points, right_points, image_size, target_error_px)


def _charuco_frame_correspondence(
    gray_l: np.ndarray,
    gray_r: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    board_obj_points: np.ndarray,
    min_shared: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """算左右影像在同一幀共同偵測到的角點，回傳(objp, imgp_l, imgp_r)。

    左右鏡頭各自可能只看到board的一部分，只要共同角點數>=min_shared就能使用，
    不需要整塊board同時入鏡，藉此解決視野重疊區域過小的問題。
    """
    det_l = detect_charuco(gray_l, board, min_corners=1)
    det_r = detect_charuco(gray_r, board, min_corners=1)
    if det_l is None or det_r is None:
        return None

    corners_l, ids_l = det_l
    corners_r, ids_r = det_r
    by_id_l = dict(zip(ids_l.flatten().tolist(), corners_l.reshape(-1, 2)))
    by_id_r = dict(zip(ids_r.flatten().tolist(), corners_r.reshape(-1, 2)))
    shared_ids = sorted(set(by_id_l) & set(by_id_r))

    if len(shared_ids) < min_shared:
        return None

    objp = board_obj_points[shared_ids].astype(np.float32)
    imgp_l = np.array([by_id_l[i] for i in shared_ids], dtype=np.float32).reshape(-1, 1, 2)
    imgp_r = np.array([by_id_r[i] for i in shared_ids], dtype=np.float32).reshape(-1, 1, 2)
    return objp, imgp_l, imgp_r


def calibrate_stereo_charuco(
    left_dir: Path,
    right_dir: Path,
    board_spec: CharucoBoardSpec,
    target_error_px: float = 0.5,
    min_shared_corners: int = _MIN_SHARED_CHARUCO_CORNERS,
) -> StereoCalibrationResult:
    """ChArUco版雙目標定。每一幀左右影像只要有足夠共同角點就能使用，
    不需要兩顆鏡頭同時拍到完整棋盤格。
    """
    board = board_spec.build_board()
    board_obj_points = board.getChessboardCorners()

    left_images = {p.name: gray for p, gray in load_gray_images(left_dir)}
    right_images = {p.name: gray for p, gray in load_gray_images(right_dir)}
    common_names = sorted(set(left_images) & set(right_images))

    if not common_names:
        raise ValueError(f"左右資料夾無同名檔案可配對（left={left_dir}, right={right_dir}）")

    obj_points: list[np.ndarray] = []
    left_points: list[np.ndarray] = []
    right_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    for name in common_names:
        gray_l, gray_r = left_images[name], right_images[name]
        if image_size is None:
            image_size = (gray_l.shape[1], gray_l.shape[0])
        corr = _charuco_frame_correspondence(gray_l, gray_r, board, board_obj_points, min_shared_corners)
        if corr is None:
            continue
        objp, imgp_l, imgp_r = corr
        obj_points.append(objp)
        left_points.append(imgp_l)
        right_points.append(imgp_r)

    if len(obj_points) < _MIN_PAIRS:
        raise ValueError(
            f"左右有足夠共同角點的只有{len(obj_points)}組，需至少{_MIN_PAIRS}組。"
            f"拍攝時讓板子多出現在兩邊視野的重疊區，或調低min_shared_corners"
        )

    assert image_size is not None
    return _run_stereo_calibration(obj_points, left_points, right_points, image_size, target_error_px)
