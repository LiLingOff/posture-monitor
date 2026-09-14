"""ChArUco版標定測試。

重點驗證：即使沒有任何一幀能拍到完整board（視野重疊區域小的情境），
只要每幀有足夠共同角點，標定依然能成功並還原正確的內參/基線長度。
"""
from pathlib import Path

import cv2
import numpy as np

from calibration.charuco import CharucoBoardSpec, detect_charuco
from calibration.mono_calibration import calibrate_mono_charuco
from calibration.stereo_calibration import calibrate_stereo_charuco
from charuco_synthetic import frontal_to_object_homography, make_charuco_frontal_image, synthesize_charuco_view

DEST_SIZE = (640, 480)
TRUE_K = np.array(
    [
        [700.0, 0.0, 320.0],
        [0.0, 700.0, 240.0],
        [0.0, 0.0, 1.0],
    ]
)
BOARD_SPEC = CharucoBoardSpec(squares_x=10, squares_y=8, square_size_mm=25.0, marker_size_mm=18.0)
TRUE_BASELINE_MM = 70.0


def _frontal_and_homography():
    frontal = make_charuco_frontal_image(BOARD_SPEC)
    H = frontal_to_object_homography(frontal, BOARD_SPEC)
    return frontal, H


def _random_pose(rng: np.random.Generator, pattern_w_mm: float, pattern_h_mm: float) -> tuple[np.ndarray, np.ndarray]:
    rvec = rng.uniform(-0.1, 0.1, size=3)
    z = rng.uniform(200.0, 240.0)  # 近距離：任一單一鏡頭都看不到完整board
    jx = rng.uniform(-15.0, 15.0)
    jy = rng.uniform(-10.0, 10.0)
    tvec = np.array([-pattern_w_mm / 2 + jx, -pattern_h_mm / 2 + jy, z])
    return rvec, tvec


def test_detect_charuco_partial_view_no_full_board():
    """驗證近距離下確實沒有單一視角能看到完整board（測試前提成立）。"""
    frontal, H = _frontal_and_homography()
    board = BOARD_SPEC.build_board()
    total_corners = board.getChessboardCorners().shape[0]
    pattern_w_mm = (BOARD_SPEC.squares_x - 1) * BOARD_SPEC.square_size_mm
    pattern_h_mm = (BOARD_SPEC.squares_y - 1) * BOARD_SPEC.square_size_mm

    rng = np.random.default_rng(0)
    for _ in range(10):
        rvec, tvec = _random_pose(rng, pattern_w_mm, pattern_h_mm)
        view = synthesize_charuco_view(frontal, H, TRUE_K, rvec, tvec, DEST_SIZE)
        detected = detect_charuco(view, board, min_corners=1)
        assert detected is not None
        assert len(detected[1]) < total_corners


def test_mono_calibration_charuco_with_partial_views(tmp_path: Path):
    frontal, H = _frontal_and_homography()
    pattern_w_mm = (BOARD_SPEC.squares_x - 1) * BOARD_SPEC.square_size_mm
    pattern_h_mm = (BOARD_SPEC.squares_y - 1) * BOARD_SPEC.square_size_mm

    rng = np.random.default_rng(42)
    image_dir = tmp_path / "front"
    image_dir.mkdir()
    for i in range(18):
        rvec, tvec = _random_pose(rng, pattern_w_mm, pattern_h_mm)
        view = synthesize_charuco_view(frontal, H, TRUE_K, rvec, tvec, DEST_SIZE)
        cv2.imwrite(str(image_dir / f"frame_{i:04d}.png"), view)

    result = calibrate_mono_charuco(image_dir, BOARD_SPEC, target_error_px=1.0)

    assert len(result.used_images) >= 15
    fx, fy = result.camera_matrix[0, 0], result.camera_matrix[1, 1]
    assert abs(fx - TRUE_K[0, 0]) / TRUE_K[0, 0] < 0.05
    assert abs(fy - TRUE_K[1, 1]) / TRUE_K[1, 1] < 0.05


def test_stereo_calibration_charuco_with_partial_overlap(tmp_path: Path):
    frontal, H = _frontal_and_homography()
    pattern_w_mm = (BOARD_SPEC.squares_x - 1) * BOARD_SPEC.square_size_mm
    pattern_h_mm = (BOARD_SPEC.squares_y - 1) * BOARD_SPEC.square_size_mm

    rng = np.random.default_rng(7)
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()

    for i in range(20):
        rvec, tvec_l = _random_pose(rng, pattern_w_mm, pattern_h_mm)
        tvec_r = tvec_l + np.array([TRUE_BASELINE_MM, 0.0, 0.0])
        view_l = synthesize_charuco_view(frontal, H, TRUE_K, rvec, tvec_l, DEST_SIZE)
        view_r = synthesize_charuco_view(frontal, H, TRUE_K, rvec, tvec_r, DEST_SIZE)
        name = f"frame_{i:04d}.png"
        cv2.imwrite(str(left_dir / name), view_l)
        cv2.imwrite(str(right_dir / name), view_r)

    result = calibrate_stereo_charuco(left_dir, right_dir, BOARD_SPEC, target_error_px=1.0)

    assert abs(result.baseline_mm - TRUE_BASELINE_MM) / TRUE_BASELINE_MM < 0.1
    identity_deviation = np.linalg.norm(result.R - np.eye(3))
    assert identity_deviation < 0.05


def test_stereo_calibration_charuco_raises_when_no_shared_corners(tmp_path: Path):
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()

    blank = np.full((480, 640, 3), 255, dtype=np.uint8)
    for i in range(3):
        name = f"frame_{i:04d}.png"
        cv2.imwrite(str(left_dir / name), blank)
        cv2.imwrite(str(right_dir / name), blank)

    try:
        calibrate_stereo_charuco(left_dir, right_dir, BOARD_SPEC)
        assert False, "應拋出 ValueError"
    except ValueError as e:
        assert "組" in str(e)
