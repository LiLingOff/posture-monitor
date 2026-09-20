"""重投影誤差計算的回歸測試。

這個數字是標定品質的唯一判準（單眼<0.3px、雙目<0.5px），算錯不會有任何外顯症狀，
只會讓品質不佳的標定悄悄通過門檻，因此用已知位移量與OpenCV自己算出的RMS兩種方式鎖定。
"""
import cv2
import numpy as np
import pytest

from calibration.charuco import CharucoBoardSpec, detect_charuco
from calibration.chessboard import ChessboardSpec, find_corners
from calibration.mono_calibration import _per_view_reprojection_errors, calibrate_mono, calibrate_mono_charuco
from charuco_synthetic import (frontal_to_object_homography,
                               make_charuco_frontal_image,
                               synthesize_charuco_view)
from synthetic import make_frontal_chessboard_image, synthesize_view

K = np.array(
    [
        [700.0, 0.0, 320.0],
        [0.0, 700.0, 240.0],
        [0.0, 0.0, 1.0],
    ]
)
SPEC = ChessboardSpec(cols=7, rows=5, square_size_mm=25.0)


def test_per_view_error_equals_known_uniform_offset():
    """每個點都位移固定d像素時，該張影像的RMS就應該等於d。

    除以N而非sqrt(N)的話會變成d/sqrt(35)，相差5.9倍。
    """
    objp = SPEC.object_points()
    rvec = np.array([0.05, -0.03, 0.02])
    tvec = np.array([-75.0, -50.0, 700.0])
    projected, _ = cv2.projectPoints(objp, rvec, tvec, K, np.zeros(5))

    offset_px = 2.0
    observed = projected + np.array([offset_px, 0.0], dtype=np.float32)

    errors = _per_view_reprojection_errors(
        [objp], [observed.astype(np.float32)], (rvec,), (tvec,), K, np.zeros(5)
    )

    assert errors[0] == pytest.approx(offset_px, abs=1e-4)


def test_rms_matches_opencv_calibrate_camera(tmp_path):
    """整條流程算出來的RMS要與cv2.calibrateCamera自己回傳的RMS一致。"""
    frontal = make_frontal_chessboard_image(SPEC, margin_squares=2)
    pattern_w = (SPEC.cols - 1) * SPEC.square_size_mm
    pattern_h = (SPEC.rows - 1) * SPEC.square_size_mm

    rng = np.random.default_rng(42)
    image_dir = tmp_path / "front"
    image_dir.mkdir()
    for i in range(18):
        rvec = rng.uniform(-0.2, 0.2, size=3)
        z = rng.uniform(600.0, 850.0)
        tvec = np.array([-pattern_w / 2 + rng.uniform(-30, 30), -pattern_h / 2 + rng.uniform(-20, 20), z])
        view = synthesize_view(frontal, SPEC, 2, K, rvec, tvec, (640, 480))
        cv2.imwrite(str(image_dir / f"frame_{i:04d}.png"), view)

    result = calibrate_mono(image_dir, SPEC, target_error_px=1.0)

    # 用同一批角點獨立執行一次calibrateCamera，以它回傳的RMS作為基準
    objp = SPEC.object_points()
    obj_points, img_points = [], []
    for path in sorted(image_dir.glob("*.png")):
        gray = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
        corners = find_corners(gray, SPEC)
        if corners is not None:
            obj_points.append(objp)
            img_points.append(corners)
    opencv_rms, *_ = cv2.calibrateCamera(obj_points, img_points, (640, 480), None, None)

    assert result.rms_reprojection_error == pytest.approx(opencv_rms, rel=1e-3)


BOARD_SPEC = CharucoBoardSpec(squares_x=10, squares_y=8, square_size_mm=25.0, marker_size_mm=18.0)


def test_rms_matches_opencv_when_corner_counts_differ(tmp_path):
    """ChArUco每張角點數不同時，總RMS仍要與cv2.calibrateCamera一致。

    總RMS的定義是sqrt(所有角點誤差平方和/角點總數)，所以各張的per-view RMS
    必須依該張角點數加權。直接取平均只有在每張角點數相同時才正確——棋盤格成立，
    ChArUco不成立，而ChArUco正是這個專案實際使用的路徑。
    直接平均在這組資料會低估約4%，角點少的視角被放大了權重。
    """
    frontal = make_charuco_frontal_image(BOARD_SPEC)
    H = frontal_to_object_homography(frontal, BOARD_SPEC)
    board = BOARD_SPEC.build_board()
    board_obj = board.getChessboardCorners()

    pattern_w = (BOARD_SPEC.squares_x - 1) * BOARD_SPEC.square_size_mm
    pattern_h = (BOARD_SPEC.squares_y - 1) * BOARD_SPEC.square_size_mm

    rng = np.random.default_rng(7)
    image_dir = tmp_path / "front"
    image_dir.mkdir()
    for i in range(20):
        rvec = rng.uniform(-0.1, 0.1, size=3)
        z = rng.uniform(200.0, 240.0)  # 近距離，每張看到的角點數不一樣
        tvec = np.array([-pattern_w / 2 + rng.uniform(-15, 15), -pattern_h / 2 + rng.uniform(-10, 10), z])
        cv2.imwrite(str(image_dir / f"frame_{i:04d}.png"),
                    synthesize_charuco_view(frontal, H, K, rvec, tvec, (640, 480)))

    result = calibrate_mono_charuco(image_dir, BOARD_SPEC, target_error_px=99.0)

    obj_points, img_points = [], []
    for path in sorted(image_dir.glob("*.png")):
        gray = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
        detected = detect_charuco(gray, board, min_corners=6)
        if detected is None:
            continue
        corners, ids = detected
        obj_points.append(board_obj[ids.flatten()].astype(np.float32))
        img_points.append(corners.reshape(-1, 1, 2).astype(np.float32))
    opencv_rms, *_ = cv2.calibrateCamera(obj_points, img_points, (640, 480), None, None)

    counts = [len(o) for o in obj_points]
    assert len(set(counts)) > 1, "這個測試的前提是各張角點數不同"
    # 容差1e-4：基準是獨立再跑一次calibrateCamera，尾數會差約1e-5。
    # 未加權的算法在這組資料會差4.4e-2，這個門檻擋得住。
    assert result.rms_reprojection_error == pytest.approx(opencv_rms, rel=1e-4)


def test_per_view_counts_recorded_and_saved(tmp_path):
    """角點數要一起存進.npz，否則重新載入後算出來的RMS會退回等權重。"""
    frontal = make_charuco_frontal_image(BOARD_SPEC)
    H = frontal_to_object_homography(frontal, BOARD_SPEC)
    pattern_w = (BOARD_SPEC.squares_x - 1) * BOARD_SPEC.square_size_mm
    pattern_h = (BOARD_SPEC.squares_y - 1) * BOARD_SPEC.square_size_mm

    rng = np.random.default_rng(11)
    image_dir = tmp_path / "front"
    image_dir.mkdir()
    for i in range(14):
        rvec = rng.uniform(-0.1, 0.1, size=3)
        tvec = np.array([-pattern_w / 2, -pattern_h / 2, rng.uniform(200.0, 240.0)])
        cv2.imwrite(str(image_dir / f"frame_{i:04d}.png"),
                    synthesize_charuco_view(frontal, H, K, rvec, tvec, (640, 480)))

    result = calibrate_mono_charuco(image_dir, BOARD_SPEC, target_error_px=99.0)
    assert len(result.per_view_point_counts) == len(result.per_view_errors)

    out = tmp_path / "mono.npz"
    result.save(out)
    from calibration.mono_calibration import MonoCalibrationResult

    reloaded = MonoCalibrationResult.load(out)
    assert reloaded.per_view_point_counts == result.per_view_point_counts
    assert reloaded.rms_reprojection_error == pytest.approx(result.rms_reprojection_error)
