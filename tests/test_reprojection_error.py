"""重投影誤差計算的回歸測試。

這個數字是標定品質的唯一判準（單眼<0.3px、雙目<0.5px），算錯不會有任何外顯症狀，
只會讓爛標定悄悄通過門檻，所以用「已知位移量」跟「OpenCV自己算的RMS」兩種方式鎖住。
"""
import cv2
import numpy as np
import pytest

from calibration.chessboard import ChessboardSpec, find_corners
from calibration.mono_calibration import _per_view_reprojection_errors, calibrate_mono
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
    """每個點都位移固定d像素時，該張影像的RMS就該等於d。

    除以N而不是sqrt(N)的話會變成d/sqrt(35)，差5.9倍。
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
    """整條管線算出來的RMS要跟cv2.calibrateCamera自己回傳的RMS一致。"""
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

    # 用同一批角點獨立跑一次calibrateCamera，拿它回傳的RMS當基準
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
