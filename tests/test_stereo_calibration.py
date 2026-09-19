from pathlib import Path

import cv2
import numpy as np

from calibration.chessboard import ChessboardSpec
from calibration.stereo_calibration import calibrate_stereo
from synthetic import make_frontal_chessboard_image, synthesize_view

DEST_SIZE = (640, 480)
TRUE_K = np.array(
    [
        [700.0, 0.0, 320.0],
        [0.0, 700.0, 240.0],
        [0.0, 0.0, 1.0],
    ]
)
TRUE_BASELINE_MM = 120.0  # 右相機相對左相機沿 X 軸的平移量


def test_stereo_calibration_recovers_known_baseline(tmp_path: Path):
    spec = ChessboardSpec(cols=7, rows=5, square_size_mm=25.0)
    margin_squares = 2
    frontal = make_frontal_chessboard_image(spec, margin_squares=margin_squares)

    pattern_w_mm = (spec.cols - 1) * spec.square_size_mm
    pattern_h_mm = (spec.rows - 1) * spec.square_size_mm

    rng = np.random.default_rng(seed=7)
    n_pairs = 18

    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()

    for i in range(n_pairs):
        rvec = rng.uniform(-0.15, 0.15, size=3)
        z = rng.uniform(650.0, 850.0)
        jitter_x = rng.uniform(-20.0, 20.0)
        jitter_y = rng.uniform(-15.0, 15.0)
        tvec_left = np.array(
            [-pattern_w_mm / 2 + jitter_x, -pattern_h_mm / 2 + jitter_y, z]
        )
        tvec_right = tvec_left + np.array([TRUE_BASELINE_MM, 0.0, 0.0])

        view_l = synthesize_view(
            frontal, spec, margin_squares, TRUE_K, rvec, tvec_left, DEST_SIZE
        )
        view_r = synthesize_view(
            frontal, spec, margin_squares, TRUE_K, rvec, tvec_right, DEST_SIZE
        )

        name = f"frame_{i:04d}.png"
        cv2.imwrite(str(left_dir / name), view_l)
        cv2.imwrite(str(right_dir / name), view_r)

    result = calibrate_stereo(left_dir, right_dir, spec, target_error_px=1.0)

    assert result.rms_error < 1.0
    assert abs(result.baseline_mm - TRUE_BASELINE_MM) / TRUE_BASELINE_MM < 0.1

    identity_deviation = np.linalg.norm(result.R - np.eye(3))
    assert identity_deviation < 0.05

    assert result.Q.shape == (4, 4)
    assert result.P1.shape == (3, 4)
    assert result.P2.shape == (3, 4)


def test_stereo_calibration_raises_when_filenames_mismatch(tmp_path: Path):
    spec = ChessboardSpec(cols=7, rows=5, square_size_mm=25.0)
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()

    blank = np.full((480, 640), 255, dtype=np.uint8)
    cv2.imwrite(str(left_dir / "a.png"), blank)
    cv2.imwrite(str(right_dir / "b.png"), blank)

    try:
        calibrate_stereo(left_dir, right_dir, spec)
        assert False, "應拋出 ValueError"
    except ValueError as e:
        assert str(left_dir) in str(e) and str(right_dir) in str(e)
