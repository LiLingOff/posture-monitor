from pathlib import Path

import cv2
import numpy as np

from calibration.chessboard import ChessboardSpec
from calibration.mono_calibration import calibrate_mono
from synthetic import make_frontal_chessboard_image, synthesize_view

DEST_SIZE = (640, 480)  # (width, height)
TRUE_K = np.array(
    [
        [700.0, 0.0, 320.0],
        [0.0, 700.0, 240.0],
        [0.0, 0.0, 1.0],
    ]
)


def _generate_poses(
    n: int, rng: np.random.Generator, pattern_w_mm: float, pattern_h_mm: float
):
    poses = []
    for _ in range(n):
        rvec = rng.uniform(-0.2, 0.2, size=3)
        z = rng.uniform(600.0, 850.0)
        jitter_x = rng.uniform(-30.0, 30.0)
        jitter_y = rng.uniform(-20.0, 20.0)
        tvec = np.array([-pattern_w_mm / 2 + jitter_x, -pattern_h_mm / 2 + jitter_y, z])
        poses.append((rvec, tvec))
    return poses


def test_mono_calibration_recovers_known_intrinsics(tmp_path: Path):
    spec = ChessboardSpec(cols=7, rows=5, square_size_mm=25.0)
    margin_squares = 2
    frontal = make_frontal_chessboard_image(spec, margin_squares=margin_squares)

    pattern_w_mm = (spec.cols - 1) * spec.square_size_mm
    pattern_h_mm = (spec.rows - 1) * spec.square_size_mm

    rng = np.random.default_rng(seed=42)
    poses = _generate_poses(18, rng, pattern_w_mm, pattern_h_mm)

    image_dir = tmp_path / "front"
    image_dir.mkdir()
    for i, (rvec, tvec) in enumerate(poses):
        view = synthesize_view(
            frontal, spec, margin_squares, TRUE_K, rvec, tvec, DEST_SIZE
        )
        cv2.imwrite(str(image_dir / f"frame_{i:04d}.png"), view)

    result = calibrate_mono(image_dir, spec, target_error_px=1.0)

    assert len(result.used_images) >= 15
    assert result.rms_reprojection_error < 1.0

    fx, fy = result.camera_matrix[0, 0], result.camera_matrix[1, 1]
    cx, cy = result.camera_matrix[0, 2], result.camera_matrix[1, 2]
    assert abs(fx - TRUE_K[0, 0]) / TRUE_K[0, 0] < 0.05
    assert abs(fy - TRUE_K[1, 1]) / TRUE_K[1, 1] < 0.05
    assert abs(cx - TRUE_K[0, 2]) < 15
    assert abs(cy - TRUE_K[1, 2]) < 15


def test_mono_calibration_raises_on_too_few_images(tmp_path: Path):
    spec = ChessboardSpec(cols=7, rows=5, square_size_mm=25.0)
    image_dir = tmp_path / "front"
    image_dir.mkdir()

    try:
        calibrate_mono(image_dir, spec)
        assert False, "應該要拋出 ValueError"
    except ValueError as e:
        assert "過少" in str(e)


def test_mono_calibration_save_and_load_roundtrip(tmp_path: Path):
    spec = ChessboardSpec(cols=7, rows=5, square_size_mm=25.0)
    margin_squares = 2
    frontal = make_frontal_chessboard_image(spec, margin_squares=margin_squares)
    pattern_w_mm = (spec.cols - 1) * spec.square_size_mm
    pattern_h_mm = (spec.rows - 1) * spec.square_size_mm

    rng = np.random.default_rng(seed=1)
    poses = _generate_poses(15, rng, pattern_w_mm, pattern_h_mm)

    image_dir = tmp_path / "front"
    image_dir.mkdir()
    for i, (rvec, tvec) in enumerate(poses):
        view = synthesize_view(
            frontal, spec, margin_squares, TRUE_K, rvec, tvec, DEST_SIZE
        )
        cv2.imwrite(str(image_dir / f"frame_{i:04d}.png"), view)

    result = calibrate_mono(image_dir, spec, target_error_px=1.0)
    out_path = tmp_path / "front.npz"
    result.save(out_path)

    from calibration.mono_calibration import MonoCalibrationResult

    loaded = MonoCalibrationResult.load(out_path)
    assert np.allclose(loaded.camera_matrix, result.camera_matrix)
    assert loaded.image_size == result.image_size
