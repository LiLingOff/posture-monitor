from pathlib import Path

import numpy as np

from calibration.charuco import CharucoBoardSpec, detect_charuco, save_board_image
from charuco_synthetic import make_charuco_frontal_image

SPEC = CharucoBoardSpec(squares_x=6, squares_y=4, square_size_mm=25.0, marker_size_mm=18.0)


def test_build_board_corner_count():
    board = SPEC.build_board()
    # NxM squares -> (N-1)x(M-1)內角點
    assert board.getChessboardCorners().shape == ((SPEC.squares_x - 1) * (SPEC.squares_y - 1), 3)


def test_invalid_dictionary_raises():
    bad = CharucoBoardSpec(squares_x=6, squares_y=4, square_size_mm=25.0, marker_size_mm=18.0, dictionary_name="NOPE")
    try:
        bad.build_board()
        assert False, "應拋出 ValueError"
    except ValueError:
        pass


def test_save_board_image(tmp_path: Path):
    out = tmp_path / "board.png"
    save_board_image(SPEC, out, pixels_per_square=50)
    assert out.exists()


def test_detect_charuco_on_frontal_image_finds_all_corners():
    board = SPEC.build_board()
    frontal = make_charuco_frontal_image(SPEC)
    detected = detect_charuco(frontal, board, min_corners=1)
    assert detected is not None
    corners, ids = detected
    assert len(ids) == board.getChessboardCorners().shape[0]


def test_detect_charuco_returns_none_on_blank():
    board = SPEC.build_board()
    blank = np.full((480, 640), 255, dtype=np.uint8)
    assert detect_charuco(blank, board, min_corners=1) is None


def test_detect_charuco_respects_min_corners_threshold():
    board = SPEC.build_board()
    frontal = make_charuco_frontal_image(SPEC)
    total = board.getChessboardCorners().shape[0]
    assert detect_charuco(frontal, board, min_corners=total + 1) is None


def test_legacy_pattern_board_still_builds_and_detects():
    legacy_spec = CharucoBoardSpec(
        squares_x=6, squares_y=4, square_size_mm=25.0, marker_size_mm=18.0, legacy_pattern=True
    )
    board = legacy_spec.build_board()
    frontal = make_charuco_frontal_image(legacy_spec)
    detected = detect_charuco(frontal, board, min_corners=1)
    assert detected is not None
