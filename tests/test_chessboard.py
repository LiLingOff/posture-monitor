import numpy as np

from calibration.chessboard import ChessboardSpec, find_corners
from synthetic import make_frontal_chessboard_image


def test_object_points_shape_and_scale():
    spec = ChessboardSpec(cols=7, rows=5, square_size_mm=25.0)
    objp = spec.object_points()

    assert objp.shape == (35, 3)
    assert np.allclose(objp[:, 2], 0.0)
    assert np.allclose(objp[0], [0, 0, 0])
    assert np.allclose(objp[1], [25.0, 0, 0])
    assert np.allclose(objp[spec.cols], [0, 25.0, 0])


def test_find_corners_on_frontal_chessboard():
    spec = ChessboardSpec(cols=7, rows=5, square_size_mm=25.0)
    img = make_frontal_chessboard_image(spec, margin_squares=2)

    corners = find_corners(img, spec)

    assert corners is not None
    assert corners.shape == (spec.cols * spec.rows, 1, 2)


def test_find_corners_returns_none_on_blank_image():
    spec = ChessboardSpec(cols=7, rows=5, square_size_mm=25.0)
    blank = np.full((480, 640), 255, dtype=np.uint8)

    assert find_corners(blank, spec) is None
