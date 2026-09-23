import cv2
import numpy as np
import pytest

from calibration.chessboard import (ChessboardSpec, ensure_uniform_size,
                                    find_corners, load_gray_images)
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


def test_load_gray_images_accepts_uppercase_and_jpeg(tmp_path):
    """手機匯出常是.JPG/.jpeg，被略過的話只會看到「影像過少」，很難聯想原因。"""
    img = np.full((48, 64, 3), 128, dtype=np.uint8)
    for name in ("a.png", "b.PNG", "c.jpg", "d.JPG", "e.jpeg", "f.bmp"):
        cv2.imwrite(str(tmp_path / name), img)
    cv2.imwrite(str(tmp_path / "ignored.tiff"), img)

    loaded = load_gray_images(tmp_path)
    assert sorted(p.name for p, _ in loaded) == ["a.png", "b.PNG", "c.jpg", "d.JPG", "e.jpeg", "f.bmp"]


def test_load_gray_images_missing_dir_returns_empty(tmp_path):
    assert load_gray_images(tmp_path / "not_there") == []


def test_ensure_uniform_size_rejects_mixed_resolutions(tmp_path):
    """混用解析度時內參沒有意義，但OpenCV只會照算不會抱怨。"""
    images = [
        (tmp_path / "a.png", np.zeros((480, 640), dtype=np.uint8)),
        (tmp_path / "b.png", np.zeros((720, 1280), dtype=np.uint8)),
    ]
    with pytest.raises(ValueError) as e:
        ensure_uniform_size(images)
    assert "解析度不一致" in str(e.value)


def test_ensure_uniform_size_returns_width_height(tmp_path):
    images = [(tmp_path / f"{i}.png", np.zeros((480, 640), dtype=np.uint8)) for i in range(3)]
    assert ensure_uniform_size(images) == (640, 480)
