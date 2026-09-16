"""擷取流程的邊界情況測試，不需要真的接相機。"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from calibration import capture
from calibration.charuco import CharucoBoardSpec
from calibration.chessboard import ChessboardSpec

SPEC = ChessboardSpec(cols=9, rows=6, square_size_mm=25.0)
BOARD_SPEC = CharucoBoardSpec(squares_x=10, squares_y=8, square_size_mm=25.0, marker_size_mm=18.0)


def _fill_with_images(directory: Path, count: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    blank = np.zeros((48, 64, 3), dtype=np.uint8)
    for i in range(count):
        cv2.imwrite(str(directory / f"frame_{i:04d}.png"), blank)


@pytest.fixture
def forbid_camera(monkeypatch):
    """任何開相機的嘗試都直接讓測試失敗——用來證明提前返回時根本沒碰硬體。"""

    def _boom(index):
        raise AssertionError(f"不該開相機（index={index}）")

    monkeypatch.setattr(capture, "_open_camera", _boom)


def test_capture_mono_returns_early_when_already_enough(tmp_path, forbid_camera, capsys):
    """張數已達標時要乾淨返回。

    沒有這個保護的話拍攝迴圈一次都不跑，顯示最後一幀的變數沒被指派，
    會以UnboundLocalError結束——重跑同一條指令就會踩到。
    """
    out_dir = tmp_path / "front"
    _fill_with_images(out_dir, 40)

    capture.capture_mono(0, out_dir, SPEC, target_count=40)

    assert "跳過拍攝" in capsys.readouterr().out


def test_capture_mono_charuco_returns_early_when_already_enough(tmp_path, forbid_camera):
    out_dir = tmp_path / "front"
    _fill_with_images(out_dir, 20)
    capture.capture_mono_charuco(0, out_dir, BOARD_SPEC, target_count=20)


def test_capture_stereo_returns_early_when_already_enough(tmp_path, forbid_camera):
    left, right = tmp_path / "left", tmp_path / "right"
    _fill_with_images(left, 20)
    _fill_with_images(right, 20)
    capture.capture_stereo(1, 2, left, right, SPEC, target_count=20)


def test_capture_stereo_single_device_returns_early_when_already_enough(tmp_path, forbid_camera):
    left, right = tmp_path / "left", tmp_path / "right"
    _fill_with_images(left, 20)
    _fill_with_images(right, 20)
    capture.capture_stereo_single_device(1, left, right, SPEC, target_count=20)


def test_capture_stereo_charuco_returns_early_when_already_enough(tmp_path, forbid_camera):
    left, right = tmp_path / "left", tmp_path / "right"
    _fill_with_images(left, 20)
    _fill_with_images(right, 20)
    capture.capture_stereo_charuco(1, 2, left, right, BOARD_SPEC, target_count=20)


def test_capture_stereo_charuco_single_device_returns_early_when_already_enough(tmp_path, forbid_camera):
    left, right = tmp_path / "left", tmp_path / "right"
    _fill_with_images(left, 20)
    _fill_with_images(right, 20)
    capture.capture_stereo_charuco_single_device(1, left, right, BOARD_SPEC, target_count=20)


def test_capture_still_opens_camera_when_images_missing(tmp_path, monkeypatch):
    """張數不足時該照常開相機（確認提前返回沒有寫成永遠跳過）。"""
    out_dir = tmp_path / "front"
    _fill_with_images(out_dir, 3)

    opened = []

    def _fake_open(index):
        opened.append(index)
        raise RuntimeError("stop here")  # 開了相機就夠了，不必真的進迴圈

    monkeypatch.setattr(capture, "_open_camera", _fake_open)

    with pytest.raises(RuntimeError):
        capture.capture_mono(0, out_dir, SPEC, target_count=40)
    assert opened == [0]
