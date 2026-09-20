"""擷取流程的邊界情況測試，不需要實際連接相機。"""
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
    """任何開啟相機的嘗試都直接讓測試失敗——用來證明提前返回時完全沒有存取硬體。"""

    def _boom(index, width=None, height=None):
        raise AssertionError(f"不應開啟相機（index={index}）")

    monkeypatch.setattr(capture, "_open_camera", _boom)


def test_capture_mono_returns_early_when_already_enough(tmp_path, forbid_camera, capsys):
    """張數已達標時要正常返回。

    沒有這個保護的話拍攝迴圈一次都不會執行，顯示最後一幀的變數從未被指派，
    會以UnboundLocalError結束——重新執行同一條指令就會遇到。
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


def test_warns_when_frame_is_not_side_by_side(capsys):
    """單眼畫面（4:3）被當成左右並排切開時要發出警告。

    這是實際遇到的狀況：Jetson上沒有指定解析度，驅動提供了單眼的640x480，
    切成兩半變成兩塊不重疊的裁切區域，看起來像兩個不同場景，標定永遠取得不到足夠的共同角點。
    """
    single_view = np.zeros((480, 640, 3), dtype=np.uint8)
    capture._warn_if_not_side_by_side(single_view, vertical_split=False)
    assert "很可能是單眼視角" in capsys.readouterr().out


def test_no_warning_for_genuine_side_by_side_frame(capsys):
    side_by_side = np.zeros((720, 2560, 3), dtype=np.uint8)  # 兩眼各1280x720
    capture._warn_if_not_side_by_side(side_by_side, vertical_split=False)
    assert capsys.readouterr().out == ""


def test_vertical_split_checks_the_other_axis(capsys):
    stacked = np.zeros((1440, 1280, 3), dtype=np.uint8)  # 上下堆疊，各1280x720
    capture._warn_if_not_side_by_side(stacked, vertical_split=True)
    assert capsys.readouterr().out == ""

    wide = np.zeros((720, 2560, 3), dtype=np.uint8)  # 左右並排的畫面拿來當成上下切開就該發出警告
    capture._warn_if_not_side_by_side(wide, vertical_split=True)
    assert "很可能是單眼視角" in capsys.readouterr().out


def test_capture_still_opens_camera_when_images_missing(tmp_path, monkeypatch):
    """張數不足時應照常開啟相機（確認提前返回沒有寫成永遠跳過）。"""
    out_dir = tmp_path / "front"
    _fill_with_images(out_dir, 3)

    opened = []

    def _fake_open(index, width=None, height=None):
        opened.append(index)
        raise RuntimeError("stop here")  # 確認有開啟相機即可，不必真的進入迴圈

    monkeypatch.setattr(capture, "_open_camera", _fake_open)

    with pytest.raises(RuntimeError):
        capture.capture_mono(0, out_dir, SPEC, target_count=40)
    assert opened == [0]


class _AlwaysOnCamera:
    """永遠讀得到畫面的假相機，畫面內容固定為fill值當指紋。"""

    def __init__(self, fill=99, shape=(48, 64, 3)):
        self._frame = np.full(shape, fill, dtype=np.uint8)
        self.grabbed = 0
        self.retrieved = 0

    def isOpened(self):
        return True

    def read(self):
        return True, self._frame.copy()

    def grab(self):
        self.grabbed += 1
        return True

    def retrieve(self):
        self.retrieved += 1
        return True, self._frame.copy()

    def release(self):
        pass


@pytest.fixture
def autopress_space(monkeypatch):
    """讓拍攝迴圈以為使用者一直在按空白鍵，且角點永遠偵測得到。"""
    monkeypatch.setattr(capture, "find_corners", lambda *a, **k: np.zeros((54, 1, 2), np.float32))
    monkeypatch.setattr(capture.cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(capture.cv2, "drawChessboardCorners", lambda *a, **k: None)
    monkeypatch.setattr(capture.cv2, "destroyAllWindows", lambda *a, **k: None)
    monkeypatch.setattr(capture.cv2, "waitKey", lambda *a, **k: 32)


def test_capture_does_not_overwrite_when_numbering_has_gaps(tmp_path, monkeypatch, autopress_space):
    """編號不連續時補拍不該蓋掉既有影像。

    拍完發現某張沒對焦、刪掉再重跑是很自然的操作。用檔案數量當索引的話
    新檔會撞到既有檔名，無聲蓋掉一張，而且印出的張數會比實際檔案數多。
    """
    out_dir = tmp_path / "front"
    out_dir.mkdir()
    for i in range(1, 6):
        cv2.imwrite(str(out_dir / f"frame_{i:04d}.png"), np.full((48, 64, 3), i * 10, np.uint8))
    (out_dir / "frame_0003.png").unlink()

    before = {p.name: int(cv2.imread(str(p))[0, 0, 0]) for p in out_dir.glob("*.png")}
    monkeypatch.setattr(capture, "_open_camera", lambda *a, **k: _AlwaysOnCamera(fill=99))

    capture.capture_mono(0, out_dir, SPEC, target_count=6)

    after = {p.name: int(cv2.imread(str(p))[0, 0, 0]) for p in out_dir.glob("*.png")}
    for name, fingerprint in before.items():
        assert after.get(name) == fingerprint, f"{name} 被覆蓋了"
    assert len(after) == 6, "印出的張數要與實際檔案數一致"


def test_next_frame_index_uses_max_not_count(tmp_path):
    tmp_path.joinpath("frame_0001.png").touch()
    tmp_path.joinpath("frame_0009.png").touch()
    tmp_path.joinpath("not_a_frame.png").touch()
    assert capture._next_frame_index(tmp_path) == 10


def test_next_frame_index_spans_both_stereo_dirs(tmp_path):
    left, right = tmp_path / "l", tmp_path / "r"
    left.mkdir()
    right.mkdir()
    left.joinpath("frame_0002.png").touch()
    right.joinpath("frame_0007.png").touch()
    assert capture._next_frame_index(left, right) == 8


def test_stereo_grabs_both_cameras_before_retrieving(tmp_path, monkeypatch, autopress_space):
    """左右要先各自grab再retrieve。

    串著呼叫read()的話，兩張畫面會差到一個影格間隔加上解碼時間，板子只要有位移
    就會污染外參R/T——而重投影誤差不會變差，看不出來。
    """
    left_out, right_out = tmp_path / "l", tmp_path / "r"
    order = []

    class _TracingCamera(_AlwaysOnCamera):
        def __init__(self, tag):
            super().__init__()
            self._tag = tag

        def grab(self):
            order.append(f"grab_{self._tag}")
            return super().grab()

        def retrieve(self):
            order.append(f"retrieve_{self._tag}")
            return super().retrieve()

        def read(self):
            order.append(f"read_{self._tag}")
            return super().read()

    cams = {0: _TracingCamera("l"), 1: _TracingCamera("r")}
    monkeypatch.setattr(capture, "_open_camera", lambda index, *a, **k: cams[index])

    capture.capture_stereo(0, 1, left_out, right_out, SPEC, target_count=1)

    assert "read_l" not in order and "read_r" not in order, "不該用read()，兩次曝光會被解碼時間拉開"
    first_cycle = order[:4]
    assert first_cycle == ["grab_l", "grab_r", "retrieve_l", "retrieve_r"], first_cycle
