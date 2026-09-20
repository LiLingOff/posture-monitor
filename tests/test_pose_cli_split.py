"""合併畫面的切割方式。

切錯方向或左右對調，效能測試看不出來（兩半都只是影格），
但接到執行期流程時會讓左右眼互換，三角測量的深度符號整個反過來。
拍攝端與執行端因此共用同一個函式，不各自維護一份。
"""
import numpy as np

from calibration.capture import split_merged_frame


def _side_by_side() -> np.ndarray:
    """左半全0、右半全255，方便辨認切出來的是哪一半。"""
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, 32:] = 255
    return frame


def _stacked() -> np.ndarray:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[24:] = 255
    return frame


def test_horizontal_split_is_the_default():
    left, right = split_merged_frame(_side_by_side())
    assert left.shape == right.shape == (48, 32, 3)
    assert left.max() == 0 and right.min() == 255


def test_swap_lr_exchanges_the_two_halves():
    left, right = split_merged_frame(_side_by_side(), swap_lr=True)
    assert left.min() == 255 and right.max() == 0


def test_vertical_split_cuts_top_and_bottom():
    top, bottom = split_merged_frame(_stacked(), vertical_split=True)
    assert top.shape == bottom.shape == (24, 64, 3)
    assert top.max() == 0 and bottom.min() == 255


def test_vertical_split_honours_swap_lr():
    top, bottom = split_merged_frame(_stacked(), vertical_split=True, swap_lr=True)
    assert top.min() == 255 and bottom.max() == 0


def test_pose_cli_has_no_duplicate_split_implementation():
    """pose.cli若自己複製一份切法，兩邊會慢慢分岔。

    這裡讀原始碼而不是import：pose/cli.py用的是相對匯入（`from ..calibration`），
    只在`python -m src.pose.cli`的情境下成立，測試把src直接放進sys.path時匯入不了。
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "src" / "pose" / "cli.py").read_text(
        encoding="utf-8"
    )
    assert "split_merged_frame" in source
    assert "_split_stereo" not in source
