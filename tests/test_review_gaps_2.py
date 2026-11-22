"""GPT 審查測試檔（2026-11-22）後三批列的缺漏，逐項核對後仍沒有覆蓋的部分。

前一批見 test_review_gaps.py。已有測試的（跨受試者分母、before-baseline 段落沒有基準、
θ 與 ± 的退路）不重複。
"""
import cv2
import numpy as np
import pytest

from geometry.judgement import AngleJudge, Posture
from geometry.session_analysis import AngleSummary
from geometry.trips import AXIS_LIMITS, INCLUDED, NOTED, axis_status
from view import window
from view.text import TextItem, TextPainter
from view_fakes import view as _view


# ---- 判定門檻的邊界：嚴格大於才超標、嚴格小於才正常 ----------------------
# 沒給誤差時遲滯寬度退回下限（θ 是 1°，肩高是 2mm），所以邊界值都是整數。

@pytest.mark.parametrize("threshold, floor, two_sided", [
    (10.0, 1.0, True),     # θ_CA、θ_sym、θ_KA
    (10.0, 2.0, False),    # 肩部垂直位移（mm）
])
def test_the_threshold_boundary_is_strict_on_both_sides(threshold, floor, two_sided):
    def judge(mean):
        j = AngleJudge(threshold, "x", two_sided=two_sided, minimum_margin=floor)
        return j.update(mean, None).posture

    assert judge(threshold + floor) is Posture.UNKNOWN, "剛好在誤差邊緣不能判超標"
    assert judge(threshold + floor + 0.01) is Posture.OVER
    assert judge(threshold - floor) is Posture.UNKNOWN, "剛好在誤差邊緣不能判正常"
    assert judge(threshold - floor - 0.01) is Posture.OK


def test_a_two_sided_axis_treats_both_directions_the_same_at_the_boundary():
    for mean in (11.01, -11.01):
        assert AngleJudge(10.0, "x", two_sided=True).update(mean, None).posture is Posture.OVER
    for mean in (11.0, -11.0):
        assert AngleJudge(10.0, "x", two_sided=True).update(mean, None).posture is Posture.UNKNOWN


def test_a_one_sided_axis_never_flags_the_opposite_direction():
    assert AngleJudge(10.0, "x", two_sided=False).update(-40.0, None).posture is Posture.OK


def test_hysteresis_keeps_the_previous_state_inside_the_margin():
    j = AngleJudge(10.0, "x", two_sided=True)
    assert j.update(20.0, None).posture is Posture.OVER
    assert j.update(10.5, None).posture is Posture.OVER, "在誤差內維持原狀"
    assert j.update(5.0, None).posture is Posture.OK
    assert j.update(10.5, None).posture is Posture.OK


# ---- 略過率與稀疏的邊界 --------------------------------------------------

@pytest.mark.parametrize("axis", sorted(AXIS_LIMITS))
def test_the_rejection_limit_is_inclusive(axis):
    limit = AXIS_LIMITS[axis]
    assert axis_status([limit], axis, turned=False) == INCLUDED, "剛好等於上限還算納入"
    assert axis_status([limit + 0.001], axis, turned=False) == NOTED
    assert axis_status([0.0, limit + 0.001], axis, turned=False) == NOTED, "配對時看較差的那段"


def test_the_sparse_floor_is_30_frames_or_a_tenth_of_the_usable_frames():
    def summary(count, of_frames):
        return AngleSummary("θ_KA 原始", np.arange(count, dtype=float), of_frames=of_frames)

    assert not summary(30, 100).is_sparse, "剛好 30 幀不算稀疏"
    assert summary(29, 100).is_sparse
    assert not summary(100, 1000).is_sparse, "可用幀的一成也是下限"
    assert summary(99, 1000).is_sparse
    assert not summary(1, None).is_sparse, "沒說應該有幾幀就不檢查"


# ---- 三角測量的輸入邊界 --------------------------------------------------

def _calib():
    from geometry_synthetic import make_synthetic_stereo_calibration

    K = np.array([[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]])
    return make_synthetic_stereo_calibration(K, 80.0)


def test_empty_input_gives_an_empty_result_not_an_axis_error():
    from geometry.triangulation import rectified_vertical_disparity, triangulate_points

    assert triangulate_points(_calib(), [], []).shape == (0, 3)
    assert triangulate_points(_calib(), np.empty((0, 2)), np.empty((0, 2))).shape == (0, 3)
    assert rectified_vertical_disparity(_calib(), [], []).shape == (0,)


def test_left_and_right_must_have_the_same_number_of_points():
    from geometry.triangulation import triangulate_points

    with pytest.raises(ValueError, match="點數不同"):
        triangulate_points(_calib(), np.zeros((2, 2)), np.zeros((1, 2)))


def test_a_wrongly_shaped_input_is_named():
    from geometry.triangulation import triangulate_points

    with pytest.raises(ValueError, match="points_left"):
        triangulate_points(_calib(), np.zeros((2, 3)), np.zeros((2, 2)))


def test_one_bad_point_does_not_spoil_its_neighbours():
    """右眼 NaN、inf 都只讓那一列變 NaN，其他列照常。"""
    from geometry.triangulation import triangulate_points

    left = np.array([[320.0, 240.0], [330.0, 240.0], [340.0, 240.0], [350.0, 240.0]])
    right = np.array([[300.0, 240.0], [np.nan, 240.0], [np.inf, 240.0], [330.0, 240.0]])
    out = triangulate_points(_calib(), left, right)
    assert np.isfinite(out[[0, 3]]).all()
    assert np.isnan(out[[1, 2]]).all()


# ---- 視窗：失效與生命週期 ------------------------------------------------

class _Cv2:
    error = cv2.error
    WINDOW_AUTOSIZE = cv2.WINDOW_AUTOSIZE

    def __init__(self, fails=None):
        self.fails = fails
        self.calls = []
        self.delays = []

    def _call(self, name):
        self.calls.append(name)
        if self.fails == name:
            raise cv2.error(f"假的 {name} 失敗")

    def namedWindow(self, title, flags=0):
        self._call("namedWindow")

    def imshow(self, title, image):
        self._call("imshow")

    def waitKey(self, delay):
        self.calls.append("waitKey")
        self.delays.append(delay)
        return 255

    def destroyAllWindows(self):
        self._call("destroyAllWindows")


def _live(monkeypatch, fails=None):
    stub = _Cv2(fails)
    monkeypatch.setattr(window, "cv2", stub)
    w = window.LiveWindow(painter=TextPainter(font_path="沒有這個字型.ttf"))
    w.open()
    return w, stub


def test_a_window_that_could_not_be_created_never_draws_or_waits(monkeypatch):
    w, stub = _live(monkeypatch, fails="namedWindow")
    for _ in range(5):
        w.show(_view())
    assert stub.calls == ["namedWindow"]


def test_after_imshow_fails_nothing_else_touches_the_gui(monkeypatch):
    w, stub = _live(monkeypatch, fails="imshow")
    for _ in range(5):
        w.show(_view())
    w.hold(_view())
    assert stub.calls.count("imshow") == 1
    assert stub.calls.count("namedWindow") == 1
    assert "waitKey" not in stub.calls


def test_a_window_that_died_while_drawing_is_still_destroyed_on_close(monkeypatch):
    """畫不出來的那種視窗是真的存在的，不收掉會留在桌面上。"""
    w, stub = _live(monkeypatch, fails="imshow")
    w.show(_view())
    w.close()
    assert stub.calls.count("destroyAllWindows") == 1


def test_nothing_is_destroyed_between_segments(monkeypatch):
    w, stub = _live(monkeypatch)
    for _ in range(3):
        w.begin()
        w.show(_view())
    assert stub.calls.count("destroyAllWindows") == 0
    w.close()
    assert stub.calls.count("destroyAllWindows") == 1


def test_show_polls_for_a_key_and_hold_blocks_on_one(monkeypatch):
    w, stub = _live(monkeypatch)
    w.show(_view())
    w.hold(_view())
    assert stub.delays == [1, 0], "show 只輪詢一毫秒，hold 要用 0 一直等到有按鍵"


# ---- 文字寬度不能比實際畫出來的小 ----------------------------------------

def _ink_width(painter, text, size, bold):
    canvas = np.zeros((size * 3, 1500, 3), np.uint8)
    painter.paint(canvas, [TextItem(x=10, y=10, text=text, ascii_text=text,
                                    colour=(255, 255, 255), size=size, bold=bold)])
    columns = np.where(canvas.sum(axis=(0, 2)) > 0)[0]
    return int(columns.max()) - 10 + 1 if columns.size else 0


@pytest.mark.parametrize("bold", [False, True])
@pytest.mark.parametrize("text", ["OK", "theta_CA +12.3 deg", "θ_{CA} +12.3°", "±"])
def test_the_measured_width_is_never_less_than_the_ink(text, bold):
    """排版靠 width() 決定下一欄放哪；低估的話兩欄會重疊。"""
    painter = TextPainter(font_path="無")
    assert painter.width(text, text, 24, bold=bold) >= _ink_width(painter, text, 24, bold)
