"""量測當下與段落結束時印出來的文字。

這些句子先前全部混在 posture.py 的迴圈裡，一條測試都沒有，而它們出過的錯
不比演算法少：訊息寫死一個會過期的相機編號、同一次錄製印出兩個不同的略過率、
註解寫 15° 而程式判 10°。文字是使用者唯一看得到的東西。
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from geometry.live_report import (angle_text, conditions_text, elapsed_text,
                                  live_line, monitor_line, study_summary_lines,
                                  summary_lines, transition_lines)
from geometry.recording import Recording, Session
from geometry.smoothing import RollingAngle
from geometry.study import SegmentReview


def _window(values, size=30):
    window = RollingAngle(size)
    for value in values:
        window.add(value)
    return window


def _measurement(distance=620.0, azimuth=43.0):
    return SimpleNamespace(reference_depth_mm=distance, camera_azimuth_deg=azimuth)


def _recording(ca, sym=(), rejected=0, frames=None, window=None):
    session = Session()
    for a, s in zip(ca, sym or ca):
        session.add((a, s))
    return Recording(
        session=session, ca_window=window or _window(ca),
        sym_window=_window(sym or ca), rejected=rejected,
        frames=len(ca) + rejected if frames is None else frames,
    )


def _segment(condition="upright", trial=1, **overrides):
    values = dict(condition=condition, trial=trial, theta_ca_deg=-1.07,
                  usable=96, frames=100, rejection_rate=0.04)
    values.update(overrides)
    return SegmentReview(**values)


# ---- 單位 --------------------------------------------------------------

@pytest.mark.parametrize("seconds, text", [
    (0.0, " 0:00"), (9.9, " 0:09"), (61.0, " 1:01"), (3599.0, "59:59"),
])
def test_elapsed_is_minutes_and_seconds(seconds, text):
    assert elapsed_text(seconds) == text


def test_an_angle_without_data_is_a_dash_not_a_zero():
    """0 是合法的角度。寫成 0 的話「量不到」與「剛好是零」長得一樣。"""
    assert "—" in angle_text(RollingAngle(30), None)
    assert "0" not in angle_text(RollingAngle(30), None)


def test_the_average_comes_before_the_single_frame_value():
    """要看的是平均。單幀誤差與判定門檻同量級。"""
    text = angle_text(_window([10.0, 12.0, 14.0]), 20.0)
    assert text.index("+12.0") < text.index("單幀")
    assert "+20.0" in text


def test_a_single_frame_that_is_missing_does_not_crash_the_average():
    assert "單幀" in angle_text(_window([10.0, 12.0]), None)


def test_the_azimuth_can_be_missing_while_the_distance_is_not():
    """一邊肩膀沒偵測到時方位角是 None，而深度仍然算得出來。直接格式化會當掉。"""
    assert "620mm" in conditions_text(_measurement(azimuth=None))
    assert "——°" in conditions_text(_measurement(azimuth=None))
    assert "——mm" in conditions_text(_measurement(distance=None))


# ---- 逐幀那一行 --------------------------------------------------------

def test_a_skipped_frame_keeps_the_numbers_it_still_has():
    """調整架設位置時正是略過最多的時候，那幾個數字不能跟著消失。"""
    line = live_line(RollingAngle(30), RollingAngle(30), _measurement(),
                     (None, None), 12, "左右只有 3 個共同關鍵點")
    assert "共同關鍵點" in line
    assert "620mm" in line and "43°" in line
    assert "已略過 12" in line


def test_the_live_line_shows_how_long_is_left():
    """受試者維持姿勢時最想知道的就是這個，不知道還要多久就容易提早鬆掉。"""
    line = live_line(_window([3.0]), _window([1.0]), _measurement(),
                     (3.0, 1.0), 0, None, remaining_s=45.0)
    assert "剩  45s" in line


def test_a_run_without_a_time_limit_has_no_countdown():
    line = live_line(_window([3.0]), _window([1.0]), _measurement(),
                     (3.0, 1.0), 0, None)
    assert "剩" not in line


def test_the_window_progress_is_on_the_live_line():
    line = live_line(_window([3.0, 4.0]), _window([1.0, 1.0]), _measurement(),
                     (4.0, 1.0), 0, None)
    assert " 2/30幀" in line


def test_the_verdict_joins_the_line_once_there_is_one():
    verdict = SimpleNamespace(posture=SimpleNamespace(value="超標"))
    line = live_line(_window([19.0]), _window([1.0]), _measurement(),
                     (19.0, 1.0), 0, None, verdict=verdict)
    assert "超標" in line


def test_the_monitor_line_says_what_phase_it_is_in():
    """看不到視窗的時候，終端機仍然要說得出現在在做什麼。"""
    state = SimpleNamespace(ca_window=_window([3.0]), sym_window=_window([1.0]))
    result = SimpleNamespace(mode=SimpleNamespace(value="取基準"),
                             phase_remaining_s=12.0, reason=None,
                             corrected=(3.0, 1.0), verdict=None)
    line = monitor_line(state, result, 0)
    assert line.startswith("取基準 剩")
    assert "θ_CA" in line


def test_the_monitor_line_keeps_the_phase_while_skipping():
    state = SimpleNamespace(ca_window=RollingAngle(30), sym_window=RollingAngle(30))
    result = SimpleNamespace(mode=SimpleNamespace(value="倒數"),
                             phase_remaining_s=3.0, reason="沒有偵測到人",
                             corrected=(None, None), verdict=None)
    line = monitor_line(state, result, 7)
    assert "倒數" in line and "沒有偵測到人" in line and "7" in line


# ---- 段落結束 ----------------------------------------------------------

def test_the_summary_says_whether_the_baseline_was_subtracted():
    """沒扣基準的數字跟扣過的差了一整個基準，標題不講就分不出來。"""
    recording = _recording([3.0, 4.0, 5.0])
    assert "未扣除個人基準" in summary_lines(recording)[0]
    assert "相對個人基準" in summary_lines(recording, baseline=object())[0]


def test_the_summary_reports_the_whole_segment_and_the_window_separately():
    """視窗只有 30 幀，拿它當結尾的摘要等於把一百秒講成最後六秒的樣子。"""
    lines = summary_lines(_recording([0.0] * 50 + [20.0] * 10))
    assert any("60 幀" in line for line in lines)
    assert any("結束前" in line for line in lines)


def test_one_frame_is_not_enough_to_report_a_spread():
    assert any("沒有足夠的量測" in line for line in summary_lines(_recording([5.0])))


def test_the_rejection_rate_is_only_printed_when_there_was_one():
    assert not any("略過" in line for line in summary_lines(_recording([1.0, 2.0])))
    lines = summary_lines(_recording([1.0, 2.0], rejected=8, frames=10))
    assert any("略過 8 / 10 幀（80%）" in line for line in lines)


def test_no_transitions_is_said_out_loud():
    """空著不印的話，看起來像這一段的輸出被截斷了。"""
    assert transition_lines([]) == ["整段沒有狀態變化"]


def test_transitions_are_reprinted_at_the_end():
    """逐幀那一行會被下一幀蓋掉，狀態變化夾在裡面很容易錯過。"""
    lines = transition_lines([" 0:06  超標：…", " 0:15  正常：…"])
    assert "2 次" in lines[0]
    assert len(lines) == 3


# ---- study 的總結 ------------------------------------------------------

def test_a_clean_run_is_not_nagged_about():
    lines = study_summary_lines([_segment()], "chenyue", [Path("upright-1.csv")])
    assert not any("建議重量" in line for line in lines)
    assert any("upright-1.csv" in line for line in lines)


def test_the_two_reasons_to_redo_are_explained_separately():
    """略過率高是偵測的問題，轉身是受試者的問題，該做的事不一樣。"""
    lines = study_summary_lines(
        [_segment(rejection_rate=0.31), _segment("head-forward", 1, turned_deg=28.0)],
        "chenyue", [None, None])
    assert any("代表性有限" in line for line in lines)
    assert any("盯著牆上那個點" in line for line in lines)
    assert any("upright #1、head-forward #1" in line for line in lines)


def test_a_retaken_segment_is_neither_redone_nor_analysed():
    """掉線那段往往只留下幾秒，跟後面完整的那份平均在一起只會把結果拉偏。"""
    lost = _segment("head-forward", 1, camera_lost=True, retaken=True,
                    rejection_rate=0.96)
    retake = _segment("head-forward", 2)
    lines = study_summary_lines([lost, retake], "chenyue",
                                [Path("head-forward-1.csv"),
                                 Path("head-forward-2.csv")])
    assert any("後面重量了" in line for line in lines)
    assert not any("建議重量" in line for line in lines)
    analyse = next(line for line in lines if "posture.py analyse" in line)
    assert "head-forward-1.csv" not in analyse
    assert "head-forward-2.csv" in analyse


def test_a_lost_camera_with_no_retake_is_still_flagged():
    lines = study_summary_lines([_segment(camera_lost=True, rejection_rate=0.96)],
                                "chenyue", [None])
    assert any("相機掉線" in line and "後面重量了" not in line for line in lines)
    assert any("建議重量" in line for line in lines)


def test_a_segment_with_no_angle_at_all_still_gets_a_row():
    """整段都被略過時 θ_CA 是 None。那一列消失的話，那一段看起來像沒量過。"""
    lines = study_summary_lines([_segment(theta_ca_deg=None, usable=0)],
                                "chenyue", [None])
    assert any("upright" in line and "—" in line for line in lines)
