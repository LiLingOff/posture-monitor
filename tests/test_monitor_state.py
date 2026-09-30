"""monitor 的狀態機。假的量測，沒有相機也沒有視窗。"""
from __future__ import annotations

import numpy as np
import pytest

from geometry.baseline import PostureBaseline
from geometry.judgement import Posture
from geometry.monitor import Mode, MonitorState
from geometry.pipeline import PostureMeasurement
from geometry.keypoints3d import PersonKeypoints3D
from pose.topology import COCO18_KEYPOINT_NAMES, NUM_KEYPOINTS, keypoint_index

_USED = ("left_shoulder", "right_shoulder", "left_ear", "right_ear", "neck",
         "nose", "left_eye", "right_eye")


def measurement(theta_ca=5.0, theta_sym=1.0, usable=True) -> PostureMeasurement:
    """一幀量測。usable=False 會踩到深度的合理範圍檢查而被略過。"""
    points = np.full((NUM_KEYPOINTS, 3), np.nan)
    for name in _USED:
        points[keypoint_index(name)] = (0.0, 0.0, 650.0)
    disparity = np.full(NUM_KEYPOINTS, np.nan)
    for name in _USED:
        disparity[keypoint_index(name)] = 0.5
    return PostureMeasurement(
        keypoints_3d=PersonKeypoints3D(points=points),
        vertical_disparity_px=disparity,
        shared_count=len(_USED),
        theta_ca_deg=theta_ca, theta_sym_deg=theta_sym,
        theta_ca_precision_deg=0.5,
        reference_depth_mm=650.0 if usable else -900.0,
        camera_azimuth_deg=20.0, theta_ca_side="right",
        edge_keypoints=[], angle_errors=[],
    )


def _baseline(theta_ca=3.0) -> PostureBaseline:
    return PostureBaseline(
        subject="chenyue", captured_at="2026-09-30T19:00:00", frames=120,
        rejected=2, duration_s=20.0, theta_ca_deg=theta_ca, theta_sym_deg=1.0,
        theta_ca_std_deg=3.0, theta_sym_std_deg=0.5,
        theta_ca_standard_error_deg=0.5, theta_sym_standard_error_deg=0.1,
        distance_mm=650.0, azimuth_deg=20.0,
    )


def _zero(state: MonitorState, t0: float = 0.0, theta_ca: float = 5.0,
          usable=True, step: float = 0.1):
    """按 c，走完倒數與取樣，回傳最後一幀的結果。"""
    state.request_baseline(t0)
    t = t0
    last = None
    while t < t0 + state_total(state) + step:
        t += step
        last = state.feed(measurement(theta_ca=theta_ca, usable=usable), t)
        if last.mode is not Mode.COUNTDOWN and last.mode is not Mode.COLLECTING:
            break
    return last


def state_total(state: MonitorState) -> float:
    return state._countdown_s + state._baseline_s


# ---- 起手式 ------------------------------------------------------------

def test_it_starts_waiting_and_judges_nothing():
    """沒有零點就不該判。門檻套在原始角度上會因人而異。"""
    state = MonitorState("chenyue", countdown_s=1.0, baseline_s=2.0)
    assert state.mode is Mode.WAITING
    result = state.feed(measurement(), 0.0)
    assert result.mode is Mode.WAITING
    assert result.verdict is None


def test_waiting_shows_the_raw_angle_not_an_offset_from_nothing():
    """那時候要看的是鏡頭有沒有對好，不是一個沒有零點的偏移量。"""
    state = MonitorState("chenyue")
    assert state.feed(measurement(theta_ca=12.5), 0.0).corrected[0] == 12.5


def test_a_preloaded_baseline_goes_straight_to_monitoring():
    state = MonitorState("chenyue", baseline=_baseline(theta_ca=3.0))
    assert state.mode is Mode.MONITORING
    assert state.feed(measurement(theta_ca=5.0), 0.0).corrected[0] == pytest.approx(2.0)


# ---- 按 c --------------------------------------------------------------

def test_c_counts_down_before_it_starts_collecting():
    state = MonitorState("chenyue", countdown_s=5.0, baseline_s=2.0)
    state.request_baseline(0.0)
    result = state.feed(measurement(), 1.0)
    assert result.mode is Mode.COUNTDOWN
    assert result.phase_remaining_s == pytest.approx(4.0)


def test_the_countdown_frames_do_not_go_into_the_baseline():
    """一按下去就開始收的話，收到的是人還在調整姿勢的那幾秒，而那正是
    散佈最大的一段，會直接灌進零點。"""
    state = MonitorState("chenyue", countdown_s=5.0, baseline_s=2.0)
    state.request_baseline(0.0)
    for t in (1.0, 2.0, 3.0, 4.0):
        # 倒數期間餵一個很偏的角度，它不該影響後面算出來的基準。
        assert state.feed(measurement(theta_ca=90.0), t).mode is Mode.COUNTDOWN
    state.feed(measurement(theta_ca=5.0), 5.1)
    for i in range(60):
        state.feed(measurement(theta_ca=5.0), 5.2 + i * 0.03)
    result = state.feed(measurement(theta_ca=5.0), 8.0)
    assert state.mode is Mode.MONITORING
    assert state.baseline.theta_ca_deg == pytest.approx(5.0)
    assert result.baseline_changed


def test_collecting_reports_how_long_is_left():
    state = MonitorState("chenyue", countdown_s=0.0, baseline_s=10.0)
    state.request_baseline(0.0)
    state.feed(measurement(), 0.1)
    assert state.feed(measurement(), 3.0).phase_remaining_s == pytest.approx(7.1)


def test_the_baseline_is_only_finished_when_the_time_is_up():
    state = MonitorState("chenyue", countdown_s=0.0, baseline_s=10.0)
    state.request_baseline(0.0)
    for i in range(50):
        result = state.feed(measurement(), 0.1 + i * 0.1)
        assert result.mode is Mode.COLLECTING
    assert state.baseline is None


def test_c_during_the_countdown_starts_over():
    state = MonitorState("chenyue", countdown_s=5.0, baseline_s=2.0)
    state.request_baseline(0.0)
    state.feed(measurement(), 3.0)
    state.request_baseline(3.0)
    assert state.feed(measurement(), 6.0).phase_remaining_s == pytest.approx(2.0)


# ---- 取不到就不採用 ----------------------------------------------------

def test_too_few_frames_is_not_adopted_and_says_to_try_again():
    """一幀的零點比沒有零點更糟：單幀散佈與判定門檻是同一個量級。"""
    state = MonitorState("chenyue", countdown_s=0.0, baseline_s=1.0)
    state.request_baseline(0.0)
    state.feed(measurement(), 0.1)
    result = state.feed(measurement(), 1.5)
    assert result.mode is Mode.WAITING
    assert state.baseline is None
    assert "按 c" in result.notice


def test_too_many_rejected_frames_is_not_adopted():
    """留下來的幀可能全都偏向同一邊，而被擋掉的那些正好是另一種姿勢。"""
    state = MonitorState("chenyue", countdown_s=0.0, baseline_s=5.0)
    state.request_baseline(0.0)
    t = 0.05
    for i in range(200):
        t += 0.02
        state.feed(measurement(usable=(i % 4 != 0)), t)
    result = state.feed(measurement(), 6.0)
    assert result.mode is Mode.WAITING
    assert state.baseline is None
    assert "略過" in result.notice


def test_a_good_baseline_is_adopted_and_monitoring_starts():
    state = MonitorState("chenyue", countdown_s=0.0, baseline_s=3.0)
    result = _zero(state, theta_ca=7.0, step=0.05)
    assert state.mode is Mode.MONITORING
    assert state.baseline.theta_ca_deg == pytest.approx(7.0)
    assert result.baseline_changed
    # 這一幀顯示的要是新零點下的偏移，不是舊的。
    assert result.corrected[0] == pytest.approx(0.0)


# ---- 重新歸零 ----------------------------------------------------------

def test_pressing_c_again_rezeroes():
    state = MonitorState("chenyue", countdown_s=0.0, baseline_s=3.0, window=5)
    _zero(state, theta_ca=7.0, step=0.05)
    first = state.baseline.theta_ca_deg
    _zero(state, t0=100.0, theta_ca=20.0, step=0.05)
    assert state.baseline.theta_ca_deg != first
    assert state.baseline.theta_ca_deg == pytest.approx(20.0)


def test_rezeroing_clears_the_moving_average():
    """不清的話，換零點之後的前 N 幀平均是兩個零點的資料混在一起，
    而判定看的正是那個平均。"""
    state = MonitorState("chenyue", countdown_s=0.0, baseline_s=3.0, window=5)
    _zero(state, theta_ca=7.0, step=0.05)
    for i in range(5):
        state.feed(measurement(theta_ca=25.0), 50.0 + i)
    assert state.ca_window.count > 0
    _zero(state, t0=100.0, theta_ca=20.0, step=0.05)
    assert state.ca_window.count == 0


# ---- 監測 --------------------------------------------------------------

def test_monitoring_judges_against_the_baseline():
    state = MonitorState("chenyue", baseline=_baseline(theta_ca=0.0), window=3)
    for i in range(5):
        result = state.feed(measurement(theta_ca=25.0), float(i))
    assert result.verdict.posture is Posture.OVER


def test_no_judge_measures_without_judging():
    state = MonitorState("chenyue", baseline=_baseline(), window=3, no_judge=True)
    for i in range(5):
        result = state.feed(measurement(theta_ca=25.0), float(i))
    assert result.verdict is None
    assert state.ca_window.count == 3


def test_a_rejected_frame_does_not_enter_the_average():
    state = MonitorState("chenyue", baseline=_baseline(), window=10)
    state.feed(measurement(usable=False), 0.0)
    assert state.ca_window.count == 0


def test_the_average_is_cleared_once_the_subject_has_been_gone_long_enough():
    """不清的話，受試者離開座位之後會對著空椅子繼續回報上一個狀態。"""
    state = MonitorState("chenyue", baseline=_baseline(), window=3)
    for i in range(3):
        state.feed(measurement(theta_ca=5.0), float(i))
    assert state.ca_window.count == 3
    for i in range(3):
        state.feed(measurement(usable=False), 10.0 + i)
    assert state.ca_window.count == 0


# ---- 離開 --------------------------------------------------------------

def test_stop_is_sticky():
    state = MonitorState("chenyue", baseline=_baseline())
    state.stop()
    assert state.stopped
    state.feed(measurement(), 0.0)
    assert state.stopped


def test_every_mode_has_a_word_for_the_panel():
    """少一個的話面板會直接印中文，退回英文時就變成一排問號。"""
    from view.overlay import MODE_WORDS

    for mode in Mode:
        assert mode.value in MODE_WORDS


def test_every_keypoint_name_is_known():
    """_USED 打錯字的話這整份測試會在一個假的拓撲上跑。"""
    for name in _USED:
        assert name in COCO18_KEYPOINT_NAMES
