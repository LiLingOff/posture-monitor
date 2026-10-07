"""肩部垂直位移：前作報告書的第四個演算法，換成毫米。"""
from __future__ import annotations

import json

import numpy as np
import pytest

from geometry.baseline import BaselineCollector, PostureBaseline
from geometry.judgement import Posture, PostureJudge
from geometry.keypoints3d import PersonKeypoints3D
from geometry.pipeline import shoulder_height_mm
from geometry.smoothing import RollingAngle
from pose.topology import NUM_KEYPOINTS, keypoint_index
from monitor_fakes import baseline as _baseline
from monitor_fakes import measurement


def _shoulders(left_y: float, right_y: float) -> PersonKeypoints3D:
    points = np.full((NUM_KEYPOINTS, 3), np.nan)
    points[keypoint_index("left_shoulder")] = (-180.0, left_y, 650.0)
    points[keypoint_index("right_shoulder")] = (180.0, right_y, 650.0)
    return PersonKeypoints3D(points=points)


# ---- 量出來的高度 ------------------------------------------------------

def test_the_height_is_the_midpoint_of_the_two_shoulders():
    """取中點不取較低的那一側。前作取低的是為了靈敏，代價是單側偵測失誤
    直接變成一次誤報；左右不對稱另外有 θ_sym 在管。"""
    assert shoulder_height_mm(_shoulders(100.0, 140.0)) == pytest.approx(-120.0)


def test_up_is_positive():
    """OpenCV 的 Y 向下為正，直接用的話「肩膀變高」會是負的。"""
    high = shoulder_height_mm(_shoulders(100.0, 100.0))
    low = shoulder_height_mm(_shoulders(160.0, 160.0))
    assert high > low


def test_a_missing_shoulder_means_no_height():
    """一邊沒偵測到就量不到，回傳 None 而不是拿單邊湊。"""
    points = np.full((NUM_KEYPOINTS, 3), np.nan)
    points[keypoint_index("left_shoulder")] = (-180.0, 100.0, 650.0)
    assert shoulder_height_mm(PersonKeypoints3D(points=points)) is None


def test_the_height_does_not_depend_on_the_distance():
    """前作用像素，20px 的門檻只在他們那個 60cm 的架設下成立。
    有真實 3D 就直接量毫米，受試者往前坐也不影響。"""
    near = _shoulders(100.0, 100.0)
    far = PersonKeypoints3D(points=near.points.copy())
    far.points[:, 2] = 1200.0
    assert shoulder_height_mm(near) == shoulder_height_mm(far)


# ---- 門檻 --------------------------------------------------------------

def test_the_threshold_comes_from_the_subjects_own_sway():
    """前作量 20 位受試者訂一個通用值。我們有這個人自己的標準差。"""
    steady = _baseline(shoulder_height_std_mm=3.0)
    restless = _baseline(shoulder_height_std_mm=12.0)
    assert restless.shoulder_drop_threshold_mm > steady.shoulder_drop_threshold_mm


def test_a_very_steady_subject_still_gets_a_floor():
    """標準差趨近 0 的話門檻也趨近 0，真實的微幅移動就會一直觸發。"""
    assert _baseline(shoulder_height_std_mm=0.1).shoulder_drop_threshold_mm >= 10.0


def test_there_is_no_threshold_without_a_recorded_height():
    assert _baseline(shoulder_height_mm=None,
                     shoulder_height_std_mm=None).shoulder_drop_threshold_mm is None


def test_the_drop_is_positive_when_the_shoulders_sink():
    base = _baseline(shoulder_height_mm=-120.0, shoulder_height_std_mm=4.0)
    assert base.shoulder_drop_mm(-140.0) == pytest.approx(20.0)
    assert base.shoulder_drop_mm(-100.0) == pytest.approx(-20.0)


def test_no_drop_without_a_height_on_either_side():
    """0 是「高度剛好一樣」，與「量不到」是兩回事。"""
    assert _baseline(shoulder_height_mm=None).shoulder_drop_mm(-140.0) is None
    assert _baseline(shoulder_height_mm=-120.0).shoulder_drop_mm(None) is None


# ---- 舊的基準檔 --------------------------------------------------------

def _old_baseline_json() -> dict:
    return {
        "subject": "chenyue", "captured_at": "2026-09-29T20:06:12", "frames": 149,
        "rejected": 0, "duration_s": 30.0, "theta_ca_deg": 3.78,
        "theta_sym_deg": 5.3, "theta_ca_std_deg": 4.5, "theta_sym_std_deg": 0.5,
        "theta_ca_standard_error_deg": 0.83, "theta_sym_standard_error_deg": 0.09,
        "distance_mm": 645.0, "azimuth_deg": 20.0,
    }


def test_a_baseline_taken_before_this_field_existed_still_loads(tmp_path):
    """讀不進來的話那些檔案沒有任何地方能補回肩高，整份基準就作廢了。"""
    path = tmp_path / "old.json"
    path.write_text(json.dumps(_old_baseline_json()), encoding="utf-8")
    loaded = PostureBaseline.load(path)
    assert loaded.theta_ca_deg == pytest.approx(3.78)
    assert loaded.shoulder_height_mm is None


def test_a_baseline_missing_a_required_field_is_still_rejected(tmp_path):
    """放寬的只有有預設值的欄位。θ_CA 少了就真的不能用。"""
    data = _old_baseline_json()
    del data["theta_ca_deg"]
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        PostureBaseline.load(path)


def test_an_old_baseline_simply_does_not_judge_this_axis(tmp_path):
    """舊的基準不該讓駝背那一項變成「未知」而把整體判定拖下水。"""
    judge = PostureJudge()
    judge.watch_shoulder_drop(PostureBaseline.load(
        _write(tmp_path, _old_baseline_json())).shoulder_drop_threshold_mm)
    assert judge.shoulder_drop is None

    ca, sym = RollingAngle(2), RollingAngle(2)
    for _ in range(2):
        ca.add(1.0)
        sym.add(0.5)
    *_, combined = judge.update(ca, sym, RollingAngle(2))
    assert combined.posture is Posture.OK


def _write(tmp_path, data) -> object:
    path = tmp_path / "b.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---- 收集 --------------------------------------------------------------

def test_the_collector_records_the_height_and_its_spread():
    collector = BaselineCollector(minimum_frames=3)
    for height in (-120.0, -124.0, -116.0, -120.0):
        collector.add(measurement(shoulder_height_mm=height))
    baseline = collector.finish("chenyue", 20.0)
    assert baseline.shoulder_height_mm == pytest.approx(-120.0)
    assert baseline.shoulder_height_std_mm == pytest.approx(2.83, abs=0.01)


def test_frames_without_a_height_do_not_stop_the_baseline():
    """肩高要雙肩同時偵測到。少了它只是這一項不判，不該讓整份基準失敗。"""
    collector = BaselineCollector(minimum_frames=3)
    for _ in range(4):
        collector.add(measurement(shoulder_height_mm=None))
    baseline = collector.finish("chenyue", 20.0)
    assert baseline.theta_ca_deg is not None
    assert baseline.shoulder_height_mm is None
    assert baseline.shoulder_drop_threshold_mm is None


# ---- 判定 --------------------------------------------------------------

def _full(window: RollingAngle, value: float) -> RollingAngle:
    for _ in range(window.window):
        window.add(value)
    return window


def test_a_sinking_shoulder_is_over_the_threshold():
    judge = PostureJudge()
    judge.watch_shoulder_drop(10.0)
    ca, sym = _full(RollingAngle(4), 1.0), _full(RollingAngle(4), 0.5)
    *_, combined = judge.update(ca, sym, _full(RollingAngle(4), 40.0))
    assert combined.posture is Posture.OVER
    assert "肩部垂直位移" in combined.reason


def test_shoulders_at_the_baseline_height_are_normal():
    judge = PostureJudge()
    judge.watch_shoulder_drop(10.0)
    ca, sym = _full(RollingAngle(4), 1.0), _full(RollingAngle(4), 0.5)
    *_, combined = judge.update(ca, sym, _full(RollingAngle(4), 0.0))
    assert combined.posture is Posture.OK


def test_sitting_up_taller_than_the_baseline_is_not_a_problem():
    """門檻是單邊的：坐得比基準挺不是駝背。"""
    judge = PostureJudge()
    judge.watch_shoulder_drop(10.0)
    ca, sym = _full(RollingAngle(4), 1.0), _full(RollingAngle(4), 0.5)
    *_, combined = judge.update(ca, sym, _full(RollingAngle(4), -40.0))
    assert combined.posture is Posture.OK


def test_the_reason_is_written_in_millimetres():
    """這一項的單位不是度。照抄角度的格式會印出「40.0°」。"""
    judge = PostureJudge()
    judge.watch_shoulder_drop(10.0)
    ca, sym = _full(RollingAngle(4), 1.0), _full(RollingAngle(4), 0.5)
    _, _, drop, _ = judge.update(ca, sym, _full(RollingAngle(4), 40.0))
    assert "mm" in drop.reason
    assert "°" not in drop.reason


def test_the_axis_is_inert_when_it_was_never_switched_on():
    """既有的記錄重播出來要與加這一項之前完全一樣。"""
    judge = PostureJudge()
    ca, sym = _full(RollingAngle(4), 1.0), _full(RollingAngle(4), 0.5)
    _, _, drop, combined = judge.update(ca, sym, _full(RollingAngle(4), 999.0))
    assert drop is None
    assert combined.posture is Posture.OK
