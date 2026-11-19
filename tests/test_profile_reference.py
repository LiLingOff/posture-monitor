"""正側面參考：角度定義、面向判斷、時間窗彙整與與雙目的比較。全部用合成資料。"""
import math

import numpy as np
import pytest

from geometry.profile_reference import (FrameSample, UP, compare, facing_sign,
                                        parse_plumb, parse_windows, pick_person,
                                        profile_angle_deg, report_lines,
                                        sample_from_person, stereo_segment,
                                        summarize_window)
from pose.keypoints import PersonKeypoints
from pose.topology import NUM_KEYPOINTS, keypoint_index


def _person(points: dict, confidence: float = 0.9) -> PersonKeypoints:
    pts = np.full((NUM_KEYPOINTS, 2), np.nan, dtype=np.float32)
    conf = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
    for name, xy in points.items():
        pts[keypoint_index(name)] = xy
        conf[keypoint_index(name)] = confidence
    return PersonKeypoints(points=pts, confidences=conf)


def _ear_for(theta_deg: float, facing: float, shoulder=(300.0, 400.0),
             length: float = 150.0, roll_deg: float = 0.0):
    """已知前傾角 θ 時，耳朵在影像裡的位置。roll_deg 是整張畫面順時針轉了幾度。"""
    t = math.radians(theta_deg)
    dx, dy_up = facing * length * math.sin(t), length * math.cos(t)
    r = math.radians(roll_deg)
    # 影像座標 y 向下，順時針旋轉 r：x' = x cos r - y sin r, y' = x sin r + y cos r
    x, y = dx, -dy_up
    return (shoulder[0] + x * math.cos(r) - y * math.sin(r),
            shoulder[1] + x * math.sin(r) + y * math.cos(r))


# ---------- 角度定義 ----------

@pytest.mark.parametrize("facing", [1.0, -1.0])
@pytest.mark.parametrize("theta", [-25.0, 0.0, 10.0, 40.0])
def test_angle_recovers_theta_for_either_facing(theta, facing):
    ear = _ear_for(theta, facing)
    assert profile_angle_deg(ear, (300.0, 400.0), facing) == pytest.approx(theta, abs=1e-6)


def test_forward_is_positive_when_facing_right():
    # 耳朵在肩膀右上方，面向右：前傾，正。
    assert profile_angle_deg((340.0, 250.0), (300.0, 400.0), 1.0) > 0
    # 同一組座標，面向左：頭是往後仰，負。
    assert profile_angle_deg((340.0, 250.0), (300.0, 400.0), -1.0) < 0


@pytest.mark.parametrize("roll", [-12.0, 3.0, 8.0])
def test_plumb_line_removes_phone_roll(roll):
    theta = 20.0
    ear = _ear_for(theta, 1.0, roll_deg=roll)
    # 鉛垂線在畫面裡同樣順時針轉了 roll：上方的點在 (x, y)，下方的點在 (0,0)
    r = math.radians(roll)
    top = (150.0 * math.sin(r), -150.0 * math.cos(r))
    plumb = parse_plumb(f"{top[0]:.4f},{top[1]:.4f},0,0")
    corrected = profile_angle_deg(ear, (300.0, 400.0), 1.0, up=plumb)
    assert corrected == pytest.approx(theta, abs=0.05)
    # 不給鉛垂線的話，翻滾會原封不動變成角度誤差
    assert profile_angle_deg(ear, (300.0, 400.0), 1.0) == pytest.approx(theta + roll, abs=0.05)


def test_missing_or_degenerate_returns_none_not_zero():
    assert profile_angle_deg(None, (1.0, 1.0), 1.0) is None
    assert profile_angle_deg((1.0, 1.0), None, 1.0) is None
    assert profile_angle_deg((300.0, 395.0), (300.0, 400.0), 1.0) is None  # 向量只有 5px
    assert profile_angle_deg((float("nan"), 1.0), (300.0, 400.0), 1.0) is None


def test_invalid_facing_is_rejected():
    with pytest.raises(ValueError):
        profile_angle_deg((340.0, 250.0), (300.0, 400.0), 0.0)


# ---------- 鉛垂線與面向 ----------

def test_parse_plumb_default_and_errors():
    assert np.allclose(parse_plumb(None), UP)
    assert np.allclose(parse_plumb(""), UP)
    assert np.allclose(parse_plumb("100,50,100,450"), UP)
    with pytest.raises(ValueError):
        parse_plumb("1,2,3")
    with pytest.raises(ValueError):
        parse_plumb("100,100,100,105")


def test_facing_sign_from_nose_and_ear():
    assert facing_sign((420.0, 200.0), (380.0, 190.0)) == 1.0
    assert facing_sign((340.0, 200.0), (380.0, 190.0)) == -1.0
    assert facing_sign((381.0, 200.0), (380.0, 190.0)) is None  # 水平差太小
    assert facing_sign(None, (380.0, 190.0)) is None


# ---------- 挑人與取樣 ----------

def test_pick_person_prefers_confident_one_with_both_points():
    a = _person({"right_ear": (1, 1), "right_shoulder": (2, 2)}, confidence=0.4)
    b = _person({"right_ear": (3, 3), "right_shoulder": (4, 4)}, confidence=0.9)
    c = _person({"right_ear": (5, 5)}, confidence=1.0)       # 缺肩，不能用
    assert pick_person([a, b, c], "right") is b
    assert pick_person([c], "right") is None
    assert pick_person([a, b], "right", min_confidence=0.95) is None


def test_pick_person_rejects_unknown_side():
    with pytest.raises(ValueError):
        pick_person([], "front")


def test_sample_from_person_records_reason_when_nobody():
    sample = sample_from_person(1.5, None, "right")
    assert sample.ear is None and "沒有偵測到" in sample.reason
    person = _person({"right_ear": (340, 250), "right_shoulder": (300, 400), "nose": (380, 260)})
    sample = sample_from_person(2.0, person, "right")
    assert sample.ear == (340.0, 250.0) and sample.nose == (380.0, 260.0)


# ---------- 時間窗 ----------

def _samples(theta_deg: float, n: int = 200, fps: float = 5.0, noise: float = 0.0,
             facing: float = 1.0, seed: int = 0, t0: float = 0.0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        ear = _ear_for(theta_deg, facing)
        ear = (ear[0] + rng.normal(0, noise), ear[1] + rng.normal(0, noise))
        nose = (ear[0] + facing * 40.0, ear[1] + 8.0)
        out.append(FrameSample(t0 + i / fps, ear=ear, shoulder=(300.0, 400.0), nose=nose))
    return out


def test_summarize_window_mean_and_facing():
    samples = _samples(15.0, noise=1.0)
    summary = summarize_window("seg", samples, 0.0, 40.0)
    assert summary.facing == 1.0 and summary.facing_agreement == 1.0
    assert summary.mean_deg == pytest.approx(15.0, abs=0.6)
    assert summary.sd_deg > 0 and summary.se_deg is not None
    assert summary.usable == len(samples) and not summary.notes


def test_summarize_window_only_uses_frames_inside():
    samples = _samples(10.0, n=100, t0=0.0) + _samples(30.0, n=100, t0=20.0)
    first = summarize_window("a", samples, 0.0, 20.0)
    second = summarize_window("b", samples, 20.0, 40.0)
    assert first.mean_deg == pytest.approx(10.0, abs=1e-6)
    assert second.mean_deg == pytest.approx(30.0, abs=1e-6)


def test_summarize_window_left_facing_gives_same_positive_angle():
    summary = summarize_window("seg", _samples(15.0, facing=-1.0), 0.0, 40.0)
    assert summary.facing == -1.0
    assert summary.mean_deg == pytest.approx(15.0, abs=1e-6)


def test_summarize_window_without_nose_needs_facing_override():
    samples = [FrameSample(s.t_s, ear=s.ear, shoulder=s.shoulder) for s in _samples(15.0)]
    summary = summarize_window("seg", samples, 0.0, 40.0)
    assert summary.mean_deg is None and any("面向" in n for n in summary.notes)
    forced = summarize_window("seg", samples, 0.0, 40.0, facing_override=1.0)
    assert forced.mean_deg == pytest.approx(15.0, abs=1e-6)


def test_summarize_window_warns_on_disagreeing_facing_and_few_frames():
    right, left = _samples(15.0, n=60), _samples(15.0, n=40, facing=-1.0, t0=12.0)
    mixed = summarize_window("seg", right + left, 0.0, 40.0)
    assert mixed.facing_agreement == pytest.approx(0.6)
    assert any("面向只有" in n for n in mixed.notes)
    few = summarize_window("seg", _samples(15.0, n=3), 0.0, 40.0)
    assert few.mean_deg is None and any("不報平均" in n for n in few.notes)
    empty = summarize_window("seg", _samples(15.0, n=10), 100.0, 110.0)
    assert empty.frames == 0 and empty.notes


def test_summarize_window_marks_mostly_missing_frames():
    good = _samples(15.0, n=30)
    bad = [FrameSample(30 / 5.0 + i / 5.0, reason="沒有人") for i in range(70)]
    summary = summarize_window("seg", good + bad, 0.0, 40.0)
    assert summary.mean_deg == pytest.approx(15.0, abs=1e-6)
    assert any("可用幀只佔" in n for n in summary.notes)


# ---------- 時間窗字串 ----------

def test_parse_windows():
    assert parse_windows("a=0:30, b=40:70") == [("a", 0.0, 30.0), ("b", 40.0, 70.0)]
    assert parse_windows(None, duration_s=95.0) == [("all", 0.0, 95.0)]
    for bad in ("a=30:0", "a=0-30", "=0:30", "a=x:30", "a=0:30,a=40:50"):
        with pytest.raises(ValueError):
            parse_windows(bad)
    with pytest.raises(ValueError):
        parse_windows(None)


# ---------- 與雙目比較 ----------

def _write_stereo_csv(path, values, usable=1):
    lines = ["# subject=S01", "# condition=upright", "frame,usable,theta_ca_deg"]
    for i, v in enumerate(values):
        lines.append(f"{i},{usable if v is not None else 0},{'' if v is None else v}")
    path.write_text("\n".join(lines), encoding="utf-8")


def test_stereo_segment_uses_only_usable_frames(tmp_path):
    path = tmp_path / "seg.csv"
    _write_stereo_csv(path, [10.0] * 40 + [None] * 10)
    segment = stereo_segment(path)
    assert segment.frames == 40 and segment.mean_deg == pytest.approx(10.0)
    short = tmp_path / "short.csv"
    _write_stereo_csv(short, [10.0, 11.0, 12.0])
    assert stereo_segment(short).mean_deg is None


def test_compare_difference_and_missing_sides(tmp_path):
    summary = summarize_window("seg", _samples(15.0, noise=1.0), 0.0, 40.0)
    path = tmp_path / "seg.csv"
    _write_stereo_csv(path, [25.0 + (i % 5) for i in range(80)])
    result = compare(summary, stereo_segment(path))
    assert result.difference_deg == pytest.approx(27.0 - summary.mean_deg, abs=0.1)
    assert result.error_deg is not None and result.error_deg > 0
    no_ref = compare(summarize_window("x", _samples(15.0, n=2), 0.0, 40.0), stereo_segment(path))
    assert no_ref.difference_deg is None and "沒有平均" in no_ref.note
    short = tmp_path / "short.csv"
    _write_stereo_csv(short, [1.0])
    assert compare(summary, stereo_segment(short)).difference_deg is None


def test_report_lines_mentions_limits(tmp_path):
    summary = summarize_window("seg", _samples(15.0), 0.0, 40.0)
    path = tmp_path / "seg.csv"
    _write_stereo_csv(path, [20.0] * 40)
    text = "\n".join(report_lines([summary], [compare(summary, stereo_segment(path))]))
    assert "seg" in text and "3D−參考" in text and "共用同一個關鍵點偵測模型" in text
