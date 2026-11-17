"""2D 關鍵點記錄與前作 2D θ_CA（說明書表4）。

合成資料：先決定真實的前傾角 θ_true 與方位角 φ，用弱透視投影成影像座標，
再餵給前作公式，看偏差的方向與大小是不是推導說的那樣。
"""
import csv
import math

import numpy as np
import pytest

from geometry.cohort import collect
from geometry.cohort_report import session_rows, to_csv, to_text
from geometry.legacy_2d import (predicted_2d_bias, theta_ca_2d,
                                theta_ca_2d_from_row)
from geometry.measurement_log import _COLUMNS, MeasurementLog
from geometry.session_analysis import (analyse_session, legacy_2d_by_azimuth,
                                       read_rows)
from geometry.session_report import format_session
from pose.topology import COCO18_KEYPOINT_NAMES, NUM_KEYPOINTS

SCALE_PX_PER_MM = 5.0
LENGTH_MM = 120.0
SHOULDER_PX = (400.0, 300.0)


def _image_offsets(theta_deg, phi_deg, side="right", pitch_deg=0.0):
    """弱透視下，肩→耳在影像上的 (x, y) 位移。

    前方分量 L·sinθ 在影像水平方向被壓成 sinφ 倍（φ=0 正面、90 正側面），
    其餘進入深度方向；向上分量 L·cosθ 不變。pitch 是相機俯仰，會讓深度
    分量混進影像的垂直方向。
    """
    t, f, p = (math.radians(v) for v in (theta_deg, phi_deg, pitch_deg))
    forward = LENGTH_MM * math.sin(t)
    up = LENGTH_MM * math.cos(t)
    x_forward = forward * math.sin(f)
    depth = forward * math.cos(f)
    up_seen = up * math.cos(p) - depth * math.sin(p)
    sign = -1.0 if side == "right" else 1.0   # 從右側看，人面向影像左方
    return sign * x_forward * SCALE_PX_PER_MM, -up_seen * SCALE_PX_PER_MM


def _theta_2d(theta_deg, phi_deg, side="right", pitch_deg=0.0):
    dx, dy = _image_offsets(theta_deg, phi_deg, side, pitch_deg)
    xs, ys = SHOULDER_PX
    return theta_ca_2d((xs + dx, ys + dy), SHOULDER_PX, side)


# ---- 公式本身 -------------------------------------------------------------

@pytest.mark.parametrize("theta", [-20.0, -5.0, 0.0, 10.0, 30.0, 50.0])
@pytest.mark.parametrize("side", ["right", "left"])
def test_matches_the_true_angle_at_45_degrees(theta, side):
    assert _theta_2d(theta, 45.0, side) == pytest.approx(theta, abs=1e-9)


@pytest.mark.parametrize("theta, phi, expected_bias", [
    (30, 30, -7.79), (30, 55, +3.78),
    (20, 30, -5.57), (20, 55, +2.86),
    (10, 30, -2.89), (10, 55, +1.54),
])
def test_bias_away_from_45_degrees_matches_the_derivation(theta, phi, expected_bias):
    measured = _theta_2d(theta, phi) - theta
    assert measured == pytest.approx(expected_bias, abs=0.01)
    assert predicted_2d_bias(theta, phi) == pytest.approx(expected_bias, abs=0.01)


def test_underestimates_below_45_and_overestimates_above():
    assert _theta_2d(25.0, 20.0) < 25.0
    assert _theta_2d(25.0, 70.0) > 25.0
    # 後仰時方向相反：低估是「絕對值變小」。
    assert -10.0 < _theta_2d(-10.0, 20.0) < 0.0


def test_the_sign_is_forward_positive_on_both_sides():
    assert _theta_2d(15.0, 45.0, "right") > 0
    assert _theta_2d(15.0, 45.0, "left") > 0
    assert _theta_2d(-15.0, 45.0, "right") < 0


def test_a_tilted_camera_biases_even_at_45_degrees():
    """相機有俯仰時 45° 也不無偏，而且偏的方向跟著俯仰的正負。"""
    down = _theta_2d(20.0, 45.0, pitch_deg=+10.0) - 20.0
    up = _theta_2d(20.0, 45.0, pitch_deg=-10.0) - 20.0
    assert abs(down) > 0.3 and abs(up) > 0.3
    assert down * up < 0


def test_missing_coordinates_give_none_not_zero():
    assert theta_ca_2d(None, (1.0, 2.0), "right") is None
    assert theta_ca_2d((float("nan"), 1.0), (1.0, 2.0), "right") is None
    with pytest.raises(ValueError):
        theta_ca_2d((0.0, 0.0), (0.0, 1.0), "middle")


def test_ear_level_with_or_below_the_shoulder_does_not_divide_by_zero():
    level = theta_ca_2d((380.0, 300.0), SHOULDER_PX, "right")
    assert level == pytest.approx(90.0)
    # 耳朵在肩膀下方且偏向前方：幾何上是超過 90° 的角，不是除以零也不是翻號。
    assert theta_ca_2d((390.0, 310.0), SHOULDER_PX, "right") == pytest.approx(
        math.degrees(math.atan2(10.0 / math.sin(math.radians(45.0)), -10.0)))


# ---- 寫進 CSV --------------------------------------------------------------

def _points(**named):
    pts = np.full((NUM_KEYPOINTS, 2), np.nan, np.float32)
    for name, xy in named.items():
        pts[COCO18_KEYPOINT_NAMES.index(name)] = xy
    return pts


def _rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def test_pixel_columns_are_last_and_blank_when_a_point_is_missing(tmp_path):
    from test_measurement_log import _measurement

    left = _points(right_ear=(10.5, 20.25), right_shoulder=(30.0, 40.0))
    right = _points(right_ear=(11.0, 21.0))
    path = tmp_path / "s.csv"
    with MeasurementLog(path) as log:
        log.write(_measurement(), keypoints_2d=(left, right))
        log.write(_measurement())            # 沒有影像
        log.write(None, reject_reason="讀取相機影格失敗")
    row, no_image, nothing = _rows(path)

    assert list(row)[-16:] == [f"{n}_{e}{a}_px"
                               for n in ("right_ear", "right_shoulder",
                                         "left_shoulder", "left_ear")
                               for e in "lr" for a in "xy"]
    assert float(row["right_ear_lx_px"]) == pytest.approx(10.5)
    assert float(row["right_ear_ly_px"]) == pytest.approx(20.25)
    assert float(row["right_shoulder_lx_px"]) == pytest.approx(30.0)
    assert float(row["right_ear_rx_px"]) == pytest.approx(11.0)
    assert row["right_shoulder_rx_px"] == ""      # 右眼沒偵測到：空白，不是 0
    assert row["left_ear_lx_px"] == ""
    assert all(no_image[c] == "" for c in _COLUMNS[-16:])
    assert all(nothing[c] == "" for c in _COLUMNS[-16:])


# ---- 分析 ------------------------------------------------------------------

def _csv_row(frame, theta_true, phi, side="right", azimuth=None, usable=1):
    dx, dy = _image_offsets(theta_true, phi, side)
    xs, ys = SHOULDER_PX
    row = {"frame": frame, "elapsed_s": frame * 0.1, "usable": usable,
           "theta_ca_deg": theta_true,
           "azimuth_deg": phi if azimuth is None else azimuth}
    row[f"{side}_ear_lx_px"], row[f"{side}_ear_ly_px"] = xs + dx, ys + dy
    row[f"{side}_shoulder_lx_px"], row[f"{side}_shoulder_ly_px"] = xs, ys
    return row


def _write(path, rows, subject="A", condition="head-forward", trial=1):
    lines = [f"# subject={subject}", f"# condition={condition}", f"# trial={trial}"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines) + "\n")
        writer = csv.DictWriter(f, fieldnames=_COLUMNS, restval="", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_analyse_reports_2d_against_3d_at_the_actual_azimuth(tmp_path):
    rows = [_csv_row(i, 30.0, 30.0) for i in range(40)]
    summary = analyse_session(_write(tmp_path / "a.csv", rows))
    legacy = summary.legacy_2d
    assert legacy.frames == 40
    assert legacy.azimuth_deg == pytest.approx(30.0)
    assert legacy.theta_3d_deg == pytest.approx(30.0)
    assert legacy.difference_deg == pytest.approx(-7.79, abs=0.01)


def test_azimuth_is_the_measured_one_not_the_nominal_one(tmp_path):
    """目標 45° 但實際算出 52°：分箱與偏差都該用 52°。"""
    rows = [_csv_row(i, 30.0, 52.0) for i in range(30)]
    summary = analyse_session(_write(tmp_path / "a.csv", rows))
    (only,) = legacy_2d_by_azimuth(summary.legacy_pairs)
    assert (only.low, only.high) == (50.0, 60.0)
    assert only.stats.difference_deg == pytest.approx(predicted_2d_bias(30.0, 52.0), abs=0.01)


def test_bins_below_the_frame_minimum_are_not_reported(tmp_path):
    rows = [_csv_row(i, 30.0, 30.0) for i in range(30)] + \
           [_csv_row(100 + i, 30.0, 55.0) for i in range(5)]
    summary = analyse_session(_write(tmp_path / "a.csv", rows))
    bins = legacy_2d_by_azimuth(summary.legacy_pairs)
    assert [b.low for b in bins] == [30.0]
    assert summary.legacy_2d.frames == 35     # 總計仍包含所有配得上的幀


def test_a_side_on_view_at_exactly_90_degrees_stays_in_the_last_bin():
    (only,) = legacy_2d_by_azimuth([(90.0, 30.0, 40.0)] * 15)
    assert (only.low, only.high) == (80.0, 90.0)


def test_unusable_frames_and_frames_missing_a_value_are_left_out(tmp_path):
    rows = [_csv_row(i, 20.0, 45.0) for i in range(10)]
    rows.append(_csv_row(11, 20.0, 45.0, usable=0))
    gap = _csv_row(12, 20.0, 45.0)
    gap["right_ear_lx_px"] = ""
    rows.append(gap)
    rows.append({**_csv_row(13, 20.0, 45.0), "azimuth_deg": ""})
    summary = analyse_session(_write(tmp_path / "a.csv", rows))
    assert summary.legacy_2d.frames == 10


def test_a_file_from_before_the_pixel_columns_still_reads(tmp_path):
    old = [c for c in _COLUMNS if not c.endswith("_px") or c.startswith("dy_")]
    path = tmp_path / "old.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("# subject=A\n")
        writer = csv.DictWriter(f, fieldnames=old, restval="", lineterminator="\n")
        writer.writeheader()
        for i in range(20):
            writer.writerow({"frame": i, "usable": 1, "theta_ca_deg": 5,
                             "azimuth_deg": 30, "elapsed_s": i * 0.1})
    summary = analyse_session(path)
    assert summary.legacy_2d is None
    assert "前作 2D" not in format_session(summary)
    assert theta_ca_2d_from_row(read_rows(path)[0][0]) is None


def test_the_left_side_is_used_when_the_right_one_is_incomplete():
    row = _csv_row(1, 20.0, 45.0, side="left")
    angle, side = theta_ca_2d_from_row(row)
    assert side == "left"
    assert angle == pytest.approx(20.0)


# ---- 報表與彙整 -------------------------------------------------------------

def test_reports_and_cohort_csv_carry_the_2d_columns(tmp_path):
    for name, phi in (("a30.csv", 30.0), ("a45.csv", 45.0), ("a55.csv", 55.0)):
        _write(tmp_path / name, [_csv_row(i, 30.0, phi) for i in range(30)])
    cohort = collect([tmp_path])
    rows = {r["theta_ca_2d_azimuth_deg"]: r for r in session_rows(cohort)}
    diffs = sorted(round(r["theta_ca_2d_minus_3d_deg"], 2) for r in rows.values())
    assert diffs == [-7.79, 0.0, 3.78]

    header = to_csv(session_rows(cohort)).splitlines()[0].split(",")
    assert header[-5:] == ["theta_ca_2d_deg", "theta_ca_3d_paired_deg",
                           "theta_ca_2d_minus_3d_deg", "theta_ca_2d_frames",
                           "theta_ca_2d_azimuth_deg"]
    text = to_text(cohort, None)
    assert "前作 2D θ_CA 對照" in text
    assert "按實際方位角分箱" in text
    assert "-7.79" in text and "+3.78" in text


def test_cohort_without_pixel_columns_has_no_2d_section(tmp_path):
    from test_theta_ka import _write_session
    _write_session(tmp_path, "old.csv", ka=None)
    cohort = collect([tmp_path])
    assert "前作 2D" not in to_text(cohort, None)
    assert session_rows(cohort)[0]["theta_ca_2d_deg"] is None
