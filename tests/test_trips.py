"""趟次層級的分析：同趟配對差值、軀幹前傾肩高、側傾、表4 比值、略過率分軸。

全部用合成的逐幀 CSV。數字刻意取整，答案可以心算。
"""
import csv
import math

import pytest

from geometry.cohort import collect
from geometry.cohort_report import session_rows, to_csv, to_text
from geometry.session_analysis import analyse_session
from geometry.trip_report import trip_csvs, trip_sections
from geometry.trips import (NOTED, UNPAIRED, VOID, axis_rejection_rate,
                            paired_differences, ratio_by_group, ratio_rows,
                            read_groups, sensitivity, tilt_by_angle, tilt_rows,
                            tilt_symmetry)

_SIN45 = math.sin(math.radians(45.0))
_FIELDS = ["frame", "elapsed_s", "usable", "reject_reason", "theta_ca_deg",
           "theta_sym_deg", "theta_ca_corrected_deg", "theta_sym_corrected_deg",
           "azimuth_deg", "shoulder_height_mm", "theta_ka_deg",
           "right_ear_lx_px", "right_ear_ly_px",
           "right_shoulder_lx_px", "right_shoulder_ly_px",
           "left_shoulder_lx_px", "left_shoulder_ly_px", "posture"]


def _write(folder, name, *, subject="s01", condition="upright", trial=1, trip="T1",
           ca=0.0, sym=0.0, ka=None, azimuth=40.0, shoulder=-100.0, frames=60,
           rejected=0, sym_missing=0, baseline=None, extra=None, theta_2d=None,
           sym_2d=None,
           captured_at="2026-11-18T10:00:00"):
    """一段合成 CSV。baseline=(θ_CA, θ_sym, 方位角) 代表這段有扣基準。"""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    lines = [f"# captured_at={captured_at}", f"# subject={subject}",
             f"# condition={condition}", f"# trial={trial}"]
    if baseline is not None:
        lines += ["# baseline_file=b.json",
                  f"# baseline_theta_ca_deg={baseline[0]}",
                  f"# baseline_theta_sym_deg={baseline[1]}",
                  f"# baseline_azimuth_deg={baseline[2]}"]
    if trip is not None:
        lines.append(f"# trip_id={trip}")
    for key, value in (extra or {}).items():
        lines.append(f"# {key}={value}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(lines) + "\n")
        writer = csv.DictWriter(handle, fieldnames=_FIELDS, lineterminator="\n")
        writer.writeheader()
        for i in range(frames):
            wiggle = 0.5 if i % 2 else -0.5
            row = {"frame": i + 1, "elapsed_s": i * 0.2, "usable": 1,
                   "theta_ca_deg": ca + wiggle,
                   "theta_sym_deg": "" if i < sym_missing else sym + wiggle,
                   "azimuth_deg": azimuth, "shoulder_height_mm": shoulder + wiggle,
                   "theta_ka_deg": "" if ka is None else ka + wiggle, "posture": ""}
            if baseline is not None:
                row["theta_ca_corrected_deg"] = ca + wiggle - baseline[0]
                if i >= sym_missing:
                    row["theta_sym_corrected_deg"] = sym + wiggle - baseline[1]
            if theta_2d is not None:
                # 右側：前方位移 = −(x_ear − x_shoulder)/sin45，向上 = y_shoulder − y_ear
                t3 = math.radians(ca + wiggle)
                t2 = math.atan(theta_2d * math.tan(t3))   # theta_2d 當作比值
                row.update({"right_shoulder_lx_px": 500.0, "right_shoulder_ly_px": 400.0,
                            "right_ear_ly_px": 300.0,
                            "right_ear_lx_px": 500.0 - math.tan(t2) * 100.0 * _SIN45})
            if sym_2d is not None:
                # 雙肩 x 差 100px，左肩 y 比右肩大多少由 2D 角度決定（右肩較高為正）
                tan2 = math.tan(math.radians(sym_2d + wiggle))
                row.update({"right_shoulder_lx_px": 400.0, "right_shoulder_ly_px": 300.0,
                            "left_shoulder_lx_px": 500.0,
                            "left_shoulder_ly_px": 300.0 + 100.0 * tan2})
            writer.writerow(row)
        for i in range(rejected):
            writer.writerow({"frame": frames + i + 1, "elapsed_s": 0, "usable": 0,
                             "reject_reason": "沒有偵測到人", "azimuth_deg": azimuth})
    return path


def _load(folder):
    return [analyse_session(p) for p in sorted(folder.rglob("*.csv"))]


# ---- 配對差值 ----------------------------------------------------------

def test_natural_is_paired_with_the_same_trip_upright_not_another_trip(tmp_path):
    folder = tmp_path / "s01"
    _write(folder, "natural-1.csv", condition="natural", trip="T1", ca=20.0, sym=3.0,
           ka=12.0, extra={"desk_level": "base", "lift_cm": 0, "desk_offset_cm": -1.5})
    _write(folder, "upright-1.csv", condition="upright", trip="T1", ca=8.0, sym=1.0,
           ka=5.0, baseline=(8.0, 1.0, 40.0), captured_at="2026-11-18T10:02:00")
    # 另一趟的 upright 零點差很多，不能被拿來配
    _write(folder, "upright-2.csv", condition="upright", trial=2, trip="T2", ca=24.0,
           sym=10.0, baseline=(24.0, 10.0, 40.0))
    rows = paired_differences(_load(tmp_path))
    assert len(rows) == 1
    row = rows[0]
    assert row.paired and row.reference.path.name == "upright-1.csv"
    assert row.differences["θ_CA"].value == pytest.approx(12.0)
    assert row.differences["θ_sym"].value == pytest.approx(2.0)
    assert row.differences["θ_KA"].value == pytest.approx(7.0)
    assert row.differences["θ_CA"].error is not None
    assert row.desk_level == "base"


def test_the_first_upright_of_the_trip_is_the_reference(tmp_path):
    """前 5 人趟 1 最後多量一段端正坐檢查漂移，配對用的是第一段。"""
    folder = tmp_path / "s01"
    _write(folder, "natural-1.csv", condition="natural", ca=20.0)
    _write(folder, "upright-1.csv", ca=8.0, captured_at="2026-11-18T10:02:00")
    _write(folder, "upright-2.csv", trial=2, ca=12.0, captured_at="2026-11-18T10:09:00")
    row = paired_differences(_load(tmp_path))[0]
    assert row.reference.path.name == "upright-1.csv"
    assert row.differences["θ_CA"].value == pytest.approx(12.0)


def test_unpaired_natural_segments_are_listed_with_a_reason(tmp_path):
    folder = tmp_path / "s01"
    _write(folder, "natural-1.csv", condition="natural", trip="T1")
    _write(folder, "natural-2.csv", condition="natural", trial=2, trip=None)
    rows = paired_differences(_load(tmp_path))
    reasons = sorted(r.reason for r in rows)
    assert all(not r.paired for r in rows)
    assert any("同一趟沒有 upright" in r for r in reasons)
    assert any("trip_id" in r for r in reasons)
    text = dict(trip_sections(_load(tmp_path)))["自然坐配對差值"]
    assert UNPAIRED in text
    csv_text = trip_csvs(_load(tmp_path))["paired_natural.csv"]
    assert csv_text.count("unpaired") == 2


def test_a_natural_segment_that_turned_is_void_even_if_clean(tmp_path):
    folder = tmp_path / "s01"
    _write(folder, "natural-1.csv", condition="natural", azimuth=58.0)
    _write(folder, "upright-1.csv", azimuth=44.0, baseline=(0.0, 0.0, 44.0),
           captured_at="2026-11-18T10:02:00")
    row = paired_differences(_load(tmp_path))[0]
    assert row.turned_deg == pytest.approx(14.0)
    assert row.differences["θ_CA"].status == VOID
    assert sensitivity([row], "θ_CA").n == 0


def test_the_regression_csv_has_the_columns_asked_for(tmp_path):
    folder = tmp_path / "s01"
    _write(folder, "natural-1.csv", condition="natural", ca=20.0,
           extra={"desk_level": "desk-up", "lift_cm": 5, "desk_offset_cm": 3.5})
    _write(folder, "upright-1.csv", ca=8.0, captured_at="2026-11-18T10:02:00")
    text = trip_csvs(_load(tmp_path))["paired_natural.csv"]
    rows = list(csv.DictReader(text.splitlines()))
    assert rows[0]["subject"] == "s01"
    assert rows[0]["desk_level"] == "desk-up"
    assert rows[0]["lift_cm"] == "5" and rows[0]["desk_offset_cm"] == "3.5"
    assert float(rows[0]["d_theta_ca_deg"]) == pytest.approx(12.0)
    for column in ("d_theta_sym_deg", "d_theta_ka_deg", "d_theta_ca_deg_error"):
        assert column in rows[0]


def test_the_cross_subject_mean_counts_people_not_rows(tmp_path):
    for subject, diff in (("s01", 10.0), ("s02", 14.0)):
        folder = tmp_path / subject
        _write(folder, "natural-1.csv", subject=subject, condition="natural",
               ca=diff, extra={"desk_level": "base"})
        _write(folder, "upright-1.csv", subject=subject, ca=0.0,
               extra={"desk_level": "base"}, captured_at="2026-11-18T10:02:00")
    rows = paired_differences(_load(tmp_path))
    g = sensitivity(rows, "θ_CA", "base")
    assert g.n == 2 and g.mean == pytest.approx(12.0)
    assert g.error == pytest.approx(2.0)     # std(ddof=1)=2.83, /√2


# ---- 略過率分軸 --------------------------------------------------------

def test_the_skip_rate_is_judged_per_axis(tmp_path):
    """θ_sym 少 18%：θ_sym 在 20% 以內是納入；θ_CA 沒缺。"""
    path = _write(tmp_path, "a.csv", frames=100, sym_missing=18)
    summary = analyse_session(path)
    assert axis_rejection_rate(summary, "θ_CA") == pytest.approx(0.0)
    assert axis_rejection_rate(summary, "θ_sym") == pytest.approx(0.18)
    assert axis_rejection_rate(summary, "θ_KA") == axis_rejection_rate(summary, "θ_CA")


def test_over_the_limit_is_noted_and_the_sensitivity_runs_exclude_it(tmp_path):
    # 18% 的整幀略過：θ_CA 超過 15%（附註納入），θ_sym 在 20% 以內
    for subject, rejected in (("s01", 0), ("s02", 22)):
        folder = tmp_path / subject
        _write(folder, "natural-1.csv", subject=subject, condition="natural", ca=10.0,
               sym=2.0, frames=100, rejected=rejected)
        _write(folder, "upright-1.csv", subject=subject, ca=0.0, sym=0.0, frames=100,
               captured_at="2026-11-18T10:02:00")
    rows = paired_differences(_load(tmp_path))
    noisy = next(r for r in rows if r.subject == "s02")
    assert noisy.differences["θ_CA"].status == NOTED
    assert noisy.differences["θ_sym"].status == "納入"
    assert sensitivity(rows, "θ_CA").n == 2
    assert sensitivity(rows, "θ_CA", cutoff=0.15).n == 1
    assert sensitivity(rows, "θ_CA", cutoff=0.20).n == 2
    assert sensitivity(rows, "θ_sym", cutoff=0.15).n == 1


def test_a_turned_segment_stays_out_of_the_cohort_whatever_the_limit(tmp_path):
    """head-forward-2 那種轉身（方位角漂超過 10°）放寬略過率也救不回來。"""
    folder = tmp_path / "s01"
    _write(folder, "upright-1.csv", ca=0.0, baseline=(0.0, 0.0, 30.0), azimuth=31.0)
    _write(folder, "head-forward-1.csv", condition="head-forward", ca=20.0,
           baseline=(0.0, 0.0, 30.0), azimuth=55.0)
    cohort = collect([tmp_path])
    assert cohort.result("s01", "head-forward").usable_sessions == []
    assert cohort.with_limit(0.20).result("s01", "head-forward").usable_sessions == []
    text = to_text(cohort, None)
    assert "作廢" in text


def test_cohort_csv_carries_the_inclusion_columns_and_old_files_still_read(tmp_path):
    _write(tmp_path / "s01", "upright-1.csv", trip=None)
    rows = list(csv.DictReader(to_csv(session_rows(collect([tmp_path]))).splitlines()))
    assert rows[0]["theta_ca_inclusion"] == "納入"
    assert rows[0]["trip_id"] == ""


# ---- 軀幹前傾 ----------------------------------------------------------

def test_forward_reports_shoulder_height_and_theta_ka_side_by_side(tmp_path):
    folder = tmp_path / "s01"
    _write(folder, "upright-1.csv", shoulder=-100.0, ka=2.0, baseline=(0, 0, 40.0))
    _write(folder, "forward-1.csv", condition="forward", shoulder=-130.0, ka=22.0,
           baseline=(0, 0, 40.0), captured_at="2026-11-18T10:05:00")
    sections = dict(trip_sections(_load(tmp_path)))
    text = sections["軀幹前傾：肩高與 θ_KA"]
    assert "不是兩份獨立證據" in text
    assert "58.0" not in text.replace("58.0mm 已作廢", "")
    rows = list(csv.DictReader(trip_csvs(_load(tmp_path))["forward_shoulder.csv"]
                               .splitlines()))
    assert float(rows[0]["d_shoulder_height_mm"]) == pytest.approx(-30.0)
    assert float(rows[0]["d_theta_ka_deg"]) == pytest.approx(20.0)


# ---- 側傾 --------------------------------------------------------------

def _tilt(folder, name, condition, sym, measured, target, subject="s01"):
    return _write(folder, name, subject=subject, condition=condition, sym=sym,
                  baseline=(0.0, 0.0, 40.0),
                  extra={"target_angle_deg": target, "measured_angle_deg": measured})


def test_leaning_left_is_expected_positive_and_the_error_is_signed(tmp_path):
    """θ_sym 右肩較高為正；往左側彎時右肩較高。"""
    folder = tmp_path / "s01"
    _tilt(folder, "tilt-left-1.csv", "tilt-left", sym=18.0, measured=20, target=20)
    _tilt(folder, "tilt-right-1.csv", "tilt-right", sym=-21.0, measured=20, target=20)
    rows = {r.summary.condition: r for r in tilt_rows(_load(tmp_path))}
    left, right = rows["tilt-left"], rows["tilt-right"]
    assert left.expected_deg == 20 and right.expected_deg == -20
    assert left.error_deg == pytest.approx(-2.0)
    assert right.error_deg == pytest.approx(-1.0)
    assert left.sign_ok and right.sign_ok
    assert left.magnitude_error_deg == pytest.approx(-2.0)
    assert right.magnitude_error_deg == pytest.approx(1.0)
    assert tilt_symmetry(list(rows.values()))["s01"] == pytest.approx((-2.0, 1.0))


def test_the_end_value_makes_the_reference_the_average_of_start_and_end(tmp_path):
    """θ_sym 是整段 60 秒的平均，上身慢慢滑掉時拿開始值比會把滑掉的算進誤差。"""
    folder = tmp_path / "s01"
    _write(folder, "tilt-left-1.csv", condition="tilt-left", sym=17.0,
           baseline=(0.0, 0.0, 40.0),
           extra={"target_angle_deg": 20, "measured_angle_deg": 20,
                  "measured_end_angle_deg": 14})
    row = tilt_rows(_load(tmp_path))[0]
    assert row.reference_deg == pytest.approx(17.0)
    assert row.drift_deg == pytest.approx(-6.0)
    assert row.expected_deg == pytest.approx(17.0)
    assert row.error_deg == pytest.approx(0.0)
    assert row.magnitude_error_deg == pytest.approx(0.0)


def test_without_an_end_value_the_start_value_is_the_reference(tmp_path):
    _tilt(tmp_path / "s01", "tilt-left-1.csv", "tilt-left", sym=18.0, measured=20,
          target=20)
    row = tilt_rows(_load(tmp_path))[0]
    assert row.measured_end_deg is None
    assert row.drift_deg is None          # 沒記結束值是看不出來，不是 0
    assert row.reference_deg == 20
    assert row.error_deg == pytest.approx(-2.0)


def test_a_slid_segment_is_called_out_in_the_report(tmp_path):
    folder = tmp_path / "s01"
    _write(folder, "tilt-left-1.csv", condition="tilt-left", sym=17.0,
           baseline=(0.0, 0.0, 40.0),
           extra={"target_angle_deg": 20, "measured_angle_deg": 20,
                  "measured_end_angle_deg": 14})
    _tilt(folder, "tilt-right-1.csv", "tilt-right", sym=-20.0, measured=20, target=20)
    text = dict(trip_sections(_load(tmp_path)))["側傾：θ_sym 對量角器"]
    assert "上身滑動" in text and "-6.0°" in text
    assert "只有開始值" in text           # tilt-right 沒記結束值


def test_the_tilt_csv_carries_the_end_value_and_the_drift(tmp_path):
    folder = tmp_path / "s01"
    _write(folder, "tilt-left-1.csv", condition="tilt-left", sym=17.0,
           baseline=(0.0, 0.0, 40.0),
           extra={"target_angle_deg": 20, "measured_angle_deg": 20,
                  "measured_end_angle_deg": 14})
    row = list(csv.DictReader(trip_csvs(_load(tmp_path))["tilt.csv"].splitlines()))[0]
    assert float(row["measured_end_angle_deg"]) == pytest.approx(14.0)
    assert float(row["angle_drift_deg"]) == pytest.approx(-6.0)
    assert float(row["reference_angle_deg"]) == pytest.approx(17.0)


def test_a_wrong_sign_is_called_out(tmp_path):
    _tilt(tmp_path / "s01", "tilt-left-1.csv", "tilt-left", sym=-18.0, measured=20,
          target=20)
    row = tilt_rows(_load(tmp_path))[0]
    assert row.sign_ok is False


def test_tilt_errors_line_up_by_angle_in_one_table(tmp_path):
    """前 5 人另有 10°、30°，要能看誤差會不會隨角度變大。"""
    folder = tmp_path / "s01"
    _tilt(folder, "tilt-left-1.csv", "tilt-left", 19.0, 20, 20)
    _tilt(folder, "tilt-left-2.csv", "tilt-left", 9.5, 10, 10)
    _tilt(folder, "tilt-right-1.csv", "tilt-right", -26.0, 30, 30)
    table = tilt_by_angle(tilt_rows(_load(tmp_path)))
    assert table[("左", 10.0)].mean == pytest.approx(-0.5)
    assert table[("左", 20.0)].mean == pytest.approx(-1.0)
    assert table[("右", 30.0)].mean == pytest.approx(4.0)


def test_a_tilt_without_the_protractor_value_has_no_error(tmp_path):
    _write(tmp_path / "s01", "tilt-left-1.csv", condition="tilt-left", sym=18.0,
           baseline=(0.0, 0.0, 40.0))
    row = tilt_rows(_load(tmp_path))[0]
    assert row.error_deg is None
    assert "沒有量角器實測值" in dict(trip_sections(_load(tmp_path)))["側傾：θ_sym 對量角器"]


# ---- 表4 ---------------------------------------------------------------

def _azimuth_trip(folder, subject="s01"):
    for trial, azimuth in ((1, 30.0), (2, 45.0), (3, 55.0)):
        ratio = math.sin(math.radians(azimuth)) / _SIN45
        _write(folder, f"head-forward-{trial}.csv", subject=subject,
               condition="head-forward", trial=trial, trip="T5", ca=20.0,
               azimuth=azimuth, theta_2d=ratio,
               extra={"azimuth_target_deg": int(azimuth)})


def test_the_tan_ratio_follows_sin_phi_over_sin_45(tmp_path):
    _azimuth_trip(tmp_path / "s01")
    rows = ratio_rows(_load(tmp_path))
    assert [r.target_deg for r in rows] == [30.0, 45.0, 55.0]
    for row in rows:
        assert row.ratio == pytest.approx(row.predicted, abs=1e-3)
        assert row.deviation == pytest.approx(0.0, abs=1e-3)
        assert not row.unstable
    assert rows[0].ratio == pytest.approx(math.sin(math.radians(30)) / _SIN45, abs=1e-3)


def test_only_tagged_segments_go_into_table_4(tmp_path):
    _write(tmp_path / "s01", "head-forward-9.csv", condition="head-forward", ca=20.0,
           theta_2d=1.0)
    assert ratio_rows(_load(tmp_path)) == []


def test_a_small_theta_3d_is_marked_unstable(tmp_path):
    _write(tmp_path / "s01", "head-forward-1.csv", condition="head-forward", ca=2.0,
           theta_2d=1.0, extra={"azimuth_target_deg": 45})
    row = ratio_rows(_load(tmp_path))[0]
    assert row.unstable
    assert ratio_by_group([row], None) == {}


def test_table_4_is_grouped_by_height_layer(tmp_path):
    _azimuth_trip(tmp_path / "s01", "s01")
    _azimuth_trip(tmp_path / "s02", "s02")
    groups_file = tmp_path / "groups.csv"
    groups_file.write_text("subject,group\ns01,男矮\ns02,男高\n", encoding="utf-8")
    groups = read_groups(groups_file)
    grouped = ratio_by_group(ratio_rows(_load(tmp_path)), groups)
    assert {key[0] for key in grouped} == {"男矮", "男高"}
    assert grouped[("男矮", 30.0)][0].n == 1
    text = to_text(collect([tmp_path]), None, groups=groups)
    assert "表4" in text and "男高" in text


def test_the_groups_file_needs_both_columns(tmp_path):
    bad = tmp_path / "g.csv"
    bad.write_text("name,layer\na,b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="subject 與 group"):
        read_groups(bad)


def test_old_files_produce_no_trip_sections(tmp_path):
    _write(tmp_path / "s01", "upright-1.csv", trip=None)
    assert trip_sections(_load(tmp_path)) == []
    assert trip_csvs(_load(tmp_path)) == {}


# ---- 2026-11-18 GPT 審查補的情境 -----------------------------------------

def test_a_pair_is_void_when_the_reference_upright_itself_turned(tmp_path):
    """upright 相對基準轉了 20°，自然坐相對它只差 1°；這一列不能算數。"""
    folder = tmp_path / "s01"
    _write(folder, "natural-1.csv", condition="natural", azimuth=61.0)
    _write(folder, "upright-1.csv", azimuth=60.0, baseline=(0.0, 0.0, 40.0),
           captured_at="2026-11-18T10:02:00")
    row = paired_differences(_load(tmp_path))[0]
    assert row.turned_deg == pytest.approx(1.0)
    assert row.reference_turned_deg == pytest.approx(20.0)
    assert row.differences["θ_CA"].status == VOID
    assert sensitivity([row], "θ_CA").n == 0


def test_a_trip_without_any_baseline_never_voids_for_turning(tmp_path):
    """整趟沒有基準時，即使有 upright，方位角差也不判轉身。"""
    folder = tmp_path / "s01"
    _write(folder, "natural-1.csv", condition="natural", azimuth=30.0)
    _write(folder, "upright-1.csv", azimuth=55.0, captured_at="2026-11-18T10:02:00")
    row = paired_differences(_load(tmp_path))[0]
    assert row.turned_deg is None
    assert row.differences["θ_CA"].status != VOID


def test_the_camera_lost_upright_is_skipped_as_the_reference(tmp_path):
    """掉線的那份 upright 檔尾有 camera_lost，配對改用重量的那一份。"""
    folder = tmp_path / "s01"
    _write(folder, "natural-1.csv", condition="natural", ca=20.0)
    lost = _write(folder, "upright-1.csv", ca=0.0, frames=10,
                  captured_at="2026-11-18T10:02:00")
    with lost.open("a", encoding="utf-8") as handle:
        handle.write("# camera_lost=1\n")
    _write(folder, "upright-2.csv", trial=2, ca=8.0, captured_at="2026-11-18T10:04:00")
    row = paired_differences(_load(tmp_path))[0]
    assert row.reference.path.name == "upright-2.csv"
    assert row.differences["θ_CA"].value == pytest.approx(12.0)


def test_the_ratio_error_survives_a_zero_numerator(tmp_path):
    """mean(tan θ_2D) = 0 時比值是 0，誤差仍然算得出來（σ_r = σ_a / |b|）。"""
    _write(tmp_path / "s01", "head-forward-1.csv", condition="head-forward", ca=20.0,
           theta_2d=0.0, extra={"azimuth_target_deg": 45})
    path = next((tmp_path / "s01").glob("*.csv"))
    # 把 2D 的耳朵 x 改成繞著肩膀晃，平均 tan θ_2D 是 0、但有散佈
    lines = path.read_text(encoding="utf-8").splitlines()
    header_at = next(i for i, l in enumerate(lines) if l.startswith("frame,"))
    columns = lines[header_at].split(",")
    x = columns.index("right_ear_lx_px")
    for k, i in enumerate(range(header_at + 1, len(lines))):
        cells = lines[i].split(",")
        cells[x] = str(500.0 + (-6.0, 1.0, 5.0)[k % 3])   # 三個一組平均為 0，區塊長度 8 切不齊
        lines[i] = ",".join(cells)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    row = ratio_rows(_load(tmp_path))[0]
    assert row.ratio == pytest.approx(0.0, abs=1e-9)
    assert row.ratio_error is not None and row.ratio_error > 0


def test_forward_and_tilt_carry_the_sensitivity_runs(tmp_path):
    folder = tmp_path / "s01"
    _write(folder, "upright-1.csv", shoulder=-100.0, ka=2.0, baseline=(0, 0, 40.0))
    _write(folder, "forward-1.csv", condition="forward", shoulder=-130.0, ka=22.0,
           baseline=(0, 0, 40.0), captured_at="2026-11-18T10:05:00")
    _tilt(folder, "tilt-left-1.csv", "tilt-left", sym=18.0, measured=20, target=20)
    sections = dict(trip_sections(_load(tmp_path)))
    assert "軀幹前傾：跨受試者平均與敏感度分析" in sections
    assert "≤15%" in sections["側傾：θ_sym 對量角器"]


# ---- 側傾：前作 2D 肩線傾斜角 -------------------------------------------

def _tilt_2d_trip(folder, *, up2d, up3d, t2d, t3d, measured, azimuth=45.0,
                  condition="tilt-left", subject="s01", with_pixels=True):
    _write(folder, "upright-1.csv", subject=subject, sym=up3d, azimuth=azimuth,
           baseline=(0.0, 0.0, azimuth), sym_2d=up2d if with_pixels else None)
    _write(folder, f"{condition}-1.csv", subject=subject, condition=condition,
           sym=t3d, azimuth=azimuth, baseline=(0.0, 0.0, azimuth),
           captured_at="2026-11-18T10:05:00",
           extra={"target_angle_deg": 20, "measured_angle_deg": measured},
           sym_2d=t2d if with_pixels else None)


def test_the_2d_tilt_error_subtracts_the_same_trip_upright_and_keeps_the_sign(tmp_path):
    # 端正坐 2D 有 +5.7° 的零點；右傾 2D 原始 −20.6°。減端正坐後是 −26.3°。
    # 先取絕對值再減會得到 +14.9°，這是要防的錯。
    _tilt_2d_trip(tmp_path / "s01", up2d=5.7, up3d=6.4, t2d=-20.6, t3d=-14.4,
                  measured=20, condition="tilt-right")
    row = tilt_rows(_load(tmp_path))[0]
    assert row.sym_2d_deg == pytest.approx(-26.3, abs=0.01)
    assert row.sym_3d_same_ref_deg == pytest.approx(-20.8, abs=0.01)
    assert row.error_2d_deg == pytest.approx(-6.3, abs=0.01)
    assert row.error_3d_same_ref_deg == pytest.approx(-0.8, abs=0.01)
    assert row.magnitude_error_2d_deg == pytest.approx(6.3, abs=0.01)
    assert row.azimuth_deg == pytest.approx(45.0)


def test_the_2d_amplification_is_the_tan_ratio_and_is_compared_with_1_over_cos(tmp_path):
    t3 = 15.0
    t2 = math.degrees(math.atan(math.tan(math.radians(t3)) / math.cos(math.radians(45.0))))
    _tilt_2d_trip(tmp_path / "s01", up2d=0.0, up3d=0.0, t2d=t2, t3d=t3, measured=15)
    row = tilt_rows(_load(tmp_path))[0]
    assert row.amplification == pytest.approx(1.0 / math.cos(math.radians(45.0)), rel=1e-3)
    assert row.predicted_amplification == pytest.approx(1.4142, rel=1e-3)


def test_a_small_3d_tilt_does_not_report_an_amplification(tmp_path):
    _tilt_2d_trip(tmp_path / "s01", up2d=0.0, up3d=0.0, t2d=2.0, t3d=1.0, measured=15)
    assert tilt_rows(_load(tmp_path))[0].amplification is None


def test_no_pixels_or_no_upright_means_no_2d_not_a_fallback_to_raw(tmp_path):
    _tilt_2d_trip(tmp_path / "old", up2d=0, up3d=0, t2d=0, t3d=-15, measured=15,
                  with_pixels=False, subject="old")
    _write(tmp_path / "lone", "tilt-left-1.csv", subject="lone", condition="tilt-left",
           sym=15.0, baseline=(0.0, 0.0, 40.0), sym_2d=20.0,
           extra={"target_angle_deg": 20, "measured_angle_deg": 15})
    summaries = _load(tmp_path)
    rows = {r.summary.subject_key: r for r in tilt_rows(summaries)}
    assert rows["old"].sym_2d_deg is None and rows["old"].error_2d_deg is None
    assert rows["lone"].sym_2d_deg is None and rows["lone"].error_2d_deg is None
    assert "前作 2D 肩線傾斜角" not in dict(trip_sections(summaries))["側傾：θ_sym 對量角器"]


def test_the_2d_by_angle_table_counts_people_not_frames(tmp_path):
    for subject, t2d in (("s01", 20.0), ("s02", 30.0)):
        _tilt_2d_trip(tmp_path / subject, up2d=0.0, up3d=0.0, t2d=t2d, t3d=15.0,
                      measured=15, subject=subject)
    table = tilt_by_angle(tilt_rows(_load(tmp_path)), attribute="error_2d_deg")
    group = table[("左", 20.0)]
    assert group.n == 2 and group.mean == pytest.approx(10.0)


def test_the_tilt_report_and_csv_carry_the_2d_columns(tmp_path):
    _tilt_2d_trip(tmp_path / "s01", up2d=0.0, up3d=0.0, t2d=20.0, t3d=15.0, measured=15)
    summaries = _load(tmp_path)
    text = dict(trip_sections(summaries))["側傾：θ_sym 對量角器"]
    assert "前作 2D 肩線傾斜角對 3D θ_sym" in text
    assert "依目標角度：2D 與 3D 同法比較" in text
    row = list(csv.DictReader(trip_csvs(summaries)["tilt.csv"].splitlines()))[0]
    assert float(row["error_2d_deg"]) == pytest.approx(5.0, abs=0.01)
    assert float(row["amplification_2d"]) > 1.0
    assert float(row["predicted_amplification_2d"]) == pytest.approx(1.4142, rel=1e-3)
