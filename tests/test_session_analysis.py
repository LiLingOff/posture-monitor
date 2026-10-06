"""分析逐幀 CSV。

重點在**舊檔案也要讀得動**。`posture` 與 `angle_max_vertical_disparity_px`
是後來才加的欄，2026-09-24 那兩份記錄裡沒有，而它們的分析價值最高。
"""
import numpy as np
import pytest

from geometry.session_analysis import (UNLABELLED, AngleSummary, analyse_session,
                                       compare, correlation, read_rows)
from geometry.session_report import format_report, format_session, segment_table

_HEADER = (
    "frame,elapsed_s,usable,reject_reason,theta_ca_deg,theta_sym_deg,"
    "theta_ca_corrected_deg,theta_sym_corrected_deg,theta_ca_mean_deg,"
    "theta_sym_mean_deg,theta_ca_standard_error_deg,distance_mm,azimuth_deg,"
    "theta_ca_precision_deg,shared_keypoints,max_vertical_disparity_px"
)
_NEW_COLUMNS = ",angle_max_vertical_disparity_px,posture"


def _write(tmp_path, rows, subject="A", header=_HEADER, name="s.csv"):
    path = tmp_path / name
    lines = ([f"# subject={subject}"] if subject else []) + [header] + rows
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _usable_row(frame, ca, sym=1.0, baseline=0.0, distance=660.0, azimuth=20.0,
                extra=""):
    return (f"{frame},{frame * 0.2:.3f},1,,{ca},{sym},{ca - baseline},{sym - baseline},"
            f"{ca},{sym},0.5,{distance},{azimuth},7.7,12,15.0{extra}")


def _old_format_session(tmp_path, count=120, baseline=-4.0, ca=lambda i: 8.0):
    rows = [_usable_row(i + 1, round(ca(i), 3), baseline=baseline) for i in range(count)]
    return _write(tmp_path, rows)


def test_reads_the_metadata_block(tmp_path):
    rows, meta = read_rows(_old_format_session(tmp_path))
    assert meta["subject"] == "A"
    assert len(rows) == 120


def test_unknown_metadata_keys_are_kept(tmp_path):
    """日後多記一項不該要求同步改讀取端。"""
    rows = [_usable_row(1, 8.0), _usable_row(2, 9.0)]
    path = _write(tmp_path, rows, subject=None)
    text = path.read_text(encoding="utf-8")
    path.write_text("# subject=B\n# something_new=42\n" + text, encoding="utf-8")
    _, meta = read_rows(path)
    assert meta["subject"] == "B"
    assert meta["something_new"] == "42"


def test_a_file_without_a_condition_says_so_rather_than_guessing(tmp_path):
    """2026-09-24 與 09-29 那幾份沒有 condition，而它們的分析價值最高。"""
    summary = analyse_session(_old_format_session(tmp_path))
    assert summary.subject == "A"
    assert summary.condition is None
    assert summary.trial is None
    # 分組時要歸到「未標註」而不是被濾掉
    assert summary.condition_key == UNLABELLED


def test_a_file_without_a_subject_still_groups_somewhere(tmp_path):
    rows = [_usable_row(i + 1, 8.0) for i in range(5)]
    summary = analyse_session(_write(tmp_path, rows, subject=None, name="0929-x.csv"))
    assert summary.subject is None
    assert summary.subject_key == "?"


def test_the_angle_accessor_prefers_the_baseline_corrected_values(tmp_path):
    """判定看的是扣除基準之後的角度，統計要用同一個。"""
    summary = analyse_session(_old_format_session(tmp_path, baseline=-4.0))
    assert summary.angle("θ_CA") is summary.angles["θ_CA 扣基準"]


def test_the_angle_accessor_falls_back_to_raw(tmp_path):
    """沒給基準的那幾份也要算得出統計，不能回傳 None。"""
    summary = analyse_session(_old_format_session(tmp_path, baseline=0.0))
    del summary.angles["θ_CA 扣基準"]
    assert summary.angle("θ_CA") is summary.angles["θ_CA 原始"]


def test_an_unknown_angle_name_is_none_not_a_crash(tmp_path):
    summary = analyse_session(_old_format_session(tmp_path))
    assert summary.angle("不存在的角度") is None


def test_a_file_with_only_a_header_says_so_rather_than_dividing_by_zero(tmp_path):
    path = _write(tmp_path, [])
    with pytest.raises(ValueError, match="只有標題列"):
        read_rows(path)


def test_an_empty_file_says_so(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="沒有任何資料列"):
        read_rows(path)


def test_it_reads_a_csv_written_before_the_newer_columns_existed(tmp_path):
    """2026-09-24 那兩份記錄沒有 posture 與 angle_max 欄，卻最需要分析。"""
    summary = analyse_session(_old_format_session(tmp_path))
    assert summary.frames == 120
    assert summary.usable == 120
    assert "垂直視差 角度用 px" not in summary.columns
    assert summary.judgement is not None
    assert summary.judgement.replayed, "沒有 posture 欄時要重播"


def test_it_prefers_the_recorded_verdict_over_replaying_it(tmp_path):
    """CSV 有 posture 欄時，當時用什麼參數判的就該報什麼結果。"""
    rows = [_usable_row(i + 1, 25.0, extra=",2.0,超標") for i in range(40)]
    path = _write(tmp_path, rows, header=_HEADER + _NEW_COLUMNS)
    summary = analyse_session(path)
    assert not summary.judgement.replayed
    assert summary.judgement.counts["超標"] == 40


def test_replay_can_be_forced_to_compare_parameters(tmp_path):
    rows = [_usable_row(i + 1, 25.0, extra=",2.0,正常") for i in range(40)]
    path = _write(tmp_path, rows, header=_HEADER + _NEW_COLUMNS)
    forced = analyse_session(path, window=10, force_replay=True)
    assert forced.judgement.replayed
    assert forced.judgement.counts["超標"] > 0, "記錄裡寫正常，重播應該算出超標"


def test_rejection_reasons_are_grouped_by_kind_not_by_number(tmp_path):
    """「深度 54mm」與「深度 61mm」是同一件事。分開計數會蓋掉真正的訊息。"""
    rows = [
        f"{i},0.1,0,深度 {-100 - i}mm 落在合理範圍外,,,,,,,,,,,3,20.0"
        for i in range(1, 8)
    ]
    summary = analyse_session(_write(tmp_path, rows))
    assert len(summary.reasons) == 1
    assert summary.reasons.most_common(1)[0][1] == 7
    # 顯示時要拿原句，不然使用者看到的是「深度 Nmm」
    assert "深度 -101mm" in next(iter(summary.reason_examples.values()))


def test_all_frames_rejected_does_not_crash_and_reports_no_angles(tmp_path):
    rows = [f"{i},0.1,0,偵測不到人,,,,,,,,,,,0," for i in range(1, 6)]
    summary = analyse_session(_write(tmp_path, rows))
    assert summary.usable == 0
    assert summary.rejection_rate == 1.0
    assert summary.angles == {}
    assert "沒有可用的角度資料" in format_session(summary)


def test_a_session_without_a_baseline_is_flagged(tmp_path):
    """判定門檻套在原始角度上會因人而異，所以這件事一定要講出來。"""
    summary = analyse_session(_old_format_session(tmp_path, baseline=0.0))
    assert not summary.has_baseline
    assert "沒有扣除個人基準" in format_session(summary)


def test_a_session_with_a_baseline_is_not_flagged(tmp_path):
    summary = analyse_session(_old_format_session(tmp_path, baseline=-4.0))
    assert summary.has_baseline
    assert "沒有扣除個人基準" not in format_session(summary)


def test_it_detects_that_neighbouring_frames_are_correlated(tmp_path):
    """這是換掉 std/√N 的依據，分析工具要看得出來。"""
    rng = np.random.default_rng(0)
    value = 0.0
    values = []
    for step in rng.normal(0.0, 3.0, 200):
        value = 0.9 * value + step
        values.append(value)
    drifting = analyse_session(
        _old_format_session(tmp_path, count=200, ca=lambda i: values[i])
    )
    angle = drifting.angles["θ_CA 原始"]
    assert angle.autocorrelation > 0.7
    assert angle.is_correlated
    assert angle.standard_error > 2.0 * angle.naive_standard_error
    assert "std/√N 不適用" in format_session(drifting)


def test_independent_frames_are_not_flagged_as_correlated(tmp_path):
    rng = np.random.default_rng(1)
    noise = rng.normal(8.0, 5.0, 200)
    summary = analyse_session(
        _old_format_session(tmp_path, count=200, ca=lambda i: noise[i])
    )
    assert not summary.angles["θ_CA 原始"].is_correlated
    assert "std/√N 不適用" not in format_session(summary)


def test_it_reports_whether_the_angle_still_depends_on_azimuth(tmp_path):
    """解剖平面定得對的話，角度與方位角該無關。這是該主張的檢查。"""
    rows = [_usable_row(i + 1, 8.0 + i * 0.5, azimuth=10.0 + i * 0.5)
            for i in range(60)]
    summary = analyse_session(_write(tmp_path, rows))
    assert "相關性偏高" in format_session(summary)


def test_correlation_needs_matching_lengths_and_some_variation():
    assert correlation(np.arange(10.0), np.arange(5.0)) is None
    assert correlation(np.ones(10), np.arange(10.0)) is None
    assert correlation(np.arange(10.0), np.arange(10.0)) == pytest.approx(1.0)


def test_comparing_two_sessions_reports_the_difference_and_its_significance():
    quiet = AngleSummary("a", np.full(100, 8.0) + np.random.default_rng(2).normal(0, 1, 100))
    lean = AngleSummary("b", np.full(100, 26.0) + np.random.default_rng(3).normal(0, 1, 100))
    comparison = compare(quiet, lean)
    assert comparison.difference == pytest.approx(18.0, abs=0.5)
    assert comparison.error is not None
    assert comparison.sigma > 10
    # 單幀散佈那個比值要一起給。只報 sigma 會高估可靠度：這組資料的
    # sigma 是 152，而差距只有單幀散佈的 17 倍。
    assert comparison.sigma > 100
    assert comparison.spread_ratio == pytest.approx(17.0, abs=1.0)


def test_comparison_appears_only_when_there_are_two_sessions(tmp_path):
    one = analyse_session(_old_format_session(tmp_path))
    assert "兩段對照" not in format_report([one])
    two = analyse_session(_old_format_session(tmp_path, ca=lambda i: 26.0))
    assert "兩段對照" in format_report([one, two])


def test_the_segment_table_shows_which_way_the_posture_drifted(tmp_path):
    summary = analyse_session(
        _old_format_session(tmp_path, count=120, ca=lambda i: 5.0 + i * 0.1)
    )
    table = segment_table(summary, segments=4)
    assert "頭尾差距" in table
    assert "+11" in table or "+10" in table


def test_the_segment_table_is_skipped_when_there_is_too_little_data(tmp_path):
    summary = analyse_session(_old_format_session(tmp_path, count=5))
    assert segment_table(summary, segments=6) == ""


def test_a_flat_line_reports_zero_error_rather_than_crashing(tmp_path):
    """完全不動的合成資料會讓散佈為 0，除法與相關係數都要撐得住。"""
    summary = analyse_session(_old_format_session(tmp_path, ca=lambda i: 8.0))
    assert summary.angles["θ_CA 原始"].standard_error == pytest.approx(0.0, abs=1e-9)
    assert summary.angles["θ_CA 原始"].autocorrelation is None
    assert format_session(summary)


def test_missing_values_are_skipped_not_read_as_zero(tmp_path):
    """0 是合法的角度值，把空格當成 0 會把平均拉向 0。"""
    rows = [_usable_row(1, 10.0), "2,0.4,1,,,,,,,,,,,,12,15.0", _usable_row(3, 10.0)]
    summary = analyse_session(_write(tmp_path, rows))
    assert summary.angles["θ_CA 原始"].count == 2
    assert summary.angles["θ_CA 原始"].mean == pytest.approx(10.0)


def test_the_report_survives_a_file_with_no_subject_comment(tmp_path):
    rows = [_usable_row(i + 1, 8.0 + i) for i in range(10)]
    summary = analyse_session(_write(tmp_path, rows, subject=None))
    assert summary.subject is None
    assert format_session(summary)


def _azimuth_row(frame, azimuth, usable=True, reason="", shared=12):
    flag = 1 if usable else 0
    ca = "8.0" if usable else ""
    return (f"{frame},{frame * 0.2:.3f},{flag},{reason},{ca},1.0,{ca},1.0,"
            f"{ca},1.0,0.5,660.0,{azimuth},7.7,{shared},15.0")


def test_rejection_rate_is_broken_down_by_azimuth(tmp_path):
    """方位角的上限不在幾何而在遮擋，所以只能從略過率實測。"""
    rows = []
    frame = 0
    for azimuth, bad in ((25.0, 0), (35.0, 2), (45.0, 12), (55.0, 16)):
        for i in range(20):
            frame += 1
            rows.append(_azimuth_row(
                frame, azimuth, usable=i >= bad,
                reason="right_shoulder 的垂直視差 8.9px，左右配對錯了" if i < bad else "",
            ))
    summary = analyse_session(_write(tmp_path, rows))
    bins = {int(b.low): b for b in summary.azimuth_bins}
    assert bins[20].rejection_rate == 0.0
    assert bins[50].rejection_rate == pytest.approx(0.8)
    assert bins[50].shoulder_rejected == 16
    assert "方位角上限的證據" in format_session(summary)


def test_a_fixed_azimuth_produces_no_breakdown(tmp_path):
    """定點量測分箱沒有意義，報一張只有一列的表只會誤導。"""
    rows = [_azimuth_row(i + 1, 20.0 + (i % 3)) for i in range(60)]
    summary = analyse_session(_write(tmp_path, rows))
    assert summary.azimuth_bins == []
    assert "方位角與遮擋" not in format_session(summary)


def test_bins_with_too_few_frames_are_left_out(tmp_path):
    """幀數太少時比例本身不可信，寧可不報。"""
    rows = [_azimuth_row(i + 1, 25.0) for i in range(40)]
    rows += [_azimuth_row(100 + i, 55.0) for i in range(3)]
    summary = analyse_session(_write(tmp_path, rows))
    assert [int(b.low) for b in summary.azimuth_bins] == [20]


def test_only_shoulder_failures_count_towards_occlusion(tmp_path):
    """手腕配錯與方位角無關，混進來會讓遮擋看起來比實際嚴重。"""
    rows = []
    for i in range(40):
        rows.append(_azimuth_row(
            i + 1, 25.0, usable=i >= 10,
            reason="right_wrist 的垂直視差 30px，左右配對錯了" if i < 10 else "",
        ))
    for i in range(40):
        rows.append(_azimuth_row(100 + i, 55.0))
    summary = analyse_session(_write(tmp_path, rows))
    low = next(b for b in summary.azimuth_bins if b.low == 20.0)
    assert low.rejected == 10
    assert low.shoulder_rejected == 0


def test_the_azimuth_story_is_not_told_backwards(tmp_path):
    """略過最多的那一箱不是方位角最大的時候，遮擋就解釋不了。

    2026-09-30 實機：10~20° 略過 7%，20~30° 略過 3%，也就是角度愈大愈乾淨。
    先前這段敘述不分方向，一律寫「遠側肩膀開始被擋住。這是方位角上限」，
    於是講出與數字相反的結論。
    """
    rows = []
    frame = 0
    for azimuth, bad in ((15.0, 14), (35.0, 1)):
        for i in range(40):
            frame += 1
            rows.append(_azimuth_row(
                frame, azimuth, usable=i >= bad,
                reason="right_shoulder 的垂直視差 43.3px，左右配對錯了" if i < bad else "",
            ))
    text = format_session(analyse_session(_write(tmp_path, rows)))
    assert "遮擋解釋不了這個順序" in text
    assert "方位角上限的證據" not in text


def test_the_azimuth_story_still_names_occlusion_when_it_fits(tmp_path):
    rows = []
    frame = 0
    for azimuth, bad in ((15.0, 1), (35.0, 14)):
        for i in range(40):
            frame += 1
            rows.append(_azimuth_row(
                frame, azimuth, usable=i >= bad,
                reason="right_shoulder 的垂直視差 43.3px，左右配對錯了" if i < bad else "",
            ))
    text = format_session(analyse_session(_write(tmp_path, rows)))
    assert "方位角上限的證據" in text


def test_frames_with_no_azimuth_are_accounted_for(tmp_path):
    """量不到方位角的幀不在任何一箱裡，而它們正是最容易被略過的那些。

    2026-09-30 實機：153 幀裡有 25 幀不在表上，於是表上寫 7% 與 3%，
    而整段是 20.9%，兩者差得莫名其妙。
    """
    rows = [_azimuth_row(i + 1, 15.0) for i in range(40)]
    rows += [_azimuth_row(100 + i, 35.0) for i in range(40)]
    # 雙肩沒同時偵測到，方位角欄位是空的
    rows += [f"{200 + i},0.5,0,左右只有 3 個共同關鍵點,,,,,,,,660.0,,7.7,3,20.0"
             for i in range(25)]
    text = format_session(analyse_session(_write(tmp_path, rows)))
    assert "另有 25 幀量不到方位角" in text
    assert "整段的略過率是 24%" in text
