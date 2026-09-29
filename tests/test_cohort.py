"""跨檔案彙整。

最重要的一條：**統計單位是受試者，不是幀。** 五個人各量 500 幀不是 2500 個
樣本，是 5 個。這個專案已經在單一 session 的層級犯過一次幀數當分母的錯
（`std/√N` 低估 2.6 倍），跨受試者犯同樣的錯會更嚴重。
"""
import pytest

from geometry.cohort import UNLABELLED, Cohort, Separation, collect
from geometry.cohort_report import (separation_markdown, session_rows,
                                    sessions_markdown, to_csv, to_markdown)

_HEADER = (
    "frame,elapsed_s,usable,reject_reason,theta_ca_deg,theta_sym_deg,"
    "theta_ca_corrected_deg,theta_sym_corrected_deg,theta_ca_mean_deg,"
    "theta_sym_mean_deg,theta_ca_standard_error_deg,distance_mm,azimuth_deg,"
    "theta_ca_precision_deg,shared_keypoints,max_vertical_disparity_px,"
    "angle_max_vertical_disparity_px,posture"
)


def _session(tmp_path, subject, condition, ca, frames=60, trial=1,
             rejected=0, posture="正常"):
    """寫一份帶中繼資料的 CSV。ca 是扣除基準之後的角度。"""
    folder = tmp_path / subject
    folder.mkdir(parents=True, exist_ok=True)
    lines = [f"# subject={subject}"]
    if condition is not None:
        lines.append(f"# condition={condition}")
        lines.append(f"# trial={trial}")
    lines.append(_HEADER)
    for i in range(frames):
        value = ca + (0.5 if i % 2 else -0.5)      # 一點散佈，避免標準差為零
        lines.append(f"{i + 1},{i * 0.2:.3f},1,,{value},1.0,{value},1.0,"
                     f"{value},1.0,0.5,660,22,6.5,13,15.0,4.0,{posture}")
    for i in range(rejected):
        lines.append(f"{frames + i + 1},{(frames + i) * 0.2:.3f},0,"
                     f"right_shoulder 的垂直視差 9.0px，左右配對錯了,"
                     f",,,,,,,660,22,6.5,9,20.0,9.0,")
    name = f"{condition or 'x'}-{trial}.csv"
    path = folder / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _cohort(tmp_path, *specs) -> Cohort:
    for spec in specs:
        _session(tmp_path, **spec)
    return collect([tmp_path])


# ---- 分組 --------------------------------------------------------------

def test_sessions_group_by_subject_and_condition(tmp_path):
    cohort = _cohort(
        tmp_path,
        dict(subject="A", condition="upright", ca=0.0),
        dict(subject="A", condition="forward", ca=20.0),
        dict(subject="B", condition="upright", ca=2.0),
    )
    assert cohort.subjects == ["A", "B"]
    assert cohort.conditions == ["forward", "upright"]
    assert cohort.result("A", "forward").theta_ca_deg == pytest.approx(20.0)
    assert cohort.result("B", "forward").theta_ca_deg is None


def test_files_without_a_condition_are_kept_and_named(tmp_path):
    """2026-09 的記錄寫在 condition 欄存在之前，而那幾份最值得分析。

    不讓它們消失，因為消失的資料沒有人會發現。
    """
    cohort = _cohort(tmp_path, dict(subject="A", condition=None, ca=5.0))
    assert cohort.conditions == [UNLABELLED]
    assert cohort.result("A", UNLABELLED).theta_ca_deg == pytest.approx(5.0)


def test_the_unlabelled_group_sorts_last(tmp_path):
    cohort = _cohort(
        tmp_path,
        dict(subject="A", condition=None, ca=5.0),
        dict(subject="A", condition="upright", ca=1.0),
    )
    assert cohort.conditions[-1] == UNLABELLED


# ---- 統計單位 ----------------------------------------------------------

def test_repeated_trials_average_as_trials_not_as_pooled_frames(tmp_path):
    """一次量 300 幀、一次量 60 幀，後者的權重不該只有五分之一。

    兩次都是獨立的一次嘗試，代表性相同。
    """
    _session(tmp_path, "A", "upright", ca=0.0, frames=300, trial=1)
    _session(tmp_path, "A", "upright", ca=10.0, frames=60, trial=2)
    cohort = collect([tmp_path])
    # 幀數加權會是 (0*300 + 10*60)/360 = 1.67，各次等權是 5.0
    assert cohort.result("A", "upright").theta_ca_deg == pytest.approx(5.0, abs=0.1)


def test_the_cross_subject_error_divides_by_people_not_frames(tmp_path):
    """五個人各量 500 幀是 5 個樣本，不是 2500 個。"""
    specs = []
    for i, subject in enumerate("ABCDE"):
        specs.append(dict(subject=subject, condition="upright", ca=0.0, frames=500))
        specs.append(dict(subject=subject, condition="forward",
                          ca=20.0 + i * 2.0, frames=500))
    cohort = _cohort(tmp_path, *specs)
    separation = Separation("upright", "forward",
                            cohort.differences("upright", "forward"))
    assert separation.subjects == 5
    assert separation.mean_deg == pytest.approx(24.0, abs=0.1)
    # 差距是 20,22,24,26,28，樣本標準差 √10≈3.16，除以 √5 得 1.41
    assert separation.error_deg == pytest.approx(1.41, abs=0.05)


def test_a_single_subject_reports_no_error_rather_than_zero(tmp_path):
    """一個樣本算不出散佈。回傳 0 會被讀成「完全沒有變異」。"""
    cohort = _cohort(
        tmp_path,
        dict(subject="A", condition="upright", ca=0.0),
        dict(subject="A", condition="forward", ca=20.0),
    )
    separation = Separation("upright", "forward",
                            cohort.differences("upright", "forward"))
    assert separation.mean_deg == pytest.approx(20.0)
    assert separation.error_deg is None
    assert separation.comparison().sigma is None
    assert "算不出跨受試者的誤差" in separation_markdown(separation, cohort)


def test_only_subjects_with_both_conditions_contribute(tmp_path):
    cohort = _cohort(
        tmp_path,
        dict(subject="A", condition="upright", ca=0.0),
        dict(subject="A", condition="forward", ca=20.0),
        dict(subject="B", condition="upright", ca=1.0),
    )
    assert set(cohort.differences("upright", "forward")) == {"A"}


# ---- 略過率 ------------------------------------------------------------

def test_a_high_rejection_session_is_shown_but_not_counted(tmp_path):
    """留下來的幀代表不了整段姿勢，但把它藏起來更糟。"""
    _session(tmp_path, "A", "upright", ca=0.0, frames=60, trial=1)
    _session(tmp_path, "A", "upright", ca=30.0, frames=60, rejected=40, trial=2)
    cohort = collect([tmp_path])

    result = cohort.result("A", "upright")
    assert len(result.sessions) == 2
    assert len(result.usable_sessions) == 1
    assert result.theta_ca_deg == pytest.approx(0.0, abs=0.1)
    table = sessions_markdown(session_rows(cohort))
    assert "⚠" in table
    assert table.count("| A |") == 2, "兩段都要列出來"


def test_a_clean_session_is_not_flagged(tmp_path):
    """圖例也不該出現。沒有東西被標卻印一行解釋，讀的人會回頭找那個符號。"""
    cohort = _cohort(tmp_path, dict(subject="A", condition="upright",
                                    ca=0.0, frames=60, rejected=2))
    assert "⚠" not in sessions_markdown(session_rows(cohort))


# ---- 判定 --------------------------------------------------------------

def test_the_over_share_is_pooled_over_usable_sessions(tmp_path):
    cohort = _cohort(
        tmp_path,
        dict(subject="A", condition="upright", ca=0.0, posture="正常"),
        dict(subject="A", condition="forward", ca=20.0, posture="超標"),
    )
    assert cohort.result("A", "upright").over_share == pytest.approx(0.0)
    assert cohort.result("A", "forward").over_share == pytest.approx(1.0)


# ---- 讀檔 --------------------------------------------------------------

def test_a_directory_is_searched_recursively(tmp_path):
    _session(tmp_path, "A", "upright", ca=0.0)
    _session(tmp_path, "B", "upright", ca=1.0)
    assert len(collect([tmp_path]).sessions) == 2


def test_non_csv_files_are_ignored(tmp_path):
    _session(tmp_path, "A", "upright", ca=0.0)
    (tmp_path / "A" / "notes.txt").write_text("x", encoding="utf-8")
    (tmp_path / "A" / "shot.png").write_bytes(b"x")
    assert len(collect([tmp_path]).sessions) == 1


def test_an_empty_csv_is_skipped_instead_of_stopping_the_batch(tmp_path):
    """量測沒跑起來就會留下那種檔案。整批彙整不該因此中斷。"""
    _session(tmp_path, "A", "upright", ca=0.0)
    (tmp_path / "A" / "broken-1.csv").write_text(_HEADER + "\n", encoding="utf-8")
    assert len(collect([tmp_path]).sessions) == 1


def test_nothing_readable_gives_an_empty_cohort(tmp_path):
    assert collect([tmp_path]).sessions == []


# ---- 說清楚為什麼算不出來 ----------------------------------------------

def test_a_missing_condition_name_is_named(tmp_path):
    """名字打錯與根本沒量過要分得出來。"""
    cohort = _cohort(
        tmp_path,
        dict(subject="A", condition="upright", ca=0.0),
        dict(subject="A", condition="slouch", ca=20.0),
    )
    separation = Separation("upright", "forward",
                            cohort.differences("upright", "forward"))
    text = separation_markdown(separation, cohort)
    assert "forward" in text
    assert "slouch" in text


def test_unlabelled_files_are_explained_rather_than_silently_dropped(tmp_path):
    cohort = _cohort(tmp_path, dict(subject="A", condition=None, ca=5.0))
    separation = Separation("upright", "forward",
                            cohort.differences("upright", "forward"))
    text = separation_markdown(separation, cohort)
    assert "沒有記錄姿勢條件" in text
    assert "事後補標會變成猜" in text


# ---- 輸出 --------------------------------------------------------------

def test_the_csv_carries_every_number_the_markdown_shows(tmp_path):
    cohort = _cohort(
        tmp_path,
        dict(subject="A", condition="upright", ca=0.0),
        dict(subject="A", condition="forward", ca=20.0),
    )
    text = to_csv(session_rows(cohort))
    assert text.splitlines()[0].startswith("subject,condition,trial,file")
    assert len(text.strip().splitlines()) == 3
    assert "upright" in text and "forward" in text


def test_a_missing_value_is_an_empty_cell_not_a_zero(tmp_path):
    """0 是合法的角度值，拿它表示缺失會讓下游算錯。"""
    cohort = _cohort(tmp_path, dict(subject="A", condition=None, ca=5.0))
    body = to_csv(session_rows(cohort)).splitlines()[1]
    assert ",," in body, "trial 沒有值，該是空格"


def test_the_whole_report_renders_without_a_separation(tmp_path):
    cohort = _cohort(tmp_path, dict(subject="A", condition="upright", ca=0.0))
    assert "量測結果彙整" in to_markdown(cohort, None)


def test_the_report_states_that_the_error_is_across_subjects(tmp_path):
    """只報「幾個標準誤差」會高估可靠度，讀的人要知道分母是什麼。"""
    specs = []
    for i, subject in enumerate("ABC"):
        specs.append(dict(subject=subject, condition="upright", ca=0.0))
        specs.append(dict(subject=subject, condition="forward", ca=20.0 + i))
    cohort = _cohort(tmp_path, *specs)
    separation = Separation("upright", "forward",
                            cohort.differences("upright", "forward"))
    text = separation_markdown(separation, cohort)
    assert "分母是人數，不是幀數" in text
    assert "差距 / 最大單幀散佈" in text


def test_both_reports_narrate_the_difference_in_the_same_words(tmp_path):
    """單一 session 的報表與跨受試者的報表共用 describe()。

    先前兩邊各自拼這一句，於是可以一邊帶誤差、一邊不帶，而讀的人分不出
    是資料不同還是排版不同。
    """
    import numpy as np

    from geometry.session_analysis import AngleSummary, compare

    specs = []
    for i, subject in enumerate("ABC"):
        specs.append(dict(subject=subject, condition="upright", ca=0.0))
        specs.append(dict(subject=subject, condition="forward", ca=20.0 + i))
    cohort = _cohort(tmp_path, *specs)
    separation = Separation("upright", "forward",
                            cohort.differences("upright", "forward"))
    across = separation.comparison(spread=5.0)

    within = compare(AngleSummary("a", np.zeros(50) + np.arange(50) * 0.01),
                     AngleSummary("b", np.full(50, 21.0) + np.arange(50) * 0.01),
                     spread=5.0)
    # 同一個型別、同一個方法，所以格式必然一致
    assert type(across) is type(within)
    for text in (across.describe(), within.describe()):
        assert "±" in text and "個標準誤差" in text


def test_the_cohort_report_still_reads_the_same(tmp_path):
    """抽出值物件不該改變輸出。"""
    specs = []
    for i, subject in enumerate("ABC"):
        specs.append(dict(subject=subject, condition="upright", ca=0.0))
        specs.append(dict(subject=subject, condition="forward", ca=20.0 + i))
    cohort = _cohort(tmp_path, *specs)
    separation = Separation("upright", "forward",
                            cohort.differences("upright", "forward"))
    text = separation_markdown(separation, cohort)
    assert "平均差距 +21.00°" in text
    assert "n = 3" in text
    assert "分母是人數，不是幀數" in text
    assert "差距 / 最大單幀散佈" in text
