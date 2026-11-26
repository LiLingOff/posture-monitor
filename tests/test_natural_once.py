"""自然坐只算第一次（2026-11-26）。

411363 重量趟 1 時又量了一次自然坐，那時受試者已經知道流程、聽過「請坐正」。
run-plan 因此不再重量它，分析也只取每人（每個桌面條件）第一次。
"""
from __future__ import annotations

from pathlib import Path

from geometry.cohort import collect
from geometry.cohort_report import session_rows, to_text
from geometry.run_plan import ExistingSegment, load_plan, without_measured_natural
from geometry.trip_report import trip_sections
from geometry.trips import REPEATED_NATURAL, paired_differences, repeated_naturals, sensitivity

from test_run_plan import PLAN, _Studio, _patch, _plan_args, _write_segment, posture
from test_trips import _load, _write

BASE = {"desk_level": "base", "lift_cm": 0, "desk_offset_cm": 5}


def _natural_run(folder, trial, trip, ca, day, extra=None):
    """一趟裡的自然坐加同趟端正坐（端正坐固定 0°）。"""
    _write(folder, f"natural-{trial}.csv", condition="natural", trial=trial, trip=trip,
           ca=ca, extra={**BASE, **(extra or {})}, captured_at=f"2026-11-{day}T10:00:00")
    _write(folder, f"upright-{trial}.csv", condition="upright", trial=trial, trip=trip,
           ca=0.0, baseline=(0.0, 0.0, 40.0), captured_at=f"2026-11-{day}T10:02:00")


def test_only_the_first_natural_of_a_subject_counts(tmp_path):
    folder = tmp_path / "s01"
    _natural_run(folder, 1, "T1", ca=12.0, day=24)
    _natural_run(folder, 2, "T2", ca=4.0, day=26)
    summaries = _load(tmp_path)
    assert {p.name for p in repeated_naturals(summaries)} == {"natural-2.csv"}
    rows = paired_differences(summaries)
    assert [r.paired for r in rows] == [True, False]
    assert rows[1].reason == REPEATED_NATURAL
    assert sensitivity(rows, "θ_CA").mean == 12.0           # 不是 (12+4)/2


def test_first_means_earliest_recording_not_file_order(tmp_path):
    folder = tmp_path / "s01"
    _natural_run(folder, 1, "T2", ca=4.0, day=26)
    _natural_run(folder, 2, "T1", ca=12.0, day=24)
    assert {p.name for p in repeated_naturals(_load(tmp_path))} == {"natural-1.csv"}


def test_each_subject_and_desk_level_keeps_its_own_first(tmp_path):
    _natural_run(tmp_path / "a", 1, "T1", ca=1.0, day=24)
    _natural_run(tmp_path / "a", 2, "T2", ca=2.0, day=26, extra={"desk_level": "desk-up"})
    _write(tmp_path / "b", "natural-1.csv", subject="s02", condition="natural", trip="T9",
           extra=BASE, captured_at="2026-11-27T10:00:00")
    assert repeated_naturals(_load(tmp_path)) == frozenset()


def test_a_camera_lost_partial_is_not_the_first_natural(tmp_path):
    folder = tmp_path / "s01"
    _natural_run(folder, 1, "T1", ca=1.0, day=24, extra={"camera_lost": 1})
    _natural_run(folder, 2, "T1b", ca=2.0, day=25)
    assert {p.name for p in repeated_naturals(_load(tmp_path))} == {"natural-1.csv"}


def test_the_cohort_report_lists_the_repeat_but_leaves_it_out_of_the_numbers(tmp_path):
    folder = tmp_path / "s01"
    _natural_run(folder, 1, "T1", ca=12.0, day=24)
    _natural_run(folder, 2, "T2", ca=4.0, day=26)
    cohort = collect([tmp_path])
    assert cohort.result("s01", "natural").theta_ca_deg == 12.0
    flags = {r["file"]: r["theta_ca_inclusion"] for r in session_rows(cohort)}
    assert flags["natural-2.csv"] == "重量不算" and flags["natural-1.csv"] != "重量不算"
    text = to_text(cohort, None)
    assert "標「重量」的自然坐" in text
    assert REPEATED_NATURAL in dict(trip_sections(cohort.sessions))["自然坐配對差值"]


def test_a_measured_natural_is_dropped_from_the_trip():
    trip1 = load_plan(Path("plans/selftest.json")).trips[0]
    measured = [ExistingSegment("natural", 1)]
    assert without_measured_natural(trip1, measured).before == ()
    assert without_measured_natural(trip1, measured).conditions == trip1.conditions
    assert without_measured_natural(trip1, []) is trip1
    # 別的桌面條件量的不算（和分析同一條規則）。
    other_desk = [ExistingSegment("natural", 1, desk_level="desk-up")]
    assert without_measured_natural(trip1, other_desk) is trip1
    same_desk = [ExistingSegment("natural", 1, desk_level="base")]
    assert without_measured_natural(trip1, same_desk).before == ()


def test_redoing_trip_1_does_not_record_natural_again(tmp_path, monkeypatch, capsys):
    folder = tmp_path / "sessions" / "S02"
    for name in ("natural-1.csv", "upright-1.csv", "head-forward-1.csv", "forward-1.csv"):
        _write_segment(folder, name)
    studio = _Studio()
    _patch(monkeypatch, studio, {"桌面高減手肘高": ["0"], "對嗎": ["y"]})
    posture._run_plan(_plan_args(tmp_path, PLAN, trips="trip1"))
    assert [r[0] for r in studio.records] == [
        "upright-2.csv", "head-forward-2.csv", "forward-2.csv", "upright-3.csv"]
    out = capsys.readouterr().out
    assert "已經量過自然坐" in out and "這次不打字" in out
    assert "段落：取基準 → upright" in out


def test_a_new_subject_still_records_natural_first(tmp_path, monkeypatch, capsys):
    studio = _Studio()
    _patch(monkeypatch, studio, {"桌面高減手肘高": ["0"], "對嗎": ["y"]})
    posture._run_plan(_plan_args(tmp_path, PLAN, trips="trip1"))
    assert studio.records[0] == ("natural-1.csv", False)
    assert "這次不打字" not in capsys.readouterr().out
