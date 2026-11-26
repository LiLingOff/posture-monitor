"""subjects 子指令：一眼看到每位受試者的組別與每一趟做到哪裡。

這個指令只讀檔，實機正在量測時也要能跑，所以不能碰相機，也不能寫任何東西。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from geometry.run_plan import (ExistingSegment, SubjectRow, load_plan, next_trips,
                               plan_for_order, resume_lines, subjects_text, trip_progress)
from posture_loader import load_posture_module

posture = load_posture_module("posture_subjects")
PLAN = load_plan(Path("plans/selftest.json"))


def _seg(condition, trial=1, rate=0.0, target=None):
    return ExistingSegment(condition, trial, rate, target)


TRIP1 = [_seg("natural"), _seg("upright", 1), _seg("forward"), _seg("head-forward"),
         _seg("upright", 2)]
TRIP4 = [_seg("right-shoulder-up"), _seg("left-shoulder-up")]
TILT_A = [_seg("tilt-left", 1, 0.0, 20.0), _seg("tilt-right", 1, 0.0, 20.0)]
TILT_B_PARTIAL = [_seg("tilt-right", 2, 0.0, 30.0)]


def _row(subject, group, segments, when=None):
    return SubjectRow(subject, group, tuple(segments), when)


def _lines(text):
    return text.splitlines()


def _line_for(text, subject):
    return next(line for line in _lines(text) if line.startswith(subject))


def test_a_subject_who_only_did_trip_1_is_listed_with_the_rest_not_done():
    text = subjects_text(PLAN, [_row("1123", "A", TRIP1)])
    row = _line_for(text, "1123")
    assert row.split()[:6] == ["1123", "A", "完成", "未做", "未做", "未做"]
    assert "--trips trip4,tiltA,tiltB" in text


def test_the_b_group_is_read_with_its_own_order_but_shown_in_the_plan_columns():
    # B 組的側傾 B 只做到右傾 30°：欄位仍是趟 1、趟 4、側傾 A、側傾 B。
    text = subjects_text(PLAN, [_row("411418", "B", TRIP1 + TILT_B_PARTIAL)])
    header = _lines(text)[2]
    assert header.index("趟 1") < header.index("趟 4") < header.index("側傾 A") \
        < header.index("側傾 B")
    row = _line_for(text, "411418")
    assert "未完" in row
    assert "tiltB 缺 tilt-left 10°" in row
    assert "--trips tiltB,tiltA,trip4" in text      # B 組照自己的順序補


def test_a_finished_trip_with_a_high_skip_rate_is_marked_redo_not_todo():
    segments = TRIP1 + TRIP4 + TILT_A + [_seg("tilt-right", 2, 0.46, 30.0),
                                          _seg("tilt-left", 2, 0.0, 10.0)]
    text = subjects_text(PLAN, [_row("411363", "B", segments)])
    assert "重量" in _line_for(text, "411363")
    assert "建議重量的趟：tiltB" in text
    assert "--trips" not in text          # 沒有還沒做完的趟，不印指令


def test_everything_done_says_so_instead_of_printing_a_command():
    segments = TRIP1 + TRIP4 + TILT_A + [_seg("tilt-right", 2, 0.0, 30.0),
                                          _seg("tilt-left", 2, 0.0, 10.0)]
    text = subjects_text(PLAN, [_row("S9", "A", segments)])
    assert "每一趟都做完了" in text and "--trips" not in text


def test_the_next_command_is_the_one_resume_lines_gives():
    progress = trip_progress(plan_for_order(PLAN, "A"), TRIP1)
    command = "python posture.py run-plan --subject X"
    resume = "\n".join(resume_lines(PLAN, progress, "X", command))
    todo, _redo, _both = next_trips(progress)
    assert f"{command} --trips {','.join(todo)}" in resume
    assert f"--subject X --trips {','.join(todo)}" in subjects_text(PLAN, [_row("X", "A", TRIP1)])


def test_last_measured_time_and_unregistered_folders_are_shown():
    text = subjects_text(PLAN, [_row("1123", "A", TRIP1, datetime(2026, 11, 23, 19, 14))],
                         unregistered=["S01", "a411127"])
    assert "11-23 19:14" in text
    assert "S01、a411127" in text
    assert "A 組 1 人" in text
    assert "轉身作廢要看 cohort" in text


def test_no_registered_subject_still_prints_a_sentence():
    text = subjects_text(PLAN, [], unregistered=["S01"])
    assert "還沒有登記組別" in text and "S01" in text


def _folder(root, name, files):
    folder = root / name
    folder.mkdir(parents=True)
    for file in files:
        (folder / file).write_text("frame\n", encoding="utf-8")


def test_the_command_reads_the_registry_next_to_the_sessions_folder(tmp_path, capsys):
    sessions = tmp_path / "data" / "sessions"
    _folder(sessions, "1123", ["natural-1.csv", "upright-1.csv"])
    _folder(sessions, "S01", ["upright-1.csv"])
    (tmp_path / "data" / "order_groups.csv").write_text("subject,group\n1123,B\n",
                                                        encoding="utf-8")
    posture._run_subjects(SimpleNamespace(plan=Path("plans/selftest.json"), out_dir=sessions))
    out = capsys.readouterr().out
    assert "1123" in _line_for(out, "1123") and "B" in _line_for(out, "1123").split()
    assert "沒有登記組別的資料夾" in out and "S01" in out.split("沒有登記組別的資料夾")[1]


def test_a_missing_sessions_folder_is_not_an_error(tmp_path, capsys):
    posture._run_subjects(SimpleNamespace(plan=Path("plans/selftest.json"),
                                          out_dir=tmp_path / "nothing"))
    assert "還沒有登記組別" in capsys.readouterr().out


def test_the_parser_knows_subjects_and_needs_no_camera():
    args = posture.build_parser().parse_args(["subjects"])
    assert args.mode == "subjects"
    assert args.plan == Path("plans/selftest.json")
    assert args.out_dir == Path("data/sessions")
    assert not hasattr(args, "calibration")


def test_a_bad_plan_file_is_a_plain_message(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        posture._run_subjects(SimpleNamespace(plan=bad, out_dir=tmp_path))


def test_a_registered_subject_with_no_folder_is_still_listed(tmp_path, capsys):
    # 2026-11-26 的 411348：分到 B 組、開跑後中斷，沒有任何量測檔。
    sessions = tmp_path / "data" / "sessions"
    _folder(sessions, "1123", ["natural-1.csv"])
    (tmp_path / "data" / "order_groups.csv").write_text(
        "subject,group\n1123,A\n411348,B\n", encoding="utf-8")
    posture._run_subjects(SimpleNamespace(plan=Path("plans/selftest.json"), out_dir=sessions))
    out = capsys.readouterr().out
    assert "A 組 1 人、B 組 1 人" in out
    assert _line_for(out, "411348").split()[:6] == ["411348", "B", "未做", "未做", "未做", "未做"]
    assert "--subject 411348 --trips trip1,tiltB,tiltA,trip4" in out
