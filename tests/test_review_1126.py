"""2026-11-26 用本地 GPT 審查 11-22 之後的改動，採用項目的回歸測試。

假相機與假錄製沿用 test_run_plan.py。
"""
from __future__ import annotations

import json

import pytest

from geometry.order_effect import OrderEffect, order_text
from geometry.rating_pack import PackItem, write_pack
from geometry.recording import Recording
from geometry.run_plan import (ExistingSegment, SubjectRow, load_plan, parse_offset,
                               parse_plan, read_order_groups, register_order_group,
                               subjects_text, trip_progress)
from test_run_plan import (PLAN, _Studio, _patch, _plan_args, _write_segment, posture)


# ---- 分組只能由程式分派 ------------------------------------------------

def test_a_new_subject_cannot_pick_the_order_group(tmp_path, monkeypatch):
    monkeypatch.setattr(posture, "_prepare", lambda *a, **k: pytest.fail("不該開相機"))
    with pytest.raises(SystemExit, match="只給已經量過"):
        posture._run_plan_body(_plan_args(tmp_path, PLAN, order_group="B"))
    assert not (tmp_path / "order_groups.csv").exists()


def test_the_suggested_command_keeps_the_group_being_filled_in(
        tmp_path, monkeypatch, capsys):
    folder = tmp_path / "sessions" / "S02"
    for name in ("natural-1.csv", "upright-1.csv"):
        _write_segment(folder, name)
    monkeypatch.setattr(posture, "_prepare", lambda *a, **k: pytest.fail("不該開相機"))
    with pytest.raises(SystemExit, match="沒有開相機"):
        posture._run_plan_body(_plan_args(tmp_path, PLAN, order_group="A"))
    commands = [line for line in capsys.readouterr().out.splitlines() if "--trips" in line]
    assert commands and all("--order-group A" in line for line in commands)


def test_the_registry_is_appended_in_its_own_column_order(tmp_path):
    path = tmp_path / "order_groups.csv"
    path.write_text("group,subject\nA,x1\n", encoding="utf-8-sig")
    register_order_group(path, "x2", "B")
    assert read_order_groups(path) == {"x1": "A", "x2": "B"}


def test_an_order_may_not_move_the_trip_with_the_natural_segment():
    data = json.loads(PLAN.read_text(encoding="utf-8"))
    data["orders"]["B"]["trips"] = ["trip4", "trip1", "tiltB", "tiltA"]
    with pytest.raises(ValueError, match="自然坐"):
        parse_plan(data)


def test_the_report_does_not_claim_everything_is_reversed():
    text = order_text(OrderEffect((), {"A": 1, "B": 1}, ()))
    assert "全部反過來" not in text
    assert "趟 1" in text and "位置不變" in text


# ---- 沒量完的不能算量完 ------------------------------------------------

def _interrupt_at(studio, n):
    """第 n 次錄製提早結束（Ctrl+C）。"""
    real = studio.record

    def record(*args, **kwargs):
        recording = real(*args, **kwargs)
        if len(studio.records) == n:
            recording.interrupted = True
        return recording

    studio.record = record


def test_a_segment_stopped_early_does_not_pass_the_check(tmp_path, monkeypatch, capsys):
    studio = _Studio()
    _interrupt_at(studio, 3)
    asked, _ = _patch(monkeypatch, studio, {
        "桌面高減手肘高": ["0"], "對嗎": ["y"], "要做嗎": ["n"], "要重量這一段嗎": ["y"]})
    posture._run_plan(_plan_args(tmp_path))
    assert "提早結束" in capsys.readouterr().out
    names = [name for name, _ in studio.records]
    assert names[2] == names[3] == "head-forward-1.csv"


def test_record_marks_a_ctrl_c_stop(tmp_path, monkeypatch):
    def stop(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(posture, "_measure_once", stop)
    monkeypatch.setattr(posture, "_make_judge", lambda *a: None)
    args = _plan_args(tmp_path, PLAN)
    recording = posture._record(None, None, None, args, None, None, None, 60.0)
    assert isinstance(recording, Recording) and recording.interrupted


def test_a_trip_cut_off_by_the_camera_is_not_listed_as_done(
        tmp_path, monkeypatch, capsys):
    studio = _Studio()
    real = studio.record

    def record(*args, **kwargs):
        recording = real(*args, **kwargs)
        if len(studio.records) >= 2:
            recording.camera_lost = "相機掉了"
        return recording

    studio.record = record
    _patch(monkeypatch, studio, {"桌面高減手肘高": ["0"], "對嗎": ["y"]})
    posture._run_plan(_plan_args(tmp_path))
    out = capsys.readouterr().out
    assert "S02 沒有量完" in out
    assert "S02 量完了" not in out
    # 掉線次數用完就停，不拿壞掉的相機開下一趟。
    assert len(studio.records) == 2 + posture._STUDY_REOPEN_BUDGET


TRIP1_OWN = ("natural", "head-forward", "forward")


def _segments(uprights, trip_id="t1"):
    out = [ExistingSegment(c, 1, 0.0, trip_id=trip_id) for c in TRIP1_OWN]
    out += [ExistingSegment("upright", n + 1, 0.0, trip_id=trip_id)
            for n in range(uprights)]
    return out


def test_a_trip_missing_its_closing_upright_is_not_done():
    plan = load_plan(PLAN)
    short = trip_progress(plan, _segments(1))[0]
    assert not short.done and short.missing == ("upright",)
    assert trip_progress(plan, _segments(2))[0].done


def test_uprights_from_another_trip_do_not_fill_the_gap():
    plan = load_plan(PLAN)
    other = [ExistingSegment("upright", n, 0.0, trip_id="t4") for n in (3, 4)]
    assert not trip_progress(plan, _segments(0) + other)[0].done


def test_old_files_without_a_trip_id_are_judged_as_before():
    plan = load_plan(PLAN)
    old = [ExistingSegment(c, 1, 0.0) for c in TRIP1_OWN]
    assert trip_progress(plan, old)[0].done


def test_the_trip_id_is_read_from_the_csv_header(tmp_path, monkeypatch):
    (tmp_path / "upright-1.csv").write_text("frame\n", encoding="utf-8")
    summary = type("S", (), {"frames": 1, "usable": 1,
                             "meta": {"trip_id": "2026-11-26T10:00:00"}})()
    monkeypatch.setattr(posture, "analyse_session", lambda path: summary)
    segment, = posture._existing_segments(tmp_path)
    assert segment.trip_id == "2026-11-26T10:00:00"


# ---- 輸入與顯示 --------------------------------------------------------

@pytest.mark.parametrize("text", ["nan", "inf", "-inf"])
def test_offset_rejects_non_finite_numbers(text):
    assert parse_offset(text) is None


def test_the_protractor_rejects_non_finite_numbers(monkeypatch):
    replies = iter(["nan", "inf", "21"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))
    args = type("A", (), {"require_angles": True})()
    assert posture._ask_tilt_angle("幾度？", args) == 21.0


def test_exactly_fifteen_percent_skipped_is_not_over_the_limit(tmp_path, monkeypatch):
    (tmp_path / "upright-1.csv").write_text("frame\n", encoding="utf-8")
    summary = type("S", (), {"frames": 100, "usable": 85, "meta": {}})()
    monkeypatch.setattr(posture, "analyse_session", lambda path: summary)
    segment, = posture._existing_segments(tmp_path)
    assert segment.rejection_rate == 0.15


def test_the_next_step_lines_can_be_pasted_as_they_are():
    text = subjects_text(load_plan(PLAN), [SubjectRow("S02", "A", ())])
    line, = [x for x in text.splitlines() if "--trips" in x]
    assert line.strip().startswith("python posture.py run-plan --subject S02")


def test_a_bad_subject_code_leaves_no_log_behind(tmp_path):
    with pytest.raises(SystemExit, match="--subject"):
        posture._run_plan(_plan_args(tmp_path, PLAN, subject="../x"))
    assert not list(tmp_path.rglob("*.log"))


# ---- 盲評包不能改掉別人的 .gitignore -----------------------------------

def _one_item(tmp_path):
    image = tmp_path / "a.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    return PackItem(source=image, kind="snapshot", subject="x", trip_id="t",
                    condition="upright", trial="1", segment="x/upright-1.csv",
                    segment_csv=tmp_path / "upright-1.csv")


def test_an_existing_gitignore_next_to_the_key_is_kept(tmp_path):
    (tmp_path / ".gitignore").write_text("data/\n", encoding="utf-8")
    write_pack([_one_item(tmp_path)], tmp_path / "pack", ["R1"],
               key_path=tmp_path / "key.csv")
    lines = (tmp_path / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert lines == ["data/", "key.csv"]
