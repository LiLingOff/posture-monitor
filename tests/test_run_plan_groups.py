"""run-plan 的順序分組（2026-11-24）：分派、登記、照組別的順序量、寫進標頭。

假相機與假錄製沿用 test_run_plan.py，這裡只驗分組那一層。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from geometry.run_plan import load_plan, plan_for_order, trip_argv
from geometry.study import build_trip_options, segment_metadata
from test_run_plan import PLAN, _Studio, _patch, _plan_args, _write_segment, posture


class _Pick:
    """固定分到最後一組（B），免得亂數讓測試時好時壞。"""

    def choice(self, seq):
        return list(seq)[-1]


# 趟 1 → 側傾 B → 側傾 A → 趟 4，每趟內也反過來。B 組的側傾 B 先右後左（30°、10°），
# 側傾 A 先右後左（20°、20°）。端正坐的編號每趟 +1。
B_ORDER = [
    "natural-1.csv", "upright-1.csv", "forward-1.csv", "head-forward-1.csv",
    "upright-2.csv",
    "upright-3.csv", "tilt-right-1.csv", "tilt-left-1.csv",
    "upright-4.csv", "tilt-right-2.csv", "tilt-left-2.csv",
    "upright-5.csv", "right-shoulder-up-1.csv", "left-shoulder-up-1.csv",
]

# 量角器每段問兩次（開始、結束）：B 組先側傾 B 右、左，再側傾 A 右、左。
ANGLES = ["31", "29", "11", "9", "21", "19", "21", "19"]


def _answers():
    return {"桌面高減手肘高": ["0"], "對嗎": ["y"], "量角器": list(ANGLES)}


def _meta(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# ") and "=" in line:
            key, _, value = line[2:].partition("=")
            out[key] = value
    return out


def test_a_new_subject_is_assigned_registered_and_measured_in_that_order(
        tmp_path, monkeypatch, capsys):
    studio = _Studio()
    asked, analysed = _patch(monkeypatch, studio, _answers())
    monkeypatch.setattr(posture, "_ORDER_RNG", _Pick())
    posture._run_plan(_plan_args(tmp_path, PLAN, order_group=None))

    assert [r[0] for r in studio.records] == B_ORDER
    assert len(analysed) == 1 and len(analysed[0]) == 14
    registry = (tmp_path / "order_groups.csv").read_text(encoding="utf-8").splitlines()
    assert registry == ["subject,group", "S02,B"]
    out = capsys.readouterr().out
    assert "S02 是 B 組（剛分派）" in out
    assert out.count("順序：B 組") == 4          # 四趟的標題各一行

    folder = tmp_path / "sessions" / "S02"
    # 組別寫進每一段；側傾 B 的右傾是 30°、左傾是 10°，側傾 A 兩邊都是 20°。
    for name in B_ORDER:
        assert _meta(folder / name)["order_group"] == "B", name
    assert _meta(folder / "tilt-right-1.csv")["target_angle_deg"] == "30"
    assert _meta(folder / "tilt-left-1.csv")["target_angle_deg"] == "10"
    assert _meta(folder / "tilt-right-2.csv")["target_angle_deg"] == "20"


def test_group_a_keeps_the_order_the_plan_file_has(tmp_path, monkeypatch):
    studio = _Studio()
    _patch(monkeypatch, studio, {"桌面高減手肘高": ["0"], "對嗎": ["y"],
                                 "量角器": list(ANGLES)})
    posture._run_plan(_plan_args(tmp_path, PLAN, order_group="A"))
    assert [r[0] for r in studio.records] == [
        "natural-1.csv", "upright-1.csv", "head-forward-1.csv", "forward-1.csv",
        "upright-2.csv", "upright-3.csv", "left-shoulder-up-1.csv",
        "right-shoulder-up-1.csv", "upright-4.csv", "tilt-left-1.csv",
        "tilt-right-1.csv", "upright-5.csv", "tilt-left-2.csv", "tilt-right-2.csv"]
    folder = tmp_path / "sessions" / "S02"
    assert _meta(folder / "tilt-left-1.csv")["target_angle_deg"] == "20"
    assert _meta(folder / "tilt-left-2.csv")["target_angle_deg"] == "10"


def test_a_registered_subject_keeps_the_registered_group(tmp_path, monkeypatch):
    (tmp_path / "order_groups.csv").write_text("subject,group\nS02,B\n", encoding="utf-8")
    studio = _Studio()
    _patch(monkeypatch, studio, _answers())
    # 就算亂數會抽到別組，已登記的人照登記。
    monkeypatch.setattr(posture, "_ORDER_RNG", type("R", (), {"choice": lambda self, s: "A"})())
    posture._run_plan(_plan_args(tmp_path, PLAN, order_group=None))
    assert [r[0] for r in studio.records] == B_ORDER


def test_a_registered_group_cannot_be_changed_by_hand(tmp_path, monkeypatch):
    (tmp_path / "order_groups.csv").write_text("subject,group\nS02,B\n", encoding="utf-8")
    _patch(monkeypatch, _Studio(), _answers())
    with pytest.raises(SystemExit, match="不能改成 A 組"):
        posture._run_plan(_plan_args(tmp_path, PLAN, order_group="A"))


def test_an_unknown_group_is_refused_before_anything_else(tmp_path, monkeypatch):
    _patch(monkeypatch, _Studio(), _answers())
    with pytest.raises(SystemExit, match="不在計畫的組別裡"):
        posture._run_plan(_plan_args(tmp_path, PLAN, order_group="Z"))


def test_a_subject_with_data_but_no_registered_group_is_not_guessed(
        tmp_path, monkeypatch):
    """1123 這種登記制度之前就量過的人，程式不知道他照哪個順序量，不能猜。"""
    folder = tmp_path / "sessions" / "S02"
    for name in ("natural-1.csv", "upright-1.csv"):
        _write_segment(folder, name)
    opened = []
    monkeypatch.setattr(posture, "_prepare", lambda *a, **k: opened.append(1))
    with pytest.raises(SystemExit, match="還沒登記順序組別"):
        posture._run_plan_body(_plan_args(tmp_path, PLAN, order_group=None))
    assert not opened and not (tmp_path / "order_groups.csv").exists()


def test_filling_in_the_group_without_trips_still_stops_and_registers_nothing(
        tmp_path, monkeypatch):
    """已經量過的人不給 --trips 就是整個重來，程式拒絕；這時不該留下登記。"""
    folder = tmp_path / "sessions" / "S02"
    for name in ("natural-1.csv", "upright-1.csv"):
        _write_segment(folder, name)
    monkeypatch.setattr(posture, "_prepare", lambda *a, **k: pytest.fail("不該開相機"))
    with pytest.raises(SystemExit, match="沒有開相機"):
        posture._run_plan_body(_plan_args(tmp_path, PLAN, order_group="A"))
    assert not (tmp_path / "order_groups.csv").exists()


def test_filling_in_the_group_with_trips_registers_it_before_the_camera_opens(
        tmp_path, monkeypatch):
    """1123 這種之前就量過的人：--order-group A --trips tiltB 補登記並接著量。"""
    folder = tmp_path / "sessions" / "S02"
    for name in ("natural-1.csv", "upright-1.csv"):
        _write_segment(folder, name)

    def stop_at_camera(*a, **k):
        raise RuntimeError("camera")

    monkeypatch.setattr(posture, "_prepare", stop_at_camera)
    with pytest.raises(RuntimeError, match="camera"):
        posture._run_plan_body(_plan_args(tmp_path, PLAN, order_group="A", trips="tiltB"))
    assert (tmp_path / "order_groups.csv").read_text(
        encoding="utf-8").splitlines() == ["subject,group", "S02,A"]


def test_a_dry_run_shows_the_group_but_registers_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(posture, "_ORDER_RNG", _Pick())
    posture._run_plan(_plan_args(tmp_path, PLAN, order_group=None, dry_run=True))
    out = capsys.readouterr().out
    assert "S02 是 B 組（預演，這次沒有登記）" in out
    assert not (tmp_path / "order_groups.csv").exists()
    # 預演印的「等同指令」也帶組別，而且是 B 組的順序。
    assert "--order-group B" in out
    assert "--conditions upright,forward,head-forward,upright" in out


def test_each_group_becomes_a_study_command_the_real_parser_accepts():
    plan = load_plan(PLAN)
    parser = posture.build_parser()
    for group in plan.groups:
        for trip in plan_for_order(plan, group).trips:
            argv = trip_argv(plan_for_order(plan, group), trip, "S02", 0.0, group)
            ns = parser.parse_args(["study"] + argv)
            assert ns.order_group == group
            assert tuple(ns.conditions.split(",")) == trip.conditions


def test_study_writes_the_group_only_when_there_is_one():
    plain = build_trip_options("t", ["upright"], desk_level="base")
    grouped = build_trip_options("t", ["upright"], desk_level="base", order_group="B")
    assert "order_group" not in segment_metadata(plain, "upright", 0, None)
    assert segment_metadata(grouped, "upright", 0, None)["order_group"] == "B"
