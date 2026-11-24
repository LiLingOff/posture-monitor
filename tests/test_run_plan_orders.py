"""順序效應（2026-11-24）：計畫檔的 orders、分組、登記。

說明書寫「隨機分兩組採不同姿態呈現順序」。A 組是計畫檔本身的順序，B 組在 orders 裡，
只換順序，不能增減段落或改側傾角度；自然坐與取基準後第一段端正坐固定不動。
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import pytest

from geometry.run_plan import (ORDER_A, assign_order, load_plan, parse_plan,
                               plan_for_order, read_order_groups,
                               register_order_group)
from geometry.run_plan import _all_keys

PLAN = Path("plans/selftest.json")


def _data():
    return json.loads(PLAN.read_text(encoding="utf-8"))


def test_the_shipped_plan_has_a_reversed_b_order():
    plan = load_plan(PLAN)
    assert plan.groups == ("A", "B")
    a = plan_for_order(plan, "A")
    b = plan_for_order(plan, "B")
    assert [t.trip_id for t in a.trips] == ["trip1", "trip4", "tiltA", "tiltB"]
    assert [t.trip_id for t in b.trips] == ["trip1", "tiltB", "tiltA", "trip4"]
    by_id = {t.trip_id: t for t in b.trips}
    assert by_id["trip1"].conditions == ("upright", "forward", "head-forward", "upright")
    assert by_id["trip4"].conditions == ("upright", "right-shoulder-up", "left-shoulder-up")
    # 側傾的目標角度跟著姿勢走：B 組先右後左，所以 30 在前、10 在後。
    assert by_id["tiltB"].conditions == ("upright", "tilt-right", "tilt-left")
    assert by_id["tiltB"].target_angles == (30.0, 10.0)
    assert by_id["tiltA"].target_angles == (20.0, 20.0)


def test_both_orders_measure_exactly_the_same_segments():
    plan = load_plan(PLAN)
    a = {t.trip_id: t for t in plan_for_order(plan, "A").trips}
    b = {t.trip_id: t for t in plan_for_order(plan, "B").trips}
    for trip_id, trip in a.items():
        assert sorted(_all_keys(trip)) == sorted(_all_keys(b[trip_id])), trip_id


def test_the_fixed_parts_do_not_move_in_the_b_order():
    plan = load_plan(PLAN)
    b = plan_for_order(plan, "B")
    # 自然坐固定在整個流程最前面：要在任何坐姿指示之前量。
    assert b.trips[0].trip_id == "trip1" and b.trips[0].before == ("natural",)
    for trip in b.trips:
        assert trip.conditions[0] == "upright"          # 取基準後第一段
    assert b.trips[0].conditions[-1] == "upright"       # 趟 1 的漂移檢查


def test_group_a_is_the_plan_itself_and_an_unknown_group_is_refused():
    plan = load_plan(PLAN)
    assert plan_for_order(plan, ORDER_A) is plan
    with pytest.raises(ValueError, match="沒有 C 組"):
        plan_for_order(plan, "C")


def _bad(mutate, message):
    data = _data()
    mutate(data)
    with pytest.raises(ValueError, match=message):
        parse_plan(data)


def test_a_plan_without_orders_still_loads_with_only_group_a():
    data = _data()
    del data["orders"]
    plan = parse_plan(data)
    assert plan.groups == ("A",) and plan.orders == ()


def test_the_order_file_is_checked_before_the_camera_opens():
    _bad(lambda d: d["orders"]["B"].update(trips=["trip1", "tiltB", "tiltA"]),
         "每一趟各寫一次")
    _bad(lambda d: d["orders"]["B"].update(trips=["trip1", "trip1", "tiltA", "tiltB"]),
         "每一趟各寫一次")
    _bad(lambda d: d["orders"].update({"A": copy.deepcopy(d["orders"]["B"])}),
         "不能再寫 A")
    _bad(lambda d: d["orders"]["B"]["conditions"].update(trip9=["upright"]),
         "沒有的趟次：trip9")
    _bad(lambda d: d.update(orders=["B"]), "要是一個物件")


def test_another_group_may_only_reorder_never_add_drop_or_change_angles():
    # 少一段。
    _bad(lambda d: d["orders"]["B"]["conditions"].update(
        trip4=["upright", "right-shoulder-up"]), "不能增減段落")
    # 側傾角度被偷偷換掉：右傾 10、左傾 30，與 A 組的左 10、右 30 不是同一批段。
    _bad(lambda d: d["orders"]["B"]["target_angle_deg"].update(tiltB=[10, 30]),
         "不能增減段落或改側傾角度")
    # 側傾段數與角度個數對不上。
    _bad(lambda d: d["orders"]["B"]["target_angle_deg"].update(tiltB=[30]),
         "target_angle_deg 卻有 1 個")


def test_the_first_upright_and_the_drift_check_cannot_be_moved():
    # 取基準後第一段不是端正坐：5° 基準檢查與轉身參考都靠它。
    _bad(lambda d: d["orders"]["B"]["conditions"].update(
        trip4=["right-shoulder-up", "upright", "left-shoulder-up"]),
        "取基準後第一段要和 A 組一樣")
    # 趟 1 最後一段端正坐是漂移檢查，不能被換走。
    _bad(lambda d: d["orders"]["B"]["conditions"].update(
        trip1=["upright", "upright", "forward", "head-forward"]),
        "最後一段要和 A 組一樣")


def test_assignment_goes_to_the_smaller_group_and_breaks_ties_at_random():
    groups = ("A", "B")
    # B 比較少，一定分到 B，不論亂數。
    registry = {"s1": "A", "s2": "A", "s3": "B"}
    assert {assign_order(registry, groups, random.Random(n)) for n in range(20)} == {"B"}
    # 一樣多時兩組都會出現。
    tie = {"s1": "A", "s2": "B"}
    assert {assign_order(tie, groups, random.Random(n)) for n in range(40)} == {"A", "B"}
    # 登記裡有計畫沒有的組別，不計入。
    assert assign_order({"s1": "Z"}, groups, random.Random(1)) in groups


def test_assignment_keeps_the_groups_within_one_person_of_each_other():
    rng = random.Random(2026)
    registry = {}
    for n in range(41):
        registry[f"S{n}"] = assign_order(registry, ("A", "B"), rng)
        counts = [list(registry.values()).count(g) for g in ("A", "B")]
        assert abs(counts[0] - counts[1]) <= 1


def test_the_registry_is_created_appended_and_read_back(tmp_path):
    path = tmp_path / "data" / "order_groups.csv"
    assert read_order_groups(path) == {}
    register_order_group(path, "1123", "A")
    register_order_group(path, "S02", "B")
    assert read_order_groups(path) == {"1123": "A", "S02": "B"}
    assert path.read_text(encoding="utf-8").splitlines()[0] == "subject,group"


def test_the_registry_survives_a_hand_made_file_without_a_final_newline(tmp_path):
    path = tmp_path / "order_groups.csv"
    path.write_bytes("subject,group\n1123,A".encode("utf-8"))
    register_order_group(path, "S02", "B")
    assert read_order_groups(path) == {"1123": "A", "S02": "B"}


def test_a_blank_group_counts_as_not_registered(tmp_path):
    path = tmp_path / "order_groups.csv"
    path.write_text("subject,group\n1123,\nS02,B\n", encoding="utf-8")
    assert read_order_groups(path) == {"S02": "B"}
