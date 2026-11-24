"""順序效應的分析：A、B 兩組量同樣的姿勢，比較兩組。全部用合成 CSV，答案可以心算。"""
import pytest

from geometry.cohort_report import to_text
from geometry.order_effect import (order_csv, order_effect, order_text,
                                   posture_key)
from geometry.session_analysis import analyse_session
from geometry.trip_report import trip_csvs, trip_sections
from test_trips import _write


def _subject(root, subject, group, *, head=20.0, sym=0.0, extra=None, tilt=None):
    """一位受試者的一趟：端正坐（零點）加頭前傾；tilt 給 (姿勢, 目標角度, θ_sym)。"""
    folder = root / subject
    marker = {} if group is None else {"order_group": group}
    _write(folder, "upright-1.csv", subject=subject, condition="upright", trip="T1",
           ca=0.0, sym=0.0, extra=marker, captured_at="2026-11-24T10:00:00")
    _write(folder, "head-forward-1.csv", subject=subject, condition="head-forward",
           trip="T1", ca=head, sym=sym, extra={**marker, **(extra or {})},
           captured_at="2026-11-24T10:02:00")
    if tilt is not None:
        condition, target, tilt_sym = tilt
        _write(folder, f"{condition}-1.csv", subject=subject, condition=condition,
               trip="T1", ca=0.0, sym=tilt_sym,
               extra={**marker, "target_angle_deg": target},
               captured_at="2026-11-24T10:04:00")


def _load(root):
    return [analyse_session(p) for p in sorted(root.rglob("*.csv"))]


def _cell(effect, posture, label):
    return next(c for c in effect.cells if c.posture == posture and c.label == label)


def _four_people(tmp_path):
    for name, group, head in (("a1", "A", 20.0), ("a2", "A", 22.0),
                              ("b1", "B", 14.0), ("b2", "B", 16.0)):
        _subject(tmp_path, name, group, head=head)
    return order_effect(_load(tmp_path))


def test_the_difference_between_groups_is_a_minus_b_with_people_as_the_denominator(tmp_path):
    effect = _four_people(tmp_path)
    cell = _cell(effect, "head-forward", "θ_CA")
    assert cell.a.n == 2 and cell.b.n == 2
    assert cell.a.mean == pytest.approx(21.0) and cell.b.mean == pytest.approx(15.0)
    assert cell.difference == pytest.approx(6.0)
    assert cell.comparable and cell.error > 0
    assert cell.multiple == pytest.approx(cell.difference / cell.error)
    assert effect.people == {"A": 2, "B": 2} and effect.ungrouped == ()


def test_each_posture_is_measured_against_its_own_trips_first_upright(tmp_path):
    """端正坐零點差很多也不影響：比的是同趟的差值，不是原始角度。"""
    for name, group, zero in (("a1", "A", 0.0), ("a2", "A", 10.0),
                              ("b1", "B", 0.0), ("b2", "B", 10.0)):
        folder = tmp_path / name
        _write(folder, "upright-1.csv", subject=name, condition="upright", trip="T1",
               ca=zero, extra={"order_group": group}, captured_at="2026-11-24T10:00:00")
        _write(folder, "head-forward-1.csv", subject=name, condition="head-forward",
               trip="T1", ca=zero + 20.0, extra={"order_group": group},
               captured_at="2026-11-24T10:02:00")
    cell = _cell(order_effect(_load(tmp_path)), "head-forward", "θ_CA")
    assert cell.a.mean == pytest.approx(20.0) and cell.b.mean == pytest.approx(20.0)
    assert cell.difference == pytest.approx(0.0)


def test_a_group_with_fewer_than_two_people_cannot_be_compared(tmp_path):
    _subject(tmp_path, "a1", "A", head=20.0)
    _subject(tmp_path, "b1", "B", head=14.0)
    _subject(tmp_path, "b2", "B", head=16.0)
    cell = _cell(order_effect(_load(tmp_path)), "head-forward", "θ_CA")
    assert cell.a.n == 1 and cell.a.error is None
    assert not cell.comparable and cell.error is None and cell.multiple is None
    assert cell.difference == pytest.approx(20.0 - 15.0)     # 數字還是列出來


def test_natural_and_upright_are_not_compared(tmp_path):
    for name, group in (("a1", "A"), ("b1", "B")):
        folder = tmp_path / name
        _write(folder, "natural-1.csv", subject=name, condition="natural", trip="T1",
               ca=15.0, extra={"order_group": group}, captured_at="2026-11-24T09:58:00")
    _subject(tmp_path, "a1", "A")
    _subject(tmp_path, "b1", "B")
    postures = {c.posture for c in order_effect(_load(tmp_path)).cells}
    assert postures == {"head-forward"}


def test_tilt_postures_are_separated_by_target_angle(tmp_path):
    _subject(tmp_path, "a1", "A", tilt=("tilt-left", 10, 9.0))
    _subject(tmp_path, "a2", "A", tilt=("tilt-left", 20, 19.0))
    _subject(tmp_path, "b1", "B", tilt=("tilt-left", 10, 11.0))
    effect = order_effect(_load(tmp_path))
    ten = _cell(effect, "tilt-left 10°", "θ_sym")
    twenty = _cell(effect, "tilt-left 20°", "θ_sym")
    assert (ten.a.n, ten.b.n) == (1, 1)
    assert ten.a.mean == pytest.approx(9.0) and ten.b.mean == pytest.approx(11.0)
    assert (twenty.a.n, twenty.b.n) == (1, 0) and twenty.b.mean is None
    assert not twenty.comparable and twenty.difference is None


def test_postures_are_listed_in_a_fixed_order_with_tilt_angles_ascending(tmp_path):
    _subject(tmp_path, "a1", "A", tilt=("tilt-right", 30, -29.0))
    _subject(tmp_path, "a2", "A", tilt=("tilt-left", 20, 19.0))
    _subject(tmp_path, "b1", "B", tilt=("tilt-left", 10, 9.0))
    names = []
    for cell in order_effect(_load(tmp_path)).cells:
        if cell.posture not in names:
            names.append(cell.posture)
    assert names == ["head-forward", "tilt-left 10°", "tilt-left 20°", "tilt-right 30°"]


def test_the_header_group_wins_and_the_registry_fills_in_old_files(tmp_path):
    # 舊檔（1123 那幾位）標頭沒有組別，靠登記表補；標頭有的以標頭為準。
    _subject(tmp_path, "old1", None, head=20.0)
    _subject(tmp_path, "old2", None, head=22.0)
    _subject(tmp_path, "new1", "B", head=14.0)
    _subject(tmp_path, "new2", "B", head=16.0)
    registry = {"old1": "A", "old2": "A", "new1": "A", "new2": "A"}   # new* 登記錯了也不理
    cell = _cell(order_effect(_load(tmp_path), registry), "head-forward", "θ_CA")
    assert (cell.a.n, cell.b.n) == (2, 2)
    assert cell.a.mean == pytest.approx(21.0) and cell.b.mean == pytest.approx(15.0)


def test_people_with_no_group_are_counted_but_left_out(tmp_path):
    _subject(tmp_path, "a1", "A")
    _subject(tmp_path, "b1", "B")
    _subject(tmp_path, "who", None)
    effect = order_effect(_load(tmp_path))
    assert effect.ungrouped == ("who",)
    assert "who" in order_text(effect)


def test_no_groups_anywhere_means_no_section(tmp_path):
    _subject(tmp_path, "s1", None)
    summaries = _load(tmp_path)
    assert order_effect(summaries) is None
    assert all("順序效應" not in title for title, _ in trip_sections(summaries))
    assert "order_effect.csv" not in trip_csvs(summaries)


def test_a_person_with_a_redone_segment_counts_once(tmp_path):
    _subject(tmp_path, "a1", "A", head=20.0)
    _write(tmp_path / "a1", "head-forward-2.csv", subject="a1", condition="head-forward",
           trial=2, trip="T1", ca=24.0, extra={"order_group": "A"},
           captured_at="2026-11-24T10:03:00")
    _subject(tmp_path, "b1", "B", head=14.0)
    cell = _cell(order_effect(_load(tmp_path)), "head-forward", "θ_CA")
    assert cell.a.n == 1 and cell.a.mean == pytest.approx(22.0)    # 20 與 24 先平均


def test_the_report_and_csv_carry_the_numbers(tmp_path):
    effect = _four_people(tmp_path)
    text = order_text(effect)
    assert "A 組 2 人、B 組 2 人" in text and "head-forward" in text
    assert "自然坐兩組都排在最前面" in text
    lines = order_csv(effect).splitlines()
    assert lines[0].startswith("posture,quantity,a_n,a_mean")
    row = next(line for line in lines[1:] if line.startswith("head-forward,θ_CA"))
    assert ",2,21.0," in row and ",2,15.0," in row and ",6.0," in row
    summaries = _load(tmp_path)
    assert "order_effect.csv" in trip_csvs(summaries)
    titles = [t for t, _ in trip_sections(summaries)]
    assert "順序效應（A 組減 B 組）" in titles


def test_the_cohort_report_gets_the_registry_not_the_height_groups(tmp_path):
    """--subject-groups 是表4 的身高層，不是順序組別，兩個不能混。"""
    from geometry.cohort import collect
    for name, head in (("a1", 20.0), ("a2", 22.0), ("b1", 14.0), ("b2", 16.0)):
        _subject(tmp_path, name, None, head=head)
    cohort = collect([tmp_path])
    height = {"a1": "高", "a2": "高", "b1": "矮", "b2": "矮"}
    registry = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}
    with_registry = to_text(cohort, None, None, height, registry)
    assert "順序效應" in with_registry and "A 組 2 人、B 組 2 人" in with_registry
    without = to_text(cohort, None, None, height, None)
    assert "順序效應" not in without


def test_posture_key_shows_the_target_angle_only_for_tilts(tmp_path):
    _subject(tmp_path, "a1", "A", tilt=("tilt-left", 10, 9.0))
    keys = {posture_key(s) for s in _load(tmp_path)}
    assert keys == {"upright", "head-forward", "tilt-left 10°"}
