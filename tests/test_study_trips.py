"""`study` 的趟次重排（2026-11-18）：基準前的段落、趟次中繼資料、轉身參考值。

說明書的核心假設要看自然坐的角度，但 study 一開始就念「請坐正」取基準，
之後量的自然坐已經被污染。實機資料又顯示原始角度跨趟差很多（θ_CA 跨趟差到
16°，θ_sym 的零點每趟不同），所以自然坐要排在基準之前，並與同一趟的端正坐
配對。這些測試盯的是順序、配對鍵，以及「不給新選項時與先前一樣」。
"""
from types import SimpleNamespace

import pytest

from geometry.baseline import PostureBaseline
from geometry.recording import Recording, RollingAngle, Session
from geometry.session_analysis import read_rows
from geometry.study import (build_trip_options, condition_brief, review,
                            segment_metadata)
from posture_loader import load_posture_module

posture = load_posture_module("posture_study")


def _baseline(**overrides) -> PostureBaseline:
    values = dict(
        subject="s01", captured_at="2026-11-18T10:00:00", frames=149, rejected=0,
        duration_s=30.0, theta_ca_deg=3.78, theta_sym_deg=5.30,
        theta_ca_std_deg=4.5, theta_sym_std_deg=0.5,
        theta_ca_standard_error_deg=0.83, theta_sym_standard_error_deg=0.09,
        distance_mm=685.0, azimuth_deg=40.0,
    )
    values.update(overrides)
    return PostureBaseline(**values)


def _args(tmp_path, **overrides):
    values = dict(
        subject="s01", conditions="upright,forward", seconds=30.0,
        seconds_each=60.0, countdown=0, out_dir=tmp_path, no_prompt=True,
        overwrite=True, window=30, margin=1.0, no_judge=False,
        snapshot_every=10.0, save_frames=None, discard=0,
        calibration="x.npz", camera=0, width=None, height=None,
        vertical_split=False, swap_lr=False, precision="fp32", device="cpu",
        checkpoint="c", engine_cache="e", repo_dir="r",
        min_confidence=0.0, input_height=256, no_subpixel=False, mode="study",
        display=False, display_width=1280, display_eyes="both",
        font=None, display_lang="en",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeCap:
    def release(self):
        pass


class _Events:
    """記下取基準與每一段錄製的先後順序，以及每一段拿到的基準。"""

    def __init__(self, azimuth=40.0, lost_on=()):
        self.events = []
        self._azimuth = azimuth
        self._lost_on = set(lost_on)
        self._calls = 0

    def collect(self, engine, calib, cap, args, seconds, display=None):
        self.events.append(("baseline", None))
        return _baseline(), []

    def record(self, engine, calib, cap, args, baseline, log, snapshots,
               seconds, display=None):
        self._calls += 1
        self.events.append(("record", baseline))
        session = Session()
        for _ in range(50):
            session.add((5.0, 1.0))
        lost = "相機連續 25 次讀不到畫面" if self._calls in self._lost_on else None
        return Recording(
            session=session, ca_window=RollingAngle(30),
            sym_window=RollingAngle(30), rejected=1, frames=51,
            camera_lost=lost, log_path=None if log is None else log.path,
            azimuths=[self._azimuth] * 50,
        )


def _patch(monkeypatch, events, countdowns=None):
    monkeypatch.setattr(posture, "_prepare", lambda args, **kw: (None, None, _FakeCap()))
    monkeypatch.setattr(posture, "_collect_baseline", events.collect)
    monkeypatch.setattr(posture, "_record", events.record)
    monkeypatch.setattr(posture, "_report", lambda *a, **k: None)
    monkeypatch.setattr(posture, "_reopen_camera", lambda args, **kw: _FakeCap())
    monkeypatch.setattr(
        posture, "_countdown",
        lambda seconds, message="": None if countdowns is None
        else countdowns.append(seconds))


def _meta(path):
    """錄製被換掉了，CSV 只有標頭沒有資料列，所以直接讀標頭。"""
    meta = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            key, _, value = line.lstrip("#").strip().partition("=")
            meta[key.strip()] = value.strip()
    return meta


# ---- 順序 --------------------------------------------------------------

def test_before_baseline_segments_come_first_and_get_no_baseline(tmp_path, monkeypatch):
    events = _Events()
    countdowns = []
    _patch(monkeypatch, events, countdowns)
    posture._run_study(_args(tmp_path, before_baseline="natural",
                             conditions="upright,head-forward,forward", countdown=17))

    kinds = [kind for kind, _ in events.events]
    assert kinds == ["record", "baseline", "record", "record", "record"]
    # 自然坐不扣基準，基準後的段落都拿到同一份基準
    assert events.events[0][1] is None
    assert all(b is not None for kind, b in events.events[2:])
    # 自然坐也要倒數（與取基準的規則一致）；第二次是取基準本身的倒數，
    # 基準後的段落維持原樣不倒數
    assert countdowns == [17, 17]

    folder = tmp_path / "s01"
    natural = _meta(folder / "natural-1.csv")
    upright = _meta(folder / "upright-1.csv")
    assert "baseline_file" not in natural
    assert "baseline_theta_ca_deg" not in natural
    assert "baseline_theta_ca_deg" in upright
    # 自然坐與端正坐靠 trip_id 配對
    assert natural["trip_id"] == upright["trip_id"]


def test_an_empty_condition_list_means_no_baseline_at_all(tmp_path, monkeypatch):
    """趟 5（表4）：不取基準，三個方位角各一段頭部前傾。"""
    events = _Events()
    _patch(monkeypatch, events)
    posture._run_study(_args(
        tmp_path, before_baseline="head-forward,head-forward,head-forward",
        conditions="", azimuth_target_deg="30,45,55"))

    assert [kind for kind, _ in events.events] == ["record"] * 3
    folder = tmp_path / "s01"
    assert not (folder / "baseline.json").exists()
    targets = [_meta(folder / f"head-forward-{i}.csv")["azimuth_target_deg"]
               for i in (1, 2, 3)]
    assert targets == ["30", "45", "55"]


def test_both_lists_empty_stops_before_the_camera(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(posture, "_prepare", lambda args, **kw: opened.append(1))
    with pytest.raises(SystemExit, match="至少要有一種姿勢"):
        posture._run_study(_args(tmp_path, before_baseline=" , ", conditions=""))
    assert not opened


def test_without_the_new_option_an_empty_condition_list_is_still_an_error(tmp_path):
    with pytest.raises(SystemExit, match="至少要有一種姿勢"):
        posture._run_study(_args(tmp_path, conditions=""))


def test_without_the_new_options_the_flow_is_unchanged(tmp_path, monkeypatch):
    """不給 --before-baseline：先取基準，各段依序，標頭只多一列 trip_id。"""
    events = _Events()
    countdowns = []
    _patch(monkeypatch, events, countdowns)
    posture._run_study(_args(tmp_path, conditions="upright,forward"))

    assert [kind for kind, _ in events.events] == ["baseline", "record", "record"]
    assert countdowns == [0]     # 只有取基準那一次（_study_baseline 的倒數）
    folder = tmp_path / "s01"
    assert sorted(p.name for p in folder.glob("*.csv")) == ["forward-1.csv",
                                                           "upright-1.csv"]
    meta = _meta(folder / "upright-1.csv")
    new_keys = {"trip_id", "desk_level", "lift_cm", "desk_offset_cm", "task",
                "target_angle_deg", "measured_angle_deg", "azimuth_target_deg"}
    assert set(meta) & new_keys == {"trip_id"}
    assert meta["condition"] == "upright" and meta["trial"] == "1"
    assert "baseline_file" in meta


def test_the_new_metadata_lines_come_after_the_old_ones(tmp_path, monkeypatch):
    _patch(monkeypatch, _Events())
    posture._run_study(_args(tmp_path, before_baseline="natural", conditions="upright",
                             desk_level="desk-up", lift_cm=5.0, desk_offset_cm=-2.5))
    lines = [line for line in
             (tmp_path / "s01" / "natural-1.csv").read_text(encoding="utf-8").splitlines()
             if line.startswith("#")]
    keys = [line[2:].split("=")[0] for line in lines]
    assert keys.index("trip_id") > keys.index("trial")
    assert keys[-5:] == ["trip_id", "desk_level", "lift_cm", "desk_offset_cm", "task"]


# ---- 標頭欄位 ----------------------------------------------------------

def test_desk_metadata_reaches_every_segment_of_the_trip(tmp_path, monkeypatch):
    _patch(monkeypatch, _Events())
    posture._run_study(_args(tmp_path, before_baseline="natural",
                             conditions="upright,tilt-left",
                             desk_level="desk-up", lift_cm=5.0, desk_offset_cm=-2.5,
                             target_angle_deg="20", measured_angle_deg="19"))
    folder = tmp_path / "s01"
    for name in ("natural-1", "upright-1", "tilt-left-1"):
        meta = _meta(folder / f"{name}.csv")
        assert meta["desk_level"] == "desk-up"
        assert meta["lift_cm"] == "5"
        assert meta["desk_offset_cm"] == "-2.5"
    assert _meta(folder / "natural-1.csv")["task"] == "copy-text"
    assert "task" not in _meta(folder / "upright-1.csv")
    tilt = _meta(folder / "tilt-left-1.csv")
    assert tilt["target_angle_deg"] == "20" and tilt["measured_angle_deg"] == "19"
    assert "target_angle_deg" not in _meta(folder / "upright-1.csv")


def test_the_base_desk_level_records_zero_lift(tmp_path, monkeypatch):
    _patch(monkeypatch, _Events())
    posture._run_study(_args(tmp_path, before_baseline="natural", conditions="upright",
                             desk_level="base", task="read-article"))
    meta = _meta(tmp_path / "s01" / "natural-1.csv")
    assert meta["lift_cm"] == "0"
    assert meta["task"] == "read-article"


def test_angles_line_up_with_the_tilt_segments_in_order(tmp_path, monkeypatch):
    """趟 2 的前 5 人：左傾 20° 之外再加左傾 10°。"""
    _patch(monkeypatch, _Events())
    posture._run_study(_args(tmp_path, conditions="upright,tilt-left,tilt-left",
                             target_angle_deg="20,10", measured_angle_deg="21,9.5"))
    folder = tmp_path / "s01"
    assert _meta(folder / "tilt-left-1.csv")["measured_angle_deg"] == "21"
    assert _meta(folder / "tilt-left-2.csv")["target_angle_deg"] == "10"
    assert _meta(folder / "tilt-left-2.csv")["measured_angle_deg"] == "9.5"


def test_a_retake_keeps_its_own_angle(tmp_path, monkeypatch):
    """掉線重量的那一段，目標角度不能被往後挪一格。"""
    _patch(monkeypatch, _Events(lost_on={2}))
    posture._run_study(_args(tmp_path, conditions="upright,tilt-left,tilt-right",
                             target_angle_deg="20,30"))
    folder = tmp_path / "s01"
    assert _meta(folder / "tilt-left-1.csv")["target_angle_deg"] == "20"
    assert _meta(folder / "tilt-left-2.csv")["target_angle_deg"] == "20"
    assert _meta(folder / "tilt-right-1.csv")["target_angle_deg"] == "30"


def test_the_measured_angle_is_asked_when_not_given(tmp_path, monkeypatch):
    _patch(monkeypatch, _Events())
    monkeypatch.setattr(posture, "_prompt", lambda message, interactive=True: None)
    answers = iter(["18.5", "16"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    posture._run_study(_args(tmp_path, conditions="upright,tilt-right",
                             no_prompt=False, target_angle_deg="20"))
    meta = _meta(tmp_path / "s01" / "tilt-right-1.csv")
    assert meta["measured_angle_deg"] == "18.5"
    # 量完那一段之後再問一次，記在檔尾（read_rows 收任何位置的 # key=value）
    assert meta["measured_end_angle_deg"] == "16"


def test_a_blank_end_angle_leaves_only_the_start_value(tmp_path, monkeypatch):
    _patch(monkeypatch, _Events())
    monkeypatch.setattr(posture, "_prompt", lambda message, interactive=True: None)
    answers = iter(["19", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    posture._run_study(_args(tmp_path, conditions="upright,tilt-left",
                             no_prompt=False, target_angle_deg="20"))
    meta = _meta(tmp_path / "s01" / "tilt-left-1.csv")
    assert meta["measured_angle_deg"] == "19"
    assert "measured_end_angle_deg" not in meta


def test_the_end_angles_given_up_front_line_up_with_the_tilt_segments(tmp_path,
                                                                       monkeypatch):
    _patch(monkeypatch, _Events())
    posture._run_study(_args(tmp_path, conditions="upright,tilt-left,tilt-left",
                             target_angle_deg="20,10", measured_angle_deg="21,9.5",
                             measured_end_angle_deg="18,9"))
    folder = tmp_path / "s01"
    assert _meta(folder / "tilt-left-1.csv")["measured_end_angle_deg"] == "18"
    assert _meta(folder / "tilt-left-2.csv")["measured_end_angle_deg"] == "9"
    assert "measured_end_angle_deg" not in _meta(folder / "upright-1.csv")


def test_no_prompt_does_not_ask_for_the_end_angle(tmp_path, monkeypatch):
    _patch(monkeypatch, _Events())

    def boom(prompt=""):
        raise AssertionError("no_prompt 不該問")

    monkeypatch.setattr("builtins.input", boom)
    posture._run_study(_args(tmp_path, conditions="upright,tilt-left",
                             target_angle_deg="20", measured_angle_deg="20"))
    assert "measured_end_angle_deg" not in _meta(tmp_path / "s01" / "tilt-left-1.csv")


@pytest.mark.parametrize("overrides, message", [
    (dict(conditions="upright,tilt-left", target_angle_deg="20,10"), "側傾"),
    (dict(conditions="upright", desk_level="base", lift_cm=3.0), "base"),
    (dict(conditions="upright,forward", azimuth_target_deg="30,45,55"), "azimuth"),
    (dict(conditions="upright,tilt-left", measured_angle_deg="abc"), "不是數字"),
    (dict(conditions="upright,tilt-left", measured_end_angle_deg="19,18"), "側傾"),
])
def test_mismatched_options_stop_before_the_camera(tmp_path, monkeypatch, overrides,
                                                    message):
    opened = []
    monkeypatch.setattr(posture, "_prepare", lambda args, **kw: opened.append(1))
    with pytest.raises(SystemExit, match=message):
        posture._run_study(_args(tmp_path, **overrides))
    assert not opened


def test_segment_metadata_leaves_out_what_was_not_given():
    options = build_trip_options("2026-11-18T10:00:00", ["upright"])
    assert segment_metadata(options, "upright", 0, None) == {
        "trip_id": "2026-11-18T10:00:00"}


def test_the_parser_knows_the_new_options():
    args = posture.build_parser().parse_args(
        ["study", "--before-baseline", "natural", "--conditions", "",
         "--desk-level", "seat-up", "--lift-cm", "5"])
    assert args.before_baseline == "natural"
    assert args.conditions == ""
    assert args.desk_level == "seat-up"
    defaults = posture.build_parser().parse_args(["study"])
    assert defaults.before_baseline is None
    assert defaults.conditions == "upright,forward"


# ---- 轉身的參考方位角 --------------------------------------------------

def _rec(condition, azimuths, before=False):
    session = Session()
    for _ in range(20):
        session.add((1.0, 1.0))
    return Recording(session=session, ca_window=RollingAngle(30),
                     sym_window=RollingAngle(30), rejected=0, frames=20,
                     condition=condition, trial=1, azimuths=list(azimuths),
                     before_baseline=before)


def test_natural_sitting_is_checked_against_the_same_trip_upright():
    segments = review([_rec("natural", [58.0], before=True),
                       _rec("upright", [44.0]), _rec("forward", [45.0])],
                      _baseline(azimuth_deg=40.0))
    assert segments[0].turned_deg == pytest.approx(14.0)   # 對 upright，不是對基準
    assert segments[0].turned_too_far
    assert segments[1].turned_deg == pytest.approx(4.0)    # 基準後照舊對基準


def test_without_an_upright_in_the_trip_natural_is_not_flagged():
    segments = review([_rec("natural", [70.0], before=True), _rec("forward", [40.0])],
                      _baseline(azimuth_deg=40.0))
    assert segments[0].turned_deg is None
    assert not segments[0].turned_too_far


def test_the_azimuth_trip_uses_its_first_segment_and_never_calls_it_a_turn():
    """趟 5 是刻意搬動相機，方位角差 25° 是設計，不是受試者轉身。"""
    segments = review([_rec("head-forward", [30.0], before=True),
                       _rec("head-forward", [45.0], before=True),
                       _rec("head-forward", [55.0], before=True)], None)
    assert [s.turned_deg for s in segments] == pytest.approx([0.0, 15.0, 25.0])
    assert not any(s.turned_too_far for s in segments)
    assert not any(s.needs_redo for s in segments)


# ---- 指導語 ------------------------------------------------------------

def test_the_new_briefs_exist_and_carry_no_markdown():
    generic = condition_brief("清單以外的名稱")
    for condition in ("natural", "tilt-left", "tilt-right"):
        brief = condition_brief(condition)
        assert brief != generic
        for mark in ("**", "__", "`"):
            assert mark not in brief
    assert "抄寫" in condition_brief("natural")
    assert "不要轉身" in condition_brief("tilt-left")
    assert condition_brief("tilt-left").startswith("請把上半身往左")
    assert condition_brief("tilt-right").startswith("請把上半身往右")


def test_the_old_briefs_are_untouched():
    assert condition_brief("upright").startswith("請坐正")
    assert condition_brief("head-back").startswith("肩膀不要動，把頭往後仰")


def test_old_csv_without_the_new_header_lines_still_reads(tmp_path):
    path = tmp_path / "old.csv"
    path.write_text("# subject=a\n# condition=upright\nframe,usable\n1,1\n",
                    encoding="utf-8")
    rows, meta = read_rows(path)
    assert "trip_id" not in meta and rows


# ---- 2026-11-18 GPT 審查補的情境 -----------------------------------------

def test_a_no_baseline_trip_with_an_upright_still_never_flags_a_turn():
    segments = review([_rec("head-forward", [30.0], before=True),
                       _rec("upright", [45.0], before=True),
                       _rec("head-forward", [55.0], before=True)], None)
    assert not any(s.turned_too_far for s in segments)
    assert not any(s.needs_redo for s in segments)


def test_a_retaken_upright_is_not_the_turn_reference():
    """upright-1 掉線、upright-2 重量；自然坐要對重量的那一份。"""
    lost = _rec("upright", [70.0])
    lost.camera_lost = "相機掉線"
    segments = review([_rec("natural", [44.0], before=True), lost,
                       _rec("upright", [45.0])], _baseline(azimuth_deg=45.0))
    assert segments[0].turned_deg == pytest.approx(-1.0)
    assert not segments[0].turned_too_far


def test_a_lost_segment_gets_a_camera_lost_line_at_the_end(tmp_path, monkeypatch):
    _patch(monkeypatch, _Events(lost_on={1}))
    posture._run_study(_args(tmp_path, conditions="upright"))
    folder = tmp_path / "s01"
    assert _meta(folder / "upright-1.csv").get("camera_lost") == "1"
    assert "camera_lost" not in _meta(folder / "upright-2.csv")
