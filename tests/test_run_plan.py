"""引導模式 `run-plan` 與上機前檢查 `preflight`（2026-11-22）。

2026-11-19 S01 的自測，問題幾乎都是操作，而且都是人走了之後才發現：側傾六段
沒有量角器數字、兩趟 desk_offset_cm 都是 10、基準和端正坐差十幾度沒有重取。
組員一次只來一小時，這些測試盯的是「當場擋下來」與「重量不弄亂檔名」。
"""
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from geometry.baseline import PostureBaseline
from geometry.preflight import FrameProbe, all_clear, git_items, probe_items, verdict_lines
from geometry.recording import Recording, RollingAngle, Session
from geometry.run_plan import (BASELINE_UPRIGHT_LIMIT_DEG, check_segment,
                               describe_offset, discard_dir, discard_segment,
                               estimated_minutes, load_plan, parse_offset,
                               parse_plan, trip_argv)
from geometry.run_plan import (ExistingSegment, resume_lines, select_trips,
                               trip_progress)
from geometry.study import SegmentReview, next_trial
from posture_loader import load_posture_module

posture = load_posture_module("posture_study")

PLAN = Path("plans/selftest.json")


# ---- 計畫檔 ------------------------------------------------------------

def _trip(**overrides):
    trip = {"id": "t1", "conditions": ["upright", "head-forward"]}
    trip.update(overrides)
    return trip


def test_the_shipped_plan_matches_the_selftest_manual():
    """計畫檔要與自測手冊的三條 study 指令一致，量測方法不能因為換成引導模式而變。"""
    plan = load_plan(PLAN)
    assert [t.trip_id for t in plan.trips] == ["trip1", "trip4", "tiltA"]
    trip1, trip4, tilt = plan.trips
    assert trip1.before == ("natural",)
    assert trip1.conditions == ("upright", "head-forward", "forward", "upright")
    assert trip4.conditions == ("upright", "left-shoulder-up", "right-shoulder-up")
    assert tilt.conditions == ("upright", "tilt-left", "tilt-right")
    assert tilt.target_angles == (20.0, 20.0)
    # 側傾全員做（2026-11-17 依指導老師意見定案），不能被跳過。
    assert not tilt.optional and not trip1.optional and not trip4.optional
    assert (plan.seconds_each, plan.countdown, plan.snapshot_raw) == (60.0, 20, True)


def test_each_trip_becomes_a_study_command_the_real_parser_accepts():
    """組成的參數交給同一個 parser，預設值只有一處。"""
    plan = load_plan(PLAN)
    parser = posture.build_parser()
    trip1 = plan.trips[0]
    ns = parser.parse_args(["study"] + trip_argv(plan, trip1, "S02", -3.5))
    assert ns.subject == "S02"
    assert ns.before_baseline == "natural"
    assert ns.conditions == "upright,head-forward,forward,upright"
    assert ns.desk_level == "base" and ns.desk_offset_cm == -3.5
    assert ns.seconds_each == 60 and ns.countdown == 20 and ns.snapshot_raw
    tilt = plan.trips[2]
    ns = parser.parse_args(["study"] + trip_argv(plan, tilt, "S02", None))
    assert ns.target_angle_deg == "20,20" and ns.desk_offset_cm is None


@pytest.mark.parametrize("data, words", [
    ([], "物件"),
    ({"trips": []}, "至少一趟"),
    ({"trips": [_trip(), _trip()]}, "重複"),
    ({"trips": [_trip(conditions=[], before=[])]}, "都是空的"),
    ({"trips": [_trip(conditions=["upright", "tilt-left"])]}, "target_angle_deg"),
    ({"trips": [_trip(conditions=["upright", "tilt-left"],
                      target_angle_deg=[20, 30])]}, "1 段側傾"),
    ({"trips": [_trip(conditions=["upright", "tilt-left"],
                      target_angle_deg=[True])]}, "數字"),
    ({"trips": [_trip(desk_level="higher")]}, "desk_level"),
    ({"trips": [_trip(conditions=["upright", " "])]}, "空的名稱"),
    ({"trips": [_trip()], "seconds_each": 0}, "正數"),
    # GPT 審查第 6 條：字串 "false" 經 bool() 會變成 True，意思反過來。
    ({"trips": [_trip()], "snapshot_raw": "false"}, "true 或 false"),
    ({"trips": [_trip(optional="false")]}, "true 或 false"),
    ({"trips": [_trip(task=123)]}, "task"),
])
def test_a_broken_plan_is_refused_before_the_camera_opens(data, words):
    with pytest.raises(ValueError, match=words):
        parse_plan(data)


def test_a_plan_that_is_not_json_says_which_line(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text('{"trips": [\n  {"id": }\n]}', encoding="utf-8")
    with pytest.raises(ValueError, match="第 2 行"):
        load_plan(path)


def test_the_time_estimate_counts_baseline_and_protractor():
    plan = load_plan(PLAN)
    trip1, trip4, tilt = plan.trips
    # 趟 1：5 段 × 90 秒 + 基準 50 秒 = 500 秒
    assert estimated_minutes(plan, trip1) == pytest.approx(500 / 60)
    # 側傾 A：3 段 × 90 + 50 + 2 段量角器 × 60 = 440 秒
    assert estimated_minutes(plan, tilt) == pytest.approx(440 / 60)


# ---- 當場檢查 ----------------------------------------------------------

def _review(**overrides):
    values = dict(condition="upright", trial=1, theta_ca_deg=1.0, usable=95,
                  frames=100, rejection_rate=0.05, turned_deg=2.0)
    values.update(overrides)
    return SegmentReview(**values)


def test_a_clean_segment_passes():
    assert check_segment(_review(), first_upright_after_baseline=True).ok


def test_too_many_rejected_frames_is_flagged():
    result = check_segment(_review(rejection_rate=0.30), first_upright_after_baseline=False)
    assert not result.ok and not result.retake_baseline
    assert "30%" in result.problems[0]


def test_turning_is_flagged():
    result = check_segment(_review(turned_deg=-17.0), first_upright_after_baseline=False)
    assert "轉身" in result.problems[0]


def test_baseline_mismatch_only_on_the_first_upright_after_the_baseline():
    """基準與接著那段端正坐差 5° 以上要重取基準。後面那段端正坐（趟 1 的漂移
    檢查）本來就可能漂，不能拿來推翻基準。"""
    far = BASELINE_UPRIGHT_LIMIT_DEG + 3
    first = check_segment(_review(theta_ca_deg=far), first_upright_after_baseline=True)
    assert first.retake_baseline and not first.ok
    later = check_segment(_review(theta_ca_deg=far), first_upright_after_baseline=False)
    assert later.ok
    edge = check_segment(_review(theta_ca_deg=BASELINE_UPRIGHT_LIMIT_DEG),
                         first_upright_after_baseline=True)
    assert edge.ok
    other = check_segment(_review(condition="head-forward", theta_ca_deg=25.0),
                          first_upright_after_baseline=True)
    assert other.ok


def test_a_camera_loss_is_left_to_study_itself():
    """掉線 study 自己會重開相機、同一段重量，這裡不再問一次。"""
    result = check_segment(_review(camera_lost=True, rejection_rate=0.9),
                           first_upright_after_baseline=True)
    assert result.ok


# ---- 作廢 --------------------------------------------------------------

def _segment(root, subject="S02", name="upright-1"):
    folder = root / subject
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / f"{name}.csv"
    csv_path.write_text("# x\n", encoding="utf-8")
    shots = folder / f"{name}-shots"
    (shots / "raw").mkdir(parents=True)
    (shots / "t10.jpg").write_bytes(b"jpg")
    return csv_path


def test_discarded_segments_go_outside_the_sessions_tree(tmp_path):
    """cohort 會遞迴找 data/sessions 底下所有 CSV，作廢的段落不能留在裡面。"""
    root = tmp_path / "sessions"
    assert root not in discard_dir(root, "S02").parents


def test_discarding_moves_the_csv_and_shots_and_frees_the_number(tmp_path):
    """重量的那段要沿用原本的編號，趟 1 的漂移檢查那段才會照舊叫 upright-2。"""
    root = tmp_path / "sessions"
    csv_path = _segment(root)
    moved = discard_segment(csv_path, root, "S02")
    assert not csv_path.exists()
    assert not (root / "S02" / "upright-1-shots").exists()
    assert moved.exists()
    assert (moved.parent / f"{moved.stem}-shots" / "t10.jpg").exists()
    assert next_trial(root, "S02", "upright") == 1
    assert not list(root.rglob("*.csv"))


def test_discarding_the_same_segment_twice_keeps_both(tmp_path):
    root = tmp_path / "sessions"
    first = discard_segment(_segment(root), root, "S02")
    second = discard_segment(_segment(root), root, "S02")
    assert first != second and first.exists() and second.exists()


# ---- 桌面減手肘 --------------------------------------------------------

@pytest.mark.parametrize("text, value", [
    ("5", 5.0), ("-3", -3.0), (" +2.5 ", 2.5), ("－4", -4.0), ("6cm", 6.0),
    ("", None), ("五", None),
])
def test_offset_input(text, value):
    assert parse_offset(text) == value


def test_offset_is_read_back_in_words():
    assert describe_offset(5) == "桌面比手肘高 5 公分"
    assert describe_offset(-3) == "桌面比手肘低 3 公分"
    assert describe_offset(0) == "桌面和手肘一樣高"


# ---- 整個流程 ----------------------------------------------------------

def _baseline() -> PostureBaseline:
    return PostureBaseline(
        subject="S02", captured_at="2026-11-23T10:00:00", frames=149, rejected=0,
        duration_s=30.0, theta_ca_deg=3.0, theta_sym_deg=1.0,
        theta_ca_std_deg=4.5, theta_sym_std_deg=0.5,
        theta_ca_standard_error_deg=0.8, theta_sym_standard_error_deg=0.1,
        distance_mm=620.0, azimuth_deg=40.0)


class _FakeCap:
    def __init__(self):
        self.released = 0

    def release(self):
        self.released += 1


class _Studio:
    """換掉相機、模型與錄製。`bad` 是 {第幾次錄製: (略過比例, θ_CA)}。"""

    def __init__(self, bad=None):
        self.bad = bad or {}
        self.records = []      # (檔名, 有沒有基準)
        self.quiet = []        # 不印逐幀那一行的段落檔名
        self.baselines = 0
        self.cap = _FakeCap()

    def collect(self, engine, calib, cap, args, seconds, display=None):
        self.baselines += 1
        return _baseline(), []

    def record(self, engine, calib, cap, args, baseline, log, snapshots,
               seconds, display=None, quiet=False):
        self.records.append((log.path.name, baseline is not None))
        if quiet:
            self.quiet.append(log.path.name)
        rate, ca = self.bad.get(len(self.records), (0.02, 1.0))
        session = Session()
        for _ in range(98):
            session.add((ca, 1.0))
        rejected = int(round(rate * 100))
        return Recording(session=session, ca_window=RollingAngle(30),
                         sym_window=RollingAngle(30), rejected=rejected,
                         frames=98 + rejected, log_path=log.path,
                         azimuths=[40.0] * 98)


def _patch(monkeypatch, studio, answers, probes=None):
    """`probes` 是預檢每一輪回傳的項目清單（照順序用，用完重複最後一份）。
    不給就是一切正常。"""
    calib = SimpleNamespace(baseline_mm=60.14)
    monkeypatch.setattr(posture, "_prepare", lambda args, **kw: (calib, None, studio.cap))
    monkeypatch.setattr(posture, "_git_state", lambda: ("a" * 40, "a" * 40, False))
    rounds = list(probes) if probes else [[]]
    studio.preflight_rounds = 0

    def fake_probe(engine, calib, cap, args, frames, warmup):
        studio.preflight_rounds += 1
        return rounds.pop(0) if len(rounds) > 1 else rounds[0]

    monkeypatch.setattr(posture, "_probe_items", fake_probe)
    monkeypatch.setattr(posture, "_collect_baseline", studio.collect)
    monkeypatch.setattr(posture, "_record", studio.record)
    monkeypatch.setattr(posture, "_report", lambda *a, **k: None)
    monkeypatch.setattr(posture, "_countdown", lambda *a, **k: None)
    monkeypatch.setattr(posture, "_reopen_camera", lambda args, **kw: _FakeCap())
    analysed = []
    monkeypatch.setattr(posture, "_run_analyse",
                        lambda a: (analysed.append(list(a.csv)),
                                   print("ANALYSE OUTPUT")))
    asked = []

    def fake_input(prompt=""):
        asked.append(prompt)
        # 先比對準備好的回答：「重新檢查」那句也含「按 Enter」，不能被吃掉。
        for key, replies in answers.items():
            if key in prompt:
                if not replies:
                    raise AssertionError(f"沒有準備「{prompt}」的回答")
                return replies.pop(0)
        if "按 Enter" in prompt:
            return ""
        raise AssertionError(f"沒料到的問題：{prompt}")

    monkeypatch.setattr("builtins.input", fake_input)
    return asked, analysed


def _skippable_tilt_plan(tmp_path):
    """正式計畫檔的副本，只把側傾改成可跳過。

    多數流程測試關心的是趟 1、趟 4，回 n 跳過側傾才不用每個都準備四個量角器
    讀數。正式計畫檔的側傾是必做的，見 test_the_shipped_plan_matches_the_selftest_manual。
    """
    data = json.loads(PLAN.read_text(encoding="utf-8"))
    data["trips"][2]["optional"] = True
    path = tmp_path / "skippable-tilt.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def _plan_args(tmp_path, plan_path=None, **overrides):
    if plan_path is None:
        plan_path = _skippable_tilt_plan(tmp_path)
    values = dict(
        mode="run-plan", plan=plan_path, subject="S02", out_dir=tmp_path / "sessions",
        dry_run=False, calibration=Path("x.npz"), camera=0, width=None, height=None,
        vertical_split=False, swap_lr=False, discard=0, save_frames=None,
        precision="fp32", device="cpu", checkpoint=Path("c"), engine_cache=Path("e"),
        repo_dir=Path("r"), min_confidence=0.0, input_height=256, no_subpixel=False,
        display=False, display_width=1280, display_eyes="left", display_lang="en",
        font=None, window=30, all_keypoints=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _names(root):
    return sorted(p.name for p in (root / "S02").glob("*.csv"))


def test_a_clean_run_records_every_trip_in_order_and_analyses(tmp_path, monkeypatch, capsys):
    studio = _Studio()
    answers = {"桌面高減手肘高": ["5"], "對嗎": ["y"],
               "量角器": ["21", "19", "20", "22"]}
    asked, analysed = _patch(monkeypatch, studio, answers)
    posture._run_plan(_plan_args(tmp_path, PLAN))
    root = tmp_path / "sessions"
    assert [r[0] for r in studio.records] == [
        "natural-1.csv", "upright-1.csv", "head-forward-1.csv", "forward-1.csv",
        "upright-2.csv", "upright-3.csv", "left-shoulder-up-1.csv",
        "right-shoulder-up-1.csv", "upright-4.csv", "tilt-left-1.csv",
        "tilt-right-1.csv"]
    assert studio.baselines == 3
    assert studio.cap.released >= 1
    # 量角器的開始值與結束值都寫進去了。
    tilt = (root / "S02" / "tilt-left-1.csv").read_text(encoding="utf-8")
    assert "measured_angle_deg=21" in tilt and "measured_end_angle_deg=19" in tilt
    # 桌面減手肘每一趟都有。
    for name in ("natural-1.csv", "left-shoulder-up-1.csv", "tilt-right-1.csv"):
        assert "desk_offset_cm=5" in (root / "S02" / name).read_text(encoding="utf-8")
    assert len(analysed) == 1 and len(analysed[0]) == 11
    saved = list((root / "S02").glob("analyse-*.txt"))
    assert saved and "ANALYSE OUTPUT" in saved[0].read_text(encoding="utf-8")
    out = capsys.readouterr().out
    assert "當場檢查：沒過" not in out
    assert "AI 會自己用 ssh 拉資料" in out
    assert not any("要做嗎" in q for q in asked), "正式計畫檔的側傾不能問要不要跳過"
    assert "可跳過" not in out
    assert not any(answers.values())


def test_the_typing_segment_shows_the_text_in_the_terminal_and_stays_quiet(
        tmp_path, monkeypatch, capsys):
    """selftest 的自然坐是 type-english：英文印在終端機，受試者直接在終端機打。
    量完把沒讀走的輸入丟掉，免得被下一個問題當成回答。
    run-plan 每一段都不印逐幀那一行（有角度，受試者看得到會跟著調），
    所以也不必叫操作者轉開螢幕。"""
    studio = _Studio()
    answers = {"桌面高減手肘高": ["5"], "對嗎": ["y"], "要做嗎": ["n"]}
    _patch(monkeypatch, studio, answers)
    flushed = []
    monkeypatch.setattr(posture, "_discard_typed_input", lambda: flushed.append(1))
    posture._run_plan(_plan_args(tmp_path))
    assert studio.quiet == [name for name, _ in studio.records]
    assert flushed == [1]
    out = capsys.readouterr().out
    assert "Every morning, the streets of the city" in out
    assert out.count("Every morning") == 1
    assert "請直接在下面打字" in out
    assert "把螢幕轉" not in out
    assert out.count("量測中，請保持姿勢") == len(studio.records) - 1


def test_typing_text_wraps_to_the_terminal_without_losing_words(monkeypatch, capsys):
    monkeypatch.setattr(posture.shutil, "get_terminal_size",
                        lambda fallback=None: os.terminal_size((60, 24)))
    posture._show_typing_text()
    lines = capsys.readouterr().out.splitlines()
    body = lines[3:-1]
    assert all(len(line) <= 59 for line in body)
    assert " ".join(body) == posture.TYPING_TEXT


def test_discarding_typed_input_does_nothing_when_stdin_is_not_a_terminal(monkeypatch):
    monkeypatch.setattr(posture.sys, "stdin", io.StringIO("leftover"))
    posture._discard_typed_input()
    assert posture.sys.stdin.read() == "leftover"


def test_a_bad_segment_is_redone_once_under_the_same_name(tmp_path, monkeypatch, capsys):
    """第 3 次錄製（head-forward）略過 30%，重量；重量那份沿用 head-forward-1。"""
    studio = _Studio(bad={3: (0.30, 1.0)})
    answers = {"桌面高減手肘高": ["0"], "對嗎": ["y"], "要做嗎": ["n"],
               "要重量這一段嗎": ["y"]}
    _patch(monkeypatch, studio, answers)
    posture._run_plan(_plan_args(tmp_path))
    root = tmp_path / "sessions"
    names = [r[0] for r in studio.records]
    assert names[:4] == ["natural-1.csv", "upright-1.csv", "head-forward-1.csv",
                         "head-forward-1.csv"]
    assert "upright-2.csv" in _names(root)
    discarded = list(discard_dir(root, "S02").glob("*.csv"))
    assert [p.name for p in discarded] == ["head-forward-1.csv"]
    out = capsys.readouterr().out
    assert "當場檢查：沒過" in out and "跳過這一趟" in out


def test_a_segment_that_fails_twice_is_kept_and_not_asked_again(tmp_path, monkeypatch, capsys):
    """重量一次還是不過就記下來繼續，不讓下一位受試者一直等。"""
    studio = _Studio(bad={3: (0.30, 1.0), 4: (0.30, 1.0)})
    answers = {"桌面高減手肘高": ["0"], "對嗎": ["y"], "要做嗎": ["n"],
               "要重量這一段嗎": ["y"]}
    _patch(monkeypatch, studio, answers)
    posture._run_plan(_plan_args(tmp_path))
    names = [r[0] for r in studio.records]
    assert names.count("head-forward-1.csv") == 2
    assert "head-forward-2.csv" not in names
    assert "已經重量過" in capsys.readouterr().out


def test_a_baseline_that_disagrees_with_the_next_upright_is_retaken(tmp_path, monkeypatch):
    """第 2 次錄製（取基準後第一段端正坐）扣基準後 +9°：重取基準，再量一次 upright-1。"""
    studio = _Studio(bad={2: (0.02, 9.0)})
    answers = {"桌面高減手肘高": ["2"], "對嗎": ["y"], "要做嗎": ["n"],
               "重取基準並重量": ["y"]}
    _patch(monkeypatch, studio, answers)
    posture._run_plan(_plan_args(tmp_path))
    names = [r[0] for r in studio.records]
    assert names[:3] == ["natural-1.csv", "upright-1.csv", "upright-1.csv"]
    # 趟 1 兩次基準、趟 4 一次。
    assert studio.baselines == 3
    # 自然坐只量一次，不因為重取基準而重量。
    assert names.count("natural-1.csv") == 1


def test_the_operator_can_decline_a_redo(tmp_path, monkeypatch, capsys):
    studio = _Studio(bad={3: (0.30, 1.0)})
    answers = {"桌面高減手肘高": ["0"], "對嗎": ["y"], "要做嗎": ["n"],
               "要重量這一段嗎": ["n"]}
    _patch(monkeypatch, studio, answers)
    posture._run_plan(_plan_args(tmp_path))
    assert [r[0] for r in studio.records].count("head-forward-1.csv") == 1
    assert "不重量，記下來" in capsys.readouterr().out


def test_a_wrong_offset_can_be_corrected_before_anything_runs(tmp_path, monkeypatch):
    studio = _Studio()
    answers = {"桌面高減手肘高": ["abc", "10", "-10"], "對嗎": ["n", "y"],
               "要做嗎": ["n"]}
    _patch(monkeypatch, studio, answers)
    posture._run_plan(_plan_args(tmp_path))
    text = (tmp_path / "sessions" / "S02" / "upright-1.csv").read_text(encoding="utf-8")
    assert "desk_offset_cm=-10" in text


def test_the_protractor_cannot_be_skipped_by_pressing_enter(tmp_path, monkeypatch, capsys):
    """S01 六段都沒有量角器數字，因為按 Enter 就能留空。"""
    plan = tmp_path / "tilt.json"
    plan.write_text(json.dumps({"trips": [{
        "id": "tilt", "conditions": ["upright", "tilt-left"],
        "target_angle_deg": [20]}]}), encoding="utf-8")
    studio = _Studio()
    # 確認句裡也有「量角器」，要排在前面先比對。
    answers = {"確定這一段不記量角器": ["n", "y"], "量角器": ["", "abc", "18", "-", "-"]}
    _patch(monkeypatch, studio, answers)
    posture._run_plan(_plan_args(tmp_path, plan_path=plan))
    text = (tmp_path / "sessions" / "S02" / "tilt-left-1.csv").read_text(encoding="utf-8")
    assert "measured_angle_deg=18" in text
    assert "measured_end_angle_deg" not in text
    out = capsys.readouterr().out
    assert "一定要填" in out and "不是數字" in out


def test_plain_study_still_lets_the_protractor_be_left_blank(tmp_path, monkeypatch):
    """study 本身不改行為：沒有 require_angles 時按 Enter 照舊留空。"""
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    assert posture._ask_tilt_angle("幾度？", SimpleNamespace()) is None


def test_dry_run_does_not_open_the_camera(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(posture, "_prepare",
                        lambda *a, **k: pytest.fail("dry-run 不該開相機"))
    posture._run_plan(_plan_args(tmp_path, dry_run=True))
    out = capsys.readouterr().out
    assert "python posture.py study --subject S02" in out
    assert "--target-angle-deg 20,20" in out


@pytest.mark.parametrize("code", ["../S02", "..", ".", "S 02", "S02/x", ""])
def test_a_subject_code_that_is_not_a_plain_name_is_refused(tmp_path, code):
    """代號會變成資料夾名稱。「..」沒有斜線也會寫到 data/sessions 的上一層（GPT 審查第 2 條）。"""
    with pytest.raises(SystemExit, match="只能用英文"):
        posture._run_plan(_plan_args(tmp_path, subject=code, dry_run=True))


def test_no_keyboard_stops_instead_of_leaving_the_offset_blank(tmp_path, monkeypatch):
    """讀不到 stdin 時默默留空，正是 run-plan 要擋的事（GPT 審查第 4 條）。"""
    studio = _Studio()
    _patch(monkeypatch, studio, {})

    def no_stdin(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", no_stdin)
    with pytest.raises(SystemExit, match="讀不到鍵盤"):
        posture._run_plan(_plan_args(tmp_path))
    assert studio.records == []


def test_no_keyboard_stops_at_a_required_protractor_reading(monkeypatch):
    def no_stdin(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", no_stdin)
    with pytest.raises(SystemExit, match="讀不到鍵盤"):
        posture._ask_tilt_angle("幾度？", SimpleNamespace(require_angles=True))


def test_a_camera_loss_in_run_plan_reuses_the_same_name(tmp_path, monkeypatch, capsys):
    """掉線的殘段搬走，重量那段照舊叫 head-forward-1（GPT 審查第 1 條）。"""
    studio = _Studio()
    original = studio.record

    def lose_third(*a, **k):
        recording = original(*a, **k)
        if len(studio.records) == 3:
            from dataclasses import replace
            return replace(recording, camera_lost="相機連續 25 次讀不到畫面")
        return recording

    answers = {"桌面高減手肘高": ["0"], "對嗎": ["y"], "要做嗎": ["n"]}
    _patch(monkeypatch, studio, answers)
    monkeypatch.setattr(posture, "_record", lose_third)
    posture._run_plan(_plan_args(tmp_path))
    root = tmp_path / "sessions"
    names = [r[0] for r in studio.records]
    assert names[2:4] == ["head-forward-1.csv", "head-forward-1.csv"]
    assert "head-forward-2.csv" not in _names(root)
    assert [p.name for p in discard_dir(root, "S02").glob("*.csv")] == ["head-forward-1.csv"]
    assert "相機掉線" in capsys.readouterr().out


def test_plain_study_keeps_the_lost_segment_and_numbers_on(tmp_path, monkeypatch):
    """study 不給 on_camera_lost，掉線殘段照舊留著，重量那段接續編號。"""
    studio = _Studio()
    original = studio.record

    def lose_first(*a, **k):
        recording = original(*a, **k)
        if len(studio.records) == 1:
            from dataclasses import replace
            return replace(recording, camera_lost="掉了")
        return recording

    _patch(monkeypatch, studio, {})
    monkeypatch.setattr(posture, "_record", lose_first)
    args = _plan_args(tmp_path)
    ns = posture._plan_trip_args(posture.build_parser(), args, load_plan(PLAN),
                                 load_plan(PLAN).trips[1], None)
    ns.no_prompt = True
    posture._run_study(ns)
    assert [r[0] for r in studio.records][:2] == ["upright-1.csv", "upright-2.csv"]


def test_a_turned_natural_segment_is_pointed_out_after_the_first_upright(
        tmp_path, monkeypatch, capsys):
    """自然坐量的當下沒有參考，量完第一段端正坐才看得出轉身（GPT 審查第 3 條，只提醒）。"""
    studio = _Studio()
    original = studio.record

    def turned_natural(*a, **k):
        recording = original(*a, **k)
        if len(studio.records) == 1:
            from dataclasses import replace
            return replace(recording, azimuths=[60.0] * 98)
        return recording

    answers = {"桌面高減手肘高": ["0"], "對嗎": ["y"], "要做嗎": ["n"]}
    _patch(monkeypatch, studio, answers)
    monkeypatch.setattr(posture, "_record", turned_natural)
    posture._run_plan(_plan_args(tmp_path))
    out = capsys.readouterr().out
    assert "natural #1 的方位角和這段端正坐差 +20°" in out
    assert [r[0] for r in studio.records].count("natural-1.csv") == 1


def test_a_display_that_fails_to_open_still_releases_the_camera(tmp_path, monkeypatch):
    """GPT 審查第 5 條：視窗開不起來時相機也要放掉。"""
    studio = _Studio()
    _patch(monkeypatch, studio, {"桌面高減手肘高": ["0"], "對嗎": ["y"]})

    def no_window(args):
        raise RuntimeError("沒有圖形環境")

    monkeypatch.setattr(posture, "_open_display", no_window)
    with pytest.raises(RuntimeError):
        posture._run_plan(_plan_args(tmp_path))
    assert studio.cap.released >= 1


def test_plain_study_writes_the_trip_id_from_before_the_camera_opened(tmp_path, monkeypatch):
    """GPT 審查第 8 條：trip_id 只建一次，是開相機之前那一刻。"""
    studio = _Studio()
    _patch(monkeypatch, studio, {})
    built = []
    real = posture._trip_options
    monkeypatch.setattr(posture, "_trip_options",
                        lambda args, segments: built.append(1) or real(args, segments))
    args = _plan_args(tmp_path)
    ns = posture._plan_trip_args(posture.build_parser(), args, load_plan(PLAN),
                                 load_plan(PLAN).trips[1], None)
    ns.no_prompt = True
    posture._run_study(ns)
    assert len(built) == 1


def test_an_error_mid_trip_tells_the_operator_what_to_do(tmp_path, monkeypatch, capsys):
    studio = _Studio()
    answers = {"桌面高減手肘高": ["0"], "對嗎": ["y"]}
    _patch(monkeypatch, studio, answers)

    def boom(*a, **k):
        raise RuntimeError("左眼沒有偵測到人")

    monkeypatch.setattr(posture, "_record", boom)
    with pytest.raises(RuntimeError):
        posture._run_plan(_plan_args(tmp_path))
    assert "拍照傳給使用者" in capsys.readouterr().out
    assert studio.cap.released >= 1


def test_the_parser_knows_both_new_commands():
    parser = posture.build_parser()
    args = parser.parse_args(["run-plan", "plans/selftest.json", "--subject", "S03",
                              "--dry-run"])
    assert args.subject == "S03" and args.dry_run
    args = parser.parse_args(["preflight", "--frames", "5"])
    assert args.frames == 5
    assert not hasattr(args, "display")
    with pytest.raises(SystemExit):
        parser.parse_args(["run-plan", "plans/selftest.json"])  # 少了 --subject


# ---- 上機前檢查 --------------------------------------------------------

def test_an_outdated_machine_blocks_the_start():
    items = git_items("a" * 40, "b" * 40, False)
    assert not all_clear(items)
    assert "git pull" in "\n".join(verdict_lines(items))


def test_no_network_is_only_a_warning():
    items = git_items("a" * 40, None, False)
    assert all_clear(items)
    assert "[注意]" in "\n".join(verdict_lines(items))


def test_up_to_date_and_clean_passes():
    items = git_items("a" * 40, "a" * 40, False)
    assert all(i.ok for i in items)


def _probe(**overrides):
    values = dict(shared_count=15, angle_disparity_px=1.5, depth_mm=620.0, unusable=None)
    values.update(overrides)
    return FrameProbe(**values)


def test_good_frames_pass():
    items = probe_items([_probe() for _ in range(10)])
    assert all_clear(items)
    assert "結論：可以開始" in "\n".join(verdict_lines(items))


def test_nobody_in_front_of_the_camera_blocks():
    items = probe_items([FrameProbe(error="左眼沒有偵測到人")] * 10)
    assert not all_clear(items)
    assert "左眼沒有偵測到人" in items[0].detail


def test_sitting_too_far_says_which_way_to_move():
    items = probe_items([_probe(depth_mm=780.0) for _ in range(10)])
    distance = [i for i in items if i.label == "受試者距離"][0]
    assert not distance.ok and "太遠" in distance.advice


def test_few_shared_keypoints_blocks():
    items = probe_items([_probe(shared_count=9) for _ in range(10)])
    assert not all_clear(items)


def test_too_many_unusable_frames_blocks_and_names_the_reason():
    probes = [_probe() for _ in range(7)] + [
        _probe(unusable="left_shoulder 的垂直視差 9.0px，左右配對錯了")] * 3
    items = probe_items(probes)
    usable = [i for i in items if i.label == "可用幀"][0]
    assert not usable.ok and "配對錯" in usable.detail


def test_disparity_above_target_is_a_warning_and_above_the_limit_blocks():
    warn = probe_items([_probe(angle_disparity_px=5.0) for _ in range(10)])
    assert all_clear(warn)
    assert any(i.label == "垂直視差" and i.warning_only for i in warn)
    # 中位數超過 8px 的幀本來就會被 unusable_reason 擋掉，這裡直接驗判斷本身。
    block = probe_items([_probe(angle_disparity_px=9.0) for _ in range(10)])
    assert not all_clear(block)


def test_preflight_tells_a_camera_read_failure_apart_from_nobody_there(monkeypatch, capsys):
    """GPT 審查第 7 條：試拍時相機讀不到，要說「相機」沒過，不是叫受試者坐好。"""
    import numpy as np
    from calibration.device import CameraReadError

    cap = _FakeCap()
    calib = SimpleNamespace(baseline_mm=60.14, image_size=(1280, 720))
    monkeypatch.setattr(posture, "_git_state", lambda: ("a" * 40, "a" * 40, False))
    monkeypatch.setattr(posture.StereoCalibrationResult, "load", lambda path: calib)
    monkeypatch.setattr(posture, "_build_engine",
                        lambda args: SimpleNamespace(warmup=lambda frame: None))
    monkeypatch.setattr(posture, "_open_selected_camera", lambda args, c: cap)
    monkeypatch.setattr(posture, "_discard_frames", lambda c, n: None)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    monkeypatch.setattr(posture, "_grab_pair", lambda c, a: (frame, frame))
    monkeypatch.setattr(posture, "describe_resolution_mismatch", lambda *a: None)

    def unplugged(*a):
        raise CameraReadError("讀取相機影格失敗")

    monkeypatch.setattr(posture, "_measure_once", unplugged)
    args = SimpleNamespace(calibration=Path("x.npz"), discard=0, vertical_split=False,
                           precision="fp32", device="cpu", frames=5)
    with pytest.raises(SystemExit):
        posture._run_preflight(args)
    out = capsys.readouterr().out
    assert "[沒過] 相機：讀不到畫面" in out
    assert "偵測到受試者" not in out
    assert cap.released == 1


# ---- 一行指令：內建預檢、畫面紀錄、預設計畫 ------------------------------

def _bad_item():
    from geometry.preflight import Item
    return Item(False, "受試者距離", "780mm（要 550～700mm）", "受試者離相機太遠")


def test_run_plan_checks_the_setup_before_asking_anything_else(tmp_path, monkeypatch, capsys):
    """操作者只打一行指令，預檢要在量之前自己跑，而且排在問桌高之前（量桌高要人坐著）。"""
    studio = _Studio()
    answers = {"桌面高減手肘高": ["0"], "對嗎": ["y"], "要做嗎": ["n"]}
    asked, _ = _patch(monkeypatch, studio, answers)
    posture._run_plan(_plan_args(tmp_path))
    out = capsys.readouterr().out
    assert "上機前檢查" in out and "結論：可以開始" in out
    assert out.index("結論：可以開始") < out.index("當場檢查")
    assert studio.preflight_rounds == 1
    assert [q for q in asked if "桌面高減手肘高" in q]


def test_a_failed_setup_blocks_until_it_is_fixed(tmp_path, monkeypatch, capsys):
    """第一輪距離太遠、操作者調整後按 Enter 重測，第二輪通過才開始量。"""
    studio = _Studio()
    answers = {"桌面高減手肘高": ["0"], "對嗎": ["y"], "要做嗎": ["n"]}
    _patch(monkeypatch, studio, answers, probes=[[_bad_item()], []])
    posture._run_plan(_plan_args(tmp_path))
    out = capsys.readouterr().out
    assert studio.preflight_rounds == 2
    assert "[沒過] 受試者距離" in out and "結論：還不能開始" in out
    assert "結論：可以開始" in out
    assert studio.records


def test_the_operator_can_quit_from_a_failed_setup(tmp_path, monkeypatch):
    studio = _Studio()
    answers = {"重新檢查": ["q"]}
    _patch(monkeypatch, studio, answers, probes=[[_bad_item()]])
    with pytest.raises(SystemExit, match="沒有量任何資料"):
        posture._run_plan(_plan_args(tmp_path))
    assert studio.records == [] and studio.baselines == 0


def test_the_screen_is_recorded_for_remote_watchers(tmp_path, monkeypatch):
    """遠端的人 tail -f data/logs/<代號>.log 看得到組員的畫面，含程式問的問題。"""
    studio = _Studio()
    answers = {"桌面高減手肘高": ["0"], "對嗎": ["y"], "要做嗎": ["n"]}
    _patch(monkeypatch, studio, answers)

    inner = __import__("builtins").input

    def writing_input(prompt=""):
        # 真的 input() 發現 stdout 沒有 fileno 時，會把提示寫進 sys.stdout。
        import sys
        sys.stdout.write(prompt)
        return inner(prompt)

    monkeypatch.setattr("builtins.input", writing_input)
    posture._run_plan(_plan_args(tmp_path))
    log = (tmp_path / "logs" / "S02.log").read_text(encoding="utf-8")
    assert "上機前檢查" in log
    assert "桌面高減手肘高是幾公分" in log
    assert "當場檢查：通過" in log
    assert "AI 會自己用 ssh 拉資料" in log


def test_running_again_with_the_same_code_appends_to_the_log(tmp_path):
    path = tmp_path / "logs" / "S02.log"
    for text in ("first", "second"):
        with posture._tee_stdout(path):
            print(text)
    log = path.read_text(encoding="utf-8")
    assert "first" in log and "second" in log and log.count("=====") >= 4


def test_stdout_is_restored_after_the_recording(tmp_path):
    import sys
    before = sys.stdout
    with posture._tee_stdout(tmp_path / "x.log"):
        assert sys.stdout is not before
        # 沒有 fileno，input() 才會把提示寫進紀錄而不是直接寫到終端機。
        with pytest.raises(OSError):
            sys.stdout.fileno()
    assert sys.stdout is before


def test_the_plan_defaults_to_the_selftest_file():
    args = posture.build_parser().parse_args(["run-plan", "--subject", "S03"])
    assert args.plan == Path("plans/selftest.json")


# ---- 只跑幾趟、接續（2026-11-23）--------------------------------------------

def test_trips_are_picked_by_id_in_plan_order():
    plan = load_plan(PLAN)
    assert [t.trip_id for t in select_trips(plan, "tiltA, trip4")] == ["trip4", "tiltA"]
    with pytest.raises(ValueError, match="沒有 trip9"):
        select_trips(plan, "trip9")
    with pytest.raises(ValueError, match="空的"):
        select_trips(plan, " , ")


def _like_1123():
    """1123 那天：趟 1、趟 4 量完，右肩高略過 17%，側傾還沒開始就中斷。"""
    rate = {"right-shoulder-up": 0.17}
    names = ["natural", "upright", "head-forward", "forward", "upright", "upright",
             "left-shoulder-up", "right-shoulder-up"]
    seen = {}
    out = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        out.append(ExistingSegment(name, seen[name], rate.get(name, 0.02)))
    return out


def test_progress_tells_done_trips_from_ones_to_redo_and_ones_not_started():
    plan = load_plan(PLAN)
    trip1, trip4, tilt = trip_progress(plan, _like_1123())
    assert trip1.done and not trip1.high_skip
    assert trip4.done and trip4.high_skip == ("right-shoulder-up #1 略過 17%",)
    assert not tilt.started and tilt.missing == ("tilt-left", "tilt-right")
    lines = resume_lines(plan, [trip1, trip4, tilt], "1123",
                         "python posture.py run-plan --subject 1123")
    text = "\n".join(lines)
    assert "  python posture.py run-plan --subject 1123 --trips tiltA" in lines
    assert "  python posture.py run-plan --subject 1123 --trips trip4,tiltA" in lines
    assert "建議整趟重量" in text and "自然坐已經量過" in text


def test_a_retaken_segment_counts_by_its_latest_trial():
    plan = load_plan(PLAN)
    segs = _like_1123() + [ExistingSegment("right-shoulder-up", 2, 0.03)]
    assert not trip_progress(plan, segs)[1].high_skip


def _write_segment(folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text("# subject=1123\n", encoding="utf-8")


def test_existing_data_without_trips_stops_before_the_camera(tmp_path, monkeypatch, capsys):
    folder = tmp_path / "sessions" / "S02"
    for name in ("natural-1.csv", "upright-1.csv", "head-forward-1.csv", "forward-1.csv"):
        _write_segment(folder, name)
    opened = []
    monkeypatch.setattr(posture, "_prepare", lambda *a, **k: opened.append(1))
    with pytest.raises(SystemExit, match="沒有開相機"):
        posture._run_plan_body(_plan_args(tmp_path, PLAN))
    assert not opened
    out = capsys.readouterr().out
    assert "S02 已經有量過的資料" in out
    assert "--trips trip4,tiltA" in out


def test_trips_runs_only_the_named_trip_and_keeps_numbering(tmp_path, monkeypatch, capsys):
    folder = tmp_path / "sessions" / "S02"
    for name in ("natural-1.csv", "upright-1.csv", "upright-2.csv", "upright-3.csv",
                 "left-shoulder-up-1.csv", "right-shoulder-up-1.csv"):
        _write_segment(folder, name)
    studio = _Studio()
    answers = {"桌面高減手肘高": ["0"], "對嗎": ["y"], "量角器": ["20", "20", "20", "20"]}
    _patch(monkeypatch, studio, answers)
    posture._run_plan(_plan_args(tmp_path, PLAN, trips="tiltA"))
    assert [r[0] for r in studio.records] == [
        "upright-4.csv", "tilt-left-1.csv", "tilt-right-1.csv"]
    out = capsys.readouterr().out
    assert "共 1 趟" in out
    assert not any(answers.values())
