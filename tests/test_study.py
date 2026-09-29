"""`study` 子指令：一個行程跑完整套實驗流程。

存在的理由是流程本身就是資料品質的一部分。2026-09-29 那晚四次量測作廢三次，
沒有一次是程式算錯：解析度不符、基準隔了九分鐘、量測中轉頭看螢幕、受試者
不自覺前傾。這些測試盯的是那些失敗還會不會再發生。
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from geometry.baseline import PostureBaseline
from posture_loader import load_posture_module

posture = load_posture_module("posture_study")


def _baseline(**overrides) -> PostureBaseline:
    values = dict(
        subject="chenyue", captured_at="2026-09-29T20:06:12", frames=149, rejected=0,
        duration_s=30.0, theta_ca_deg=3.78, theta_sym_deg=5.30,
        theta_ca_std_deg=4.5, theta_sym_std_deg=0.5,
        theta_ca_standard_error_deg=0.83, theta_sym_standard_error_deg=0.09,
        distance_mm=685.0, azimuth_deg=43.0,
    )
    values.update(overrides)
    return PostureBaseline(**values)


def _args(tmp_path, **overrides):
    values = dict(
        subject="chenyue", conditions="upright,forward", seconds=30.0,
        seconds_each=90.0, countdown=0, out_dir=tmp_path, no_prompt=True,
        overwrite=True, window=30, margin=1.0, no_judge=False,
        snapshot_every=10.0, save_frames=None, discard=0,
        calibration=Path("x.npz"), camera=0, width=None, height=None,
        vertical_split=False, swap_lr=False, precision="fp32", device="cpu",
        checkpoint=Path("c"), engine_cache=Path("e"), repo_dir=Path("r"),
        min_confidence=0.0, input_height=256, no_subpixel=False, mode="study",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


# ---- 檔名與批次編號 ----------------------------------------------------

def test_the_paths_say_who_what_and_which_attempt(tmp_path):
    csv_path, shots, baseline = posture.session_paths(
        tmp_path, "chenyue", "upright", 2
    )
    assert csv_path == tmp_path / "chenyue" / "upright-2.csv"
    assert shots == tmp_path / "chenyue" / "upright-2-shots"
    assert baseline == tmp_path / "chenyue" / "baseline.json"


def test_the_trial_number_continues_from_what_is_already_there(tmp_path):
    """撞名是 2026-09-29 真的發生過的事，而當時受試者正坐著等。"""
    folder = tmp_path / "chenyue"
    folder.mkdir()
    assert posture._next_trial(tmp_path, "chenyue", "upright") == 1
    (folder / "upright-1.csv").write_text("x", encoding="utf-8")
    (folder / "upright-2.csv").write_text("x", encoding="utf-8")
    assert posture._next_trial(tmp_path, "chenyue", "upright") == 3


def test_each_condition_counts_separately(tmp_path):
    folder = tmp_path / "chenyue"
    folder.mkdir()
    (folder / "upright-1.csv").write_text("x", encoding="utf-8")
    assert posture._next_trial(tmp_path, "chenyue", "forward") == 1


def test_a_missing_folder_starts_at_one(tmp_path):
    assert posture._next_trial(tmp_path, "nobody", "upright") == 1


def test_a_stray_filename_does_not_break_the_numbering(tmp_path):
    """資料夾裡本來就可能有別的東西，不該因此當掉。"""
    folder = tmp_path / "chenyue"
    folder.mkdir()
    (folder / "upright-notes.csv").write_text("x", encoding="utf-8")
    (folder / "upright-3.csv").write_text("x", encoding="utf-8")
    assert posture._next_trial(tmp_path, "chenyue", "upright") == 4


# ---- 指導語 ------------------------------------------------------------

def test_the_known_conditions_carry_the_rule_that_was_learned_the_hard_way():
    """量測中轉頭看螢幕會把上半身帶過去，方位角跟著漂，那次資料作廢。"""
    assert "不要看螢幕" in posture._condition_brief("upright")


def test_an_unnamed_condition_still_gets_something_usable():
    """姿勢種類還沒定案，程式不該把清單釘死。"""
    brief = posture._condition_brief("slouch")
    assert "slouch" in brief
    assert "固定點" in brief


# ---- 流程 --------------------------------------------------------------

class _Recorder:
    """把 _record 換掉，記下每一次被要求量什麼。"""

    def __init__(self, rejection_rate=0.04):
        self.calls = []
        self._rate = rejection_rate

    def __call__(self, engine, calib, cap, args, baseline, log, snapshots, seconds):
        self.calls.append(seconds)
        session = posture._Session()
        for _ in range(100):
            session.add((5.0, 1.0))
        rejected = int(round(self._rate * 100 / (1 - self._rate)))
        return posture.Recording(
            session=session, ca_window=posture.RollingAngle(30),
            sym_window=posture.RollingAngle(30), rejected=rejected,
            frames=100 + rejected, transitions=[],
            log_path=None if log is None else log.path,
        )


def _patch(monkeypatch, recorder, baseline=None):
    monkeypatch.setattr(posture, "_prepare", lambda args: (None, None, _FakeCap()))
    monkeypatch.setattr(
        posture, "_collect_baseline",
        lambda engine, calib, cap, args, seconds: (baseline or _baseline(), []),
    )
    monkeypatch.setattr(posture, "_record", recorder)
    monkeypatch.setattr(posture, "_report", lambda *a, **k: None)


class _FakeCap:
    def release(self):
        pass


def test_every_condition_gets_measured_in_order(tmp_path, monkeypatch, capsys):
    recorder = _Recorder()
    _patch(monkeypatch, recorder)
    posture._run_study(_args(tmp_path, conditions="upright,forward"))

    assert recorder.calls == [90.0, 90.0]
    folder = tmp_path / "chenyue"
    assert (folder / "upright-1.csv").exists()
    assert (folder / "forward-1.csv").exists()
    assert (folder / "baseline.json").exists()


def test_the_conditions_are_whatever_the_user_names(tmp_path, monkeypatch):
    recorder = _Recorder()
    _patch(monkeypatch, recorder)
    posture._run_study(_args(tmp_path, conditions="slouch, lean-left ,upright"))

    folder = tmp_path / "chenyue"
    assert (folder / "slouch-1.csv").exists()
    assert (folder / "lean-left-1.csv").exists()
    assert (folder / "upright-1.csv").exists()


def test_an_empty_condition_list_stops_before_touching_the_camera(tmp_path):
    with pytest.raises(SystemExit, match="至少要有一種姿勢"):
        posture._run_study(_args(tmp_path, conditions=" , "))


def test_the_baseline_is_taken_once_and_shared_by_every_condition(tmp_path, monkeypatch):
    """基準與量測之間沒有模型重載的空檔，這是零間隔的來源。"""
    taken = []
    recorder = _Recorder()
    _patch(monkeypatch, recorder)
    monkeypatch.setattr(
        posture, "_collect_baseline",
        lambda *a, **k: (taken.append(1), (_baseline(), []))[1],
    )
    posture._run_study(_args(tmp_path, conditions="upright,forward"))
    assert len(taken) == 1


def test_the_baseline_values_reach_every_csv(tmp_path, monkeypatch):
    """基準日後可能被覆蓋，數值沒跟著存進去就再也對不回來。"""
    _patch(monkeypatch, _Recorder())
    posture._run_study(_args(tmp_path, conditions="upright"))
    head = (tmp_path / "chenyue" / "upright-1.csv").read_text(encoding="utf-8")
    assert "# condition=upright" in head
    assert "# trial=1" in head
    assert "# baseline_theta_ca_deg=3.780" in head


def test_a_high_rejection_rate_is_called_out_at_the_end(tmp_path, monkeypatch, capsys):
    """2026-09-29 那份 23% 的資料，共同關鍵點從 12.2 掉到 9.0。"""
    _patch(monkeypatch, _Recorder(rejection_rate=0.30))
    posture._run_study(_args(tmp_path, conditions="upright"))
    printed = capsys.readouterr().out
    assert "略過率偏高" in printed
    assert "建議重量" in printed


def test_clean_segments_are_not_nagged_about(tmp_path, monkeypatch, capsys):
    _patch(monkeypatch, _Recorder(rejection_rate=0.04))
    posture._run_study(_args(tmp_path, conditions="upright"))
    printed = capsys.readouterr().out
    assert "略過率偏高" not in printed
    assert "建議重量" not in printed


def test_the_summary_tells_you_to_look_at_the_pictures_first(tmp_path, monkeypatch, capsys):
    """兩次量測的結論被推翻，都是因為沒有人核對當時的姿勢。"""
    _patch(monkeypatch, _Recorder())
    posture._run_study(_args(tmp_path, conditions="upright,forward"))
    printed = capsys.readouterr().out
    assert "先翻一遍存下來的畫面" in printed
    assert "posture.py analyse" in printed


def test_a_bad_baseline_stops_and_asks_rather_than_carrying_on(tmp_path, monkeypatch):
    """取完就往下走是不行的。那個偏移會留在之後每一次判定裡。"""
    asked = []
    attempts = []

    def collect(engine, calib, cap, args, seconds):
        attempts.append(1)
        # 第一次有警告，第二次乾淨
        return _baseline(), ["略過 41/150 幀（27%）"] if len(attempts) == 1 else []

    monkeypatch.setattr(posture, "_prepare", lambda args: (None, None, _FakeCap()))
    monkeypatch.setattr(posture, "_collect_baseline", collect)
    monkeypatch.setattr(posture, "_record", _Recorder())
    monkeypatch.setattr(posture, "_report", lambda *a, **k: None)
    monkeypatch.setattr(posture, "_ask_yes",
                        lambda q: (asked.append(q), True)[1] if len(asked) < 1 else False)

    posture._run_study(_args(tmp_path, conditions="upright", no_prompt=False))
    assert len(attempts) == 2, "回答要重取就該再取一次"
    assert asked


def test_no_prompt_never_blocks_on_a_bad_baseline(tmp_path, monkeypatch):
    """沒有人在旁邊的時候不能停在 input() 上等一個永遠不會來的按鍵。"""
    attempts = []

    def collect(engine, calib, cap, args, seconds):
        attempts.append(1)
        return _baseline(), ["略過 41/150 幀（27%）"]

    monkeypatch.setattr(posture, "_prepare", lambda args: (None, None, _FakeCap()))
    monkeypatch.setattr(posture, "_collect_baseline", collect)
    monkeypatch.setattr(posture, "_record", _Recorder())
    monkeypatch.setattr(posture, "_report", lambda *a, **k: None)

    posture._run_study(_args(tmp_path, conditions="upright", no_prompt=True))
    assert len(attempts) == 1


def test_the_camera_is_released_even_when_a_segment_blows_up(tmp_path, monkeypatch):
    """相機沒放掉的話，下一次執行會開不起來，而那要重開機才查得出原因。"""
    cap = _FakeCap()
    released = []
    cap.release = lambda: released.append(1)

    monkeypatch.setattr(posture, "_prepare", lambda args: (None, None, cap))
    monkeypatch.setattr(posture, "_collect_baseline",
                        lambda *a, **k: (_baseline(), []))
    monkeypatch.setattr(posture, "_report", lambda *a, **k: None)

    def boom(*a, **k):
        raise RuntimeError("相機炸了")

    monkeypatch.setattr(posture, "_record", boom)
    with pytest.raises(RuntimeError):
        posture._run_study(_args(tmp_path, conditions="upright"))
    assert released
