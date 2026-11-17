"""θ_KA（軀幹前傾角）在基準、判定、記錄、分析與彙整這幾層的行為。

角度本身的幾何在 test_posture_angles。這裡要守的是**不能動到既有的行為**：
舊的基準檔、舊的 CSV、沒有打開這一軸的判定，結果都要與補回 θ_KA 之前一樣。
"""
import csv
import json
from dataclasses import asdict

import pytest

from geometry.baseline import BaselineCollector, PostureBaseline
from geometry.cohort import collect
from geometry.cohort_report import session_rows, to_csv
from geometry.judgement import THETA_KA_THRESHOLD_DEG, Posture, PostureJudge
from geometry.measurement_log import MeasurementLog
from geometry.monitor import Mode, MonitorState
from geometry.recording import Session
from geometry.session_analysis import analyse_session
from geometry.smoothing import RollingAngle
from monitor_fakes import baseline as _baseline
from monitor_fakes import measurement


def _full(window: RollingAngle, value: float) -> RollingAngle:
    for _ in range(window.window):
        window.add(value)
    return window


# ---- 基準 ---------------------------------------------------------------

def test_an_old_baseline_file_without_theta_ka_still_loads(tmp_path):
    """後加的欄位一定要有預設值，否則所有既有的基準檔都讀不進來。"""
    data = asdict(_baseline())
    for key in ("theta_ka_deg", "theta_ka_std_deg", "theta_ka_standard_error_deg"):
        data.pop(key)
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = PostureBaseline.load(path)
    assert loaded.theta_ka_deg is None
    assert loaded.trunk_threshold_deg is None
    assert loaded.correct_ka(12.0) is None


def test_the_baseline_subtracts_its_own_zero():
    base = _baseline(theta_ka_deg=4.0)
    assert base.correct_ka(14.0) == pytest.approx(10.0)
    assert base.correct_ka(None) is None
    assert base.trunk_threshold_deg == THETA_KA_THRESHOLD_DEG


def test_the_collector_does_not_reject_frames_without_theta_ka_and_needs_enough_of_them():
    collector = BaselineCollector(minimum_frames=40)
    for i in range(60):
        # 一半的幀沒有 θ_KA（髖被擋住），那些幀照常算數。
        reason = collector.add(measurement(theta_ka=5.0 if i % 2 else None))
        assert reason is None
    baseline = collector.finish("s", 20.0)
    assert collector.count == 60
    assert baseline.theta_ka_deg is None, "30 幀 < 下限 40，不拿幾幀當零點"


def test_the_collector_keeps_theta_ka_when_enough_frames_have_it():
    collector = BaselineCollector(minimum_frames=40)
    for _ in range(60):
        collector.add(measurement(theta_ka=5.0))
    baseline = collector.finish("s", 20.0)
    assert baseline.theta_ka_deg == pytest.approx(5.0)
    assert baseline.trunk_threshold_deg == THETA_KA_THRESHOLD_DEG
    assert "θ_KA" in baseline.describe()


# ---- 判定 ---------------------------------------------------------------

def test_the_trunk_axis_is_inert_until_it_is_switched_on():
    judge = PostureJudge()
    ca, sym = _full(RollingAngle(4), 0.0), _full(RollingAngle(4), 0.0)
    *_, ka, combined = judge.update(ca, sym, None, _full(RollingAngle(4), 40.0))
    assert ka is None
    assert combined.posture is Posture.OK


@pytest.mark.parametrize("value", [15.0, -15.0])
def test_trunk_lean_over_the_threshold_is_over_in_both_directions(value):
    judge = PostureJudge()
    judge.watch_trunk()
    ca, sym = _full(RollingAngle(4), 0.0), _full(RollingAngle(4), 0.0)
    *_, ka, combined = judge.update(ca, sym, None, _full(RollingAngle(4), value))
    assert ka.posture is Posture.OVER
    assert combined.posture is Posture.OVER
    assert "θ_KA" in combined.reason


def test_a_trunk_within_the_threshold_is_ok():
    judge = PostureJudge()
    judge.watch_trunk()
    ca, sym = _full(RollingAngle(4), 0.0), _full(RollingAngle(4), 0.0)
    *_, ka, combined = judge.update(ca, sym, None, _full(RollingAngle(4), 3.0))
    assert ka.posture is Posture.OK and combined.posture is Posture.OK


def test_an_unfilled_trunk_window_keeps_the_whole_verdict_unknown():
    """有一項還不知道就不能說正常。"""
    judge = PostureJudge()
    judge.watch_trunk()
    ca, sym = _full(RollingAngle(4), 0.0), _full(RollingAngle(4), 0.0)
    *_, combined = judge.update(ca, sym, None, RollingAngle(4))
    assert combined.posture is Posture.UNKNOWN


# ---- monitor ------------------------------------------------------------

def _monitoring(baseline) -> MonitorState:
    state = MonitorState("s", window=3, baseline=baseline)
    assert state.mode is Mode.MONITORING
    return state


def test_monitor_judges_the_trunk_only_when_the_baseline_has_it():
    base = dict(theta_ca_deg=5.0, theta_sym_deg=1.0)
    with_ka = _monitoring(_baseline(theta_ka_deg=0.0, **base))
    without = _monitoring(_baseline(**base))
    last = None
    for t in range(6):
        m = measurement(theta_ca=5.0, theta_sym=1.0, theta_ka=25.0)
        last = (with_ka.feed(m, t), without.feed(m, t))
    assert last[0].verdict.posture is Posture.OVER
    assert last[1].verdict.posture is not Posture.OVER


def test_monitor_resets_the_trunk_window_when_the_baseline_changes():
    state = _monitoring(_baseline(theta_ka_deg=0.0))
    for t in range(4):
        state.feed(measurement(theta_ka=25.0), t)
    assert state.ka_window.mean == pytest.approx(25.0)
    state._adopt(_baseline(theta_ka_deg=0.0))
    assert state.ka_window.mean is None


# ---- 錄製與記錄 ---------------------------------------------------------

def test_the_session_averages_theta_ka_and_says_none_when_it_has_none():
    session = Session()
    assert session.mean_ka is None
    session.add((1.0, 1.0), 6.0)
    session.add((1.0, 1.0), None)
    session.add((1.0, 1.0), 8.0)
    assert session.mean_ka == pytest.approx(7.0)


def _rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def test_the_log_writes_theta_ka_and_leaves_blanks_blank(tmp_path):
    path = tmp_path / "s.csv"
    with MeasurementLog(path, baseline=_baseline(theta_ka_deg=2.0)) as log:
        log.write(measurement(theta_ka=12.0), theta_ka_corrected=10.0, theta_ka_mean=9.5)
        log.write(measurement(theta_ka=None))
    first, second = _rows(path)
    assert float(first["theta_ka_deg"]) == 12.0
    assert float(first["theta_ka_corrected_deg"]) == 10.0
    assert float(first["theta_ka_mean_deg"]) == 9.5
    assert first["theta_ka_side"] == "both"
    # 沒有值寫空字串，不寫 0：0 是合法的角度。
    assert second["theta_ka_deg"] == "" and second["theta_ka_corrected_deg"] == ""
    assert "# baseline_theta_ka_deg=2.000" in path.read_text(encoding="utf-8")


def test_the_log_header_omits_theta_ka_for_a_baseline_without_it(tmp_path):
    path = tmp_path / "s.csv"
    with MeasurementLog(path, baseline=_baseline()) as log:
        log.write(measurement())
    assert "baseline_theta_ka_deg" not in path.read_text(encoding="utf-8")


def test_new_columns_come_after_the_old_ones(tmp_path):
    """既有的程式與讀 CSV 的人靠欄位位置，新欄位只能加在最後面。"""
    path = tmp_path / "s.csv"
    with MeasurementLog(path) as log:
        log.write(measurement())
    ordered = list(_rows(path)[0].keys())
    assert ordered[-8:] == ["theta_ka_deg", "theta_ka_corrected_deg",
                            "theta_ka_mean_deg", "theta_ka_side",
                            "dy_neck_px", "dy_left_hip_px", "dy_right_hip_px",
                            "theta_ka_reason"]
    assert ordered.index("dy_left_ear_px") == len(ordered) - 9


def test_the_log_says_why_theta_ka_is_missing(tmp_path):
    """第一批實機資料有值的幀只佔 0～15%，要分得出是沒偵測到還是視差被擋。"""
    path = tmp_path / "s.csv"
    missing = measurement(theta_ka=None)
    missing.angle_errors.append("theta_ka: left_hip與right_hip都未偵測到，無法計算theta_ka")
    with MeasurementLog(path) as log:
        log.write(missing)
        log.write(measurement(theta_ka=5.0))
        log.write(None, reject_reason="沒有偵測到人")
    gone, fine, nothing = _rows(path)
    assert "都未偵測到" in gone["theta_ka_reason"]
    assert fine["theta_ka_reason"] == ""
    assert nothing["theta_ka_reason"] == ""
    # 假量測有頸沒有髖：頸有帶正負號的視差，髖是空字串，不是 0。
    assert float(gone["dy_neck_px"]) == 0.5
    assert gone["dy_left_hip_px"] == "" and gone["dy_right_hip_px"] == ""
    assert nothing["dy_neck_px"] == ""


# ---- analyse 與 cohort --------------------------------------------------

_OLD_HEADER = (
    "frame,elapsed_s,usable,reject_reason,theta_ca_deg,theta_sym_deg,"
    "theta_ca_corrected_deg,theta_sym_corrected_deg,theta_ca_mean_deg,"
    "theta_sym_mean_deg,theta_ca_standard_error_deg,distance_mm,azimuth_deg,"
    "theta_ca_precision_deg,shared_keypoints,max_vertical_disparity_px,"
    "angle_max_vertical_disparity_px,posture"
)


def _write_session(tmp_path, name, ka=None, frames=60, posture="正常"):
    header = _OLD_HEADER + ("" if ka is None else
                            ",theta_ka_deg,theta_ka_corrected_deg,theta_ka_mean_deg,theta_ka_side")
    lines = ["# subject=A", "# condition=slouch", "# trial=1", header]
    for i in range(frames):
        wobble = 0.5 if i % 2 else -0.5
        row = f"{i + 1},{i * 0.2:.3f},1,,5,1,0,0,0,0,0.5,660,22,6.5,13,15.0,4.0,{posture}"
        if ka is not None:
            raw, corrected = ka + wobble + 2.0, ka + wobble
            row += f",{raw},{corrected},{corrected},both"
        lines.append(row)
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_analyse_reports_theta_ka_when_the_file_has_it(tmp_path):
    summary = analyse_session(_write_session(tmp_path, "a.csv", ka=8.0))
    assert summary.angles["θ_KA 扣基準"].mean == pytest.approx(8.0, abs=0.1)
    assert summary.angles["θ_KA 原始"].mean == pytest.approx(10.0, abs=0.1)


def test_analyse_still_reads_a_file_from_before_theta_ka_existed(tmp_path):
    summary = analyse_session(_write_session(tmp_path, "old.csv", ka=None))
    assert not [n for n in summary.angles if "θ_KA" in n]
    assert "θ_CA 扣基準" in summary.angles


def test_replay_judges_the_trunk_only_when_the_csv_recorded_it(tmp_path):
    over = analyse_session(_write_session(tmp_path, "a.csv", ka=25.0), force_replay=True)
    plain = analyse_session(_write_session(tmp_path, "old.csv", ka=None), force_replay=True)
    assert over.judgement.share(Posture.OVER.value) > 0.5
    assert plain.judgement.share(Posture.OVER.value) == 0.0


def test_cohort_rows_and_csv_carry_theta_ka(tmp_path):
    _write_session(tmp_path, "a.csv", ka=8.0)
    rows = session_rows(collect([tmp_path]))
    assert rows[0]["theta_ka_deg"] == pytest.approx(8.0, abs=0.1)
    assert "theta_ka_deg" in to_csv(rows).splitlines()[0]


def test_cohort_csv_puts_the_new_column_last(tmp_path):
    """靠欄位位置讀這份 CSV 的人不會因為新欄位而錯位。"""
    _write_session(tmp_path, "a.csv", ka=8.0)
    header = to_csv(session_rows(collect([tmp_path]))).splitlines()[0].split(",")
    assert header[-1] == "theta_ka_deg"
    assert header.index("distance_mm") == header.index("theta_sym_deg") + 1


# ---- 缺值不能讓舊平均一直留著 -------------------------------------------

def test_stale_theta_ka_is_cleared_when_the_hips_disappear_but_frames_stay_usable():
    """髖被擋住時 θ_CA、θ_sym 照常，這一幀算數，只是 θ_KA 沒有值。

    沒有值的幀不會進視窗，也推不走舊的值：舊平均是 +25° 就會一直報超標。
    """
    state = _monitoring(_baseline(theta_ka_deg=0.0))
    for t in range(4):
        result = state.feed(measurement(theta_ka=25.0), t)
    assert result.verdict.posture is Posture.OVER

    for t in range(4, 12):
        result = state.feed(measurement(theta_ka=None), t)
    assert state.ka_window.mean is None
    assert result.verdict.posture is Posture.UNKNOWN, "資料沒了不能繼續報超標，也不能報正常"


def test_a_brief_gap_does_not_discard_the_window():
    state = _monitoring(_baseline(theta_ka_deg=0.0))
    for t in range(4):
        state.feed(measurement(theta_ka=25.0), t)
    state.feed(measurement(theta_ka=None), 4)
    assert state.ka_window.mean == pytest.approx(25.0)


def test_replay_clears_a_stale_theta_ka_window(tmp_path):
    path = _write_session(tmp_path, "a.csv", ka=25.0, frames=40)
    lines = path.read_text(encoding="utf-8").splitlines()
    # 後 40 幀仍是 usable，但 θ_KA 的欄位全空。
    blank = ",".join(lines[-1].split(",")[:-4]) + ",,,,"
    lines += [blank] * 40
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = analyse_session(path, window=10, force_replay=True)
    # 前 40 幀超標，之後視窗清掉變未知；若舊平均一直留著，80 幀都會是超標。
    assert 0.0 < summary.judgement.share(Posture.OVER.value) < 0.7


def test_an_enabled_trunk_axis_without_a_window_is_unknown_not_silently_skipped():
    judge = PostureJudge()
    judge.watch_trunk()
    ca, sym = _full(RollingAngle(4), 0.0), _full(RollingAngle(4), 0.0)
    *_, ka, combined = judge.update(ca, sym)
    assert ka.posture is Posture.UNKNOWN
    assert combined.posture is Posture.UNKNOWN


def test_cohort_leaves_theta_ka_blank_for_an_old_file(tmp_path):
    _write_session(tmp_path, "old.csv", ka=None)
    rows = session_rows(collect([tmp_path]))
    assert rows[0]["theta_ka_deg"] is None
    # CSV 的空格是空字串，不是 0。
    body = to_csv(rows).splitlines()
    header, data = body[0].split(","), body[1].split(",")
    assert data[header.index("theta_ka_deg")] == ""
