"""θ_CA、θ_sym、肩高某一軸單獨缺值，舊平均也要清掉（與 θ_KA 同一類問題）。

GPT 審查（2026-11-22）指出：左肩被擋時 θ_CA 照常、θ_sym 算不出來，這一幀仍算數，
整幀略過計數歸零，θ_sym 視窗卻沒有新值推走舊值，永遠不會過期。
"""

import numpy as np

from geometry.monitor import Mode, MonitorState
from monitor_fakes import baseline as _baseline
from monitor_fakes import measurement


def _monitoring() -> MonitorState:
    state = MonitorState("s", window=3, baseline=_baseline())
    assert state.mode is Mode.MONITORING
    return state


def test_stale_theta_sym_is_cleared_when_one_shoulder_disappears():
    state = _monitoring()
    for t in range(4):
        state.feed(measurement(theta_sym=20.0), t)
    assert state.sym_window.mean is not None

    for t in range(4, 12):
        state.feed(measurement(theta_ca=5.0, theta_sym=None), t)
    assert state.sym_window.mean is None
    assert state.ca_window.mean is not None, "θ_CA 照常，不能被一起清掉"


def test_stale_theta_ca_is_cleared_when_only_theta_ca_disappears():
    state = _monitoring()
    for t in range(4):
        state.feed(measurement(theta_ca=30.0), t)
    for t in range(4, 12):
        state.feed(measurement(theta_ca=None), t)
    assert state.ca_window.mean is None
    assert state.sym_window.mean is not None


def test_a_brief_theta_sym_gap_keeps_the_window():
    state = _monitoring()
    for t in range(4):
        state.feed(measurement(theta_sym=20.0), t)
    state.feed(measurement(theta_sym=None), 4)
    assert state.sym_window.mean is not None


# ---- GPT 審查 2026-11-22：CSV 解析的邊界 --------------------------------

def test_non_finite_cells_count_as_missing():
    from geometry.session_analysis import _number

    for text in ("nan", "NaN", "inf", "-inf"):
        assert _number(text) is None, text
    assert _number("0") == 0.0, "0 是合法的角度，不能當缺值"
    assert _number("-1.5") == -1.5


def test_a_legacy_file_with_unequal_column_lengths_does_not_crash():
    from types import SimpleNamespace

    from geometry.session_analysis import _used_a_baseline

    summary = SimpleNamespace(angles={
        "θ_CA 原始": SimpleNamespace(values=np.array([1.0, 2.0, 3.0])),
        "θ_CA 扣基準": SimpleNamespace(values=np.array([1.0, 2.0])),
    })
    assert _used_a_baseline(summary, {}) is True


def test_equal_length_columns_with_the_same_values_mean_no_baseline():
    from types import SimpleNamespace

    from geometry.session_analysis import _used_a_baseline

    same = np.array([1.0, 2.0, 3.0])
    summary = SimpleNamespace(angles={
        "θ_CA 原始": SimpleNamespace(values=same),
        "θ_CA 扣基準": SimpleNamespace(values=same.copy()),
    })
    assert _used_a_baseline(summary, {}) is False
    summary.angles["θ_CA 扣基準"] = SimpleNamespace(values=same - 5.0)
    assert _used_a_baseline(summary, {}) is True
