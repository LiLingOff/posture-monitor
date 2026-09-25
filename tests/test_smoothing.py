"""時間序列平均與壞幀排除。

這兩件事在這個硬體上是必要的而非加分項：2026-09-23 實機量測中受試者保持不動，
θ_CA 的單幀值掃過 50 度，而判定門檻是 10 度。
"""
import numpy as np
import pytest

from geometry.smoothing import RollingAngle


def test_averaging_reduces_noise_by_the_square_root_of_the_window():
    """平均 N 幀把雜訊降到 1/√N，這是整個執行期監測設計的依據。

    用實機量到的雜訊量（±11.5°）當輸入，檢查 30 幀之後落在 ±2.1° 附近。
    """
    rng = np.random.default_rng(0)
    truth, noise, window = 12.0, 11.5, 30

    errors = []
    for _ in range(200):
        rolling = RollingAngle(window)
        for value in rng.normal(truth, noise, window):
            rolling.add(value)
        errors.append(rolling.mean - truth)

    assert np.std(errors) == pytest.approx(noise / np.sqrt(window), rel=0.2)
    assert np.std(errors) < 3.0, "平均之後仍然壓不到判定門檻以下"


def test_keeps_only_the_most_recent_values():
    rolling = RollingAngle(3)
    for value in (1.0, 2.0, 3.0, 4.0):
        rolling.add(value)
    assert rolling.count == 3
    assert rolling.mean == pytest.approx(3.0)


def test_reports_whether_the_window_is_full():
    """填滿之前的平均降噪效果還沒到宣稱的程度，使用者要看得出來。"""
    rolling = RollingAngle(3)
    rolling.add(1.0)
    assert not rolling.is_full and rolling.count == 1
    rolling.add(2.0)
    rolling.add(3.0)
    assert rolling.is_full


def test_standard_error_is_the_spread_of_the_mean_not_of_the_samples():
    """判定要看的是平均值的誤差，不是單幀的散佈。

    資料彼此獨立時兩者差約 √N 倍。這裡放寬到 40%，因為區塊平均法是用有限個
    區塊估出來的，本身就有取樣變異，不會剛好等於 std/√N。
    """
    rolling = RollingAngle(36)
    rng = np.random.default_rng(1)
    for value in rng.normal(0.0, 8.0, 36):
        rolling.add(value)
    assert rolling.standard_error == pytest.approx(rolling.std / 6.0, rel=0.4)
    assert rolling.standard_error < rolling.std


def test_a_drifting_window_reports_a_larger_error_than_std_over_root_n():
    """相鄰幀相關時 std/√N 會低估，而 θ_CA 在實機上就是這種資料。

    2026-09-24 的坐正量測，相鄰幀自相關 0.73，std/√N 把誤差低估了 2.6 倍。
    """
    rolling = RollingAngle(60)
    rng = np.random.default_rng(3)
    value = 0.0
    for step in rng.normal(0.0, 3.0, 60):
        value = 0.9 * value + step     # 慢慢漂，不是白雜訊
        rolling.add(value)
    naive = rolling.std / np.sqrt(rolling.count)
    assert rolling.standard_error > 2.0 * naive


def test_no_values_yet_reports_nothing_rather_than_zero():
    """空視窗回傳 0 的話會被讀成完美姿勢，那是最糟的失效方式。"""
    rolling = RollingAngle(10)
    assert rolling.mean is None
    assert rolling.std is None
    assert rolling.standard_error is None


def test_single_value_has_no_spread_to_report():
    rolling = RollingAngle(10)
    rolling.add(5.0)
    assert rolling.mean == pytest.approx(5.0)
    assert rolling.std is None


def test_window_must_be_positive():
    with pytest.raises(ValueError):
        RollingAngle(0)
