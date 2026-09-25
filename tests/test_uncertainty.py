"""平均值誤差的估計。"""
import numpy as np
import pytest

from geometry.uncertainty import block_size, lag1_autocorrelation, standard_error


def _ar1(rho: float, sigma: float, count: int, seed: int) -> np.ndarray:
    """產生一段會慢慢漂的序列，用來模擬受試者微調坐姿。"""
    rng = np.random.default_rng(seed)
    values = np.empty(count)
    value = 0.0
    for i, step in enumerate(rng.normal(0.0, sigma, count)):
        value = rho * value + step
        values[i] = value
    return values


def test_independent_samples_give_roughly_std_over_root_n():
    """資料真的獨立時，這個估計要退回到熟悉的 std/√N，不能憑空變大。"""
    values = np.random.default_rng(0).normal(0.0, 8.0, 400)
    naive = values.std(ddof=1) / np.sqrt(values.size)
    assert standard_error(values) == pytest.approx(naive, rel=0.3)


def test_drifting_samples_give_a_larger_error_than_std_over_root_n():
    """這是 θ_CA 實際的樣子，也是換掉 std/√N 的理由。"""
    values = _ar1(0.85, 3.0, 400, seed=1)
    naive = values.std(ddof=1) / np.sqrt(values.size)
    assert standard_error(values) > 2.0 * naive


def test_the_estimate_matches_how_much_the_mean_actually_moves():
    """最直接的檢驗：重複取樣，看平均值實際跳動多少。

    估出來的誤差就該是這個跳動量。差太多的話這個數字寫進論文是不能用的。
    """
    window = 120
    series = _ar1(0.85, 3.0, 20000, seed=2)
    observed = np.std([series[i:i + window].mean()
                       for i in range(0, series.size - window, window)])
    estimated = np.mean([standard_error(series[i:i + window])
                         for i in range(0, series.size - window, window)])
    assert estimated == pytest.approx(observed, rel=0.35)


def test_a_flat_line_has_no_uncertainty():
    assert standard_error([5.0] * 50) == pytest.approx(0.0, abs=1e-12)


def test_too_few_samples_fall_back_rather_than_return_nothing():
    """少於八個樣本切不出區塊。回到 std/√N，偏小但仍然比不給好。"""
    values = [1.0, 3.0, 2.0, 4.0]
    assert standard_error(values) == pytest.approx(
        np.std(values, ddof=1) / 2.0, rel=1e-9
    )


def test_a_single_value_has_no_spread_to_report():
    """回傳 0 會被讀成完美的量測，那是最糟的失效方式。"""
    assert standard_error([7.0]) is None
    assert standard_error([]) is None


def test_block_size_grows_with_the_sample():
    assert block_size(400) == 20
    assert block_size(3) == 2


def test_autocorrelation_separates_drift_from_noise():
    independent = np.random.default_rng(4).normal(0.0, 1.0, 500)
    assert abs(lag1_autocorrelation(independent)) < 0.15
    assert lag1_autocorrelation(_ar1(0.85, 1.0, 500, seed=5)) > 0.7


def test_autocorrelation_is_undefined_without_variation():
    assert lag1_autocorrelation([2.0] * 20) is None
    assert lag1_autocorrelation([1.0, 2.0]) is None
