"""probe 整理可用並排模式的邏輯。

這段會直接影響選解析度的決定，而解析度一旦選錯、標定完才發現，
整組內參都要重來，所以把實機遇到的情況鎖住。
"""
from calibration.capture import _summarize_side_by_side


# 2026-09-21 在 Jetson 上實際測到的結果：(實際寬, 實際高, fps, 是否為直接要到的模式)
JETSON_RESULTS = [
    (640, 480, 30.1, True),
    (640, 480, 30.0, False),      # 要求 800x600
    (1280, 720, 30.1, True),
    (1920, 1080, 28.8, True),
    (640, 472, 30.0, False),      # 要求 640x240
    (1280, 480, 30.0, True),
    (2560, 720, 32.4, True),      # 要求 2560x720，直接給
    (2560, 720, 31.5, False),     # 要求 2560x960，退回 2560x720
    (3840, 1200, 13.9, False),    # 要求 3040x1520
    (3840, 1080, 15.5, True),
]


def test_native_mode_not_overwritten_by_a_later_fallback():
    """同一個實際模式被多個要求達成時，原生支援的身分不能被退回結果蓋掉。

    2560x720 是直接要得到的，但要求 2560x960 時驅動也退回到 2560x720。
    若後者覆蓋前者，摘要會把唯一最適合的模式標成「並非原生支援」，
    剛好誤導掉「優先選原生支援」這條挑選原則。
    """
    summary = _summarize_side_by_side(JETSON_RESULTS)
    fps, exact = summary[(2560, 720)]
    assert exact is True, "2560x720 是原生支援的模式"
    assert fps == 32.4, "fps 要取原生那次量到的值"


def test_only_side_by_side_modes_are_listed():
    summary = _summarize_side_by_side(JETSON_RESULTS)
    assert set(summary) == {(1280, 480), (2560, 720), (3840, 1080), (3840, 1200)}


def test_pure_fallback_stays_marked_as_fallback():
    """3840x1200 只由退回產生，要維持標記。"""
    summary = _summarize_side_by_side(JETSON_RESULTS)
    assert summary[(3840, 1200)][1] is False


def test_order_of_results_does_not_matter():
    """先遇到退回、後遇到原生，結果要一樣。"""
    reversed_order = list(reversed(JETSON_RESULTS))
    assert _summarize_side_by_side(reversed_order)[(2560, 720)][1] is True


def test_no_side_by_side_modes():
    mono_only = [(640, 480, 30.0, True), (1280, 720, 30.0, True)]
    assert _summarize_side_by_side(mono_only) == {}
