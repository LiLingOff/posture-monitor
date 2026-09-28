"""超標判定與遲滯。"""
from geometry.judgement import (AngleJudge, Posture, PostureJudge,
                                windows_are_stale)
from geometry.smoothing import RollingAngle


class _Window:
    """RollingAngle 的替身，直接指定平均與誤差。

    要測的是判定的邏輯，用真的視窗就得反推出一串能產生特定誤差的值。
    """

    def __init__(self, mean, standard_error, is_full=True):
        self.mean = mean
        self.standard_error = standard_error
        self.is_full = is_full


def _judge(**kwargs):
    return AngleJudge(10.0, "θ_CA", **kwargs)


def test_confidently_over_the_threshold_warns():
    assert _judge().update(15.0, 1.0).posture is Posture.OVER


def test_confidently_under_the_threshold_is_normal():
    assert _judge().update(3.0, 1.0).posture is Posture.OK


def test_a_value_within_the_error_of_the_threshold_holds_the_previous_state():
    """這就是遲滯。判斷不出來的時候不要改變狀態。"""
    judge = _judge()
    assert judge.update(3.0, 1.0).posture is Posture.OK
    assert judge.update(10.5, 1.0).posture is Posture.OK       # 差 0.5°，誤差 1°
    assert judge.update(12.0, 1.0).posture is Posture.OVER     # 差 2°，確定超了
    assert judge.update(9.5, 1.0).posture is Posture.OVER      # 回到誤差範圍內，維持
    assert judge.update(8.0, 1.0).posture is Posture.OK        # 確定低於門檻才解除


def test_the_dead_band_widens_when_the_measurement_gets_noisier():
    """寬度從誤差來，不是常數。同樣的角度在誤差大的時候不該下判斷。"""
    assert _judge().update(13.0, 1.0).posture is Posture.OVER
    assert _judge().update(13.0, 4.0).posture is Posture.UNKNOWN


def test_a_larger_margin_factor_demands_more_evidence():
    assert _judge(margin_factor=1.0).update(12.0, 1.5).posture is Posture.OVER
    assert _judge(margin_factor=3.0).update(12.0, 1.5).posture is Posture.UNKNOWN


def test_an_unfilled_window_is_unknown_not_normal():
    """降噪還沒到位，這時候的平均比宣稱的更吵。"""
    judge = _judge()
    assert judge.update(3.0, 1.0, ready=False).posture is Posture.UNKNOWN
    assert judge.update(3.0, 1.0, ready=True).posture is Posture.OK


def test_no_measurement_reports_unknown_rather_than_normal():
    """整段都被略過時回報正常的話，偵測失效會被讀成坐姿良好。"""
    assert _judge().update(None, None).posture is Posture.UNKNOWN


def test_a_missing_error_falls_back_to_the_floor_not_to_zero():
    """視窗裡只有一個值時誤差是 None。當成零的話遲滯就消失了。"""
    judge = _judge(minimum_margin_deg=2.0)
    assert judge.update(11.0, None).posture is Posture.UNKNOWN
    assert judge.update(13.0, None).posture is Posture.OVER


def test_a_very_still_subject_still_gets_a_dead_band():
    """誤差趨近 0 時寬度不能跟著趨近 0，否則真實的微幅移動會讓狀態翻來翻去。"""
    judge = _judge(minimum_margin_deg=1.0)
    judge.update(3.0, 0.0)
    assert judge.update(10.4, 0.0).posture is Posture.OK


def test_theta_ca_only_warns_about_leaning_forward():
    """往後靠不是這個系統要提醒的事，前作的門檻也是單邊的。"""
    judge = AngleJudge(10.0, "θ_CA", two_sided=False)
    assert judge.update(-20.0, 1.0).posture is Posture.OK


def test_theta_sym_warns_about_either_shoulder_being_higher():
    """肩膀往左歪往右歪都算歪。"""
    judge = AngleJudge(5.0, "θ_sym", two_sided=True)
    assert judge.update(-8.0, 1.0).posture is Posture.OVER
    assert AngleJudge(5.0, "θ_sym", two_sided=True).update(8.0, 1.0).posture is Posture.OVER


def test_changed_is_true_only_on_a_transition():
    """回饋要在狀態改變時觸發，不是每一幀都觸發。"""
    judge = _judge()
    assert judge.update(3.0, 1.0).changed is True
    assert judge.update(3.0, 1.0).changed is False
    assert judge.update(15.0, 1.0).changed is True


def test_frames_in_state_counts_how_long_the_state_has_held():
    """持續多久才提醒是呼叫端的決定，這一層負責把時間長度算出來。"""
    judge = _judge()
    judge.update(15.0, 1.0)
    judge.update(15.0, 1.0)
    result = judge.update(15.0, 1.0)
    assert result.frames_in_state == 3
    assert judge.update(3.0, 1.0).frames_in_state == 1


def test_the_reason_carries_the_numbers_that_produced_the_verdict():
    """從一個列舉值反推不出「為什麼判成這樣」，調參數與寫報告都要用到。"""
    reason = _judge().update(15.0, 1.2).reason
    assert "θ_CA" in reason
    assert "+15.0" in reason
    assert "1.2" in reason
    assert "10" in reason


def test_either_angle_over_the_threshold_makes_the_whole_posture_over():
    judge = PostureJudge()
    _, _, combined = judge.update(_Window(3.0, 0.5), _Window(9.0, 0.5))
    assert combined.posture is Posture.OVER
    assert "θ_sym" in combined.reason


def test_both_angles_under_the_threshold_is_normal():
    judge = PostureJudge()
    _, _, combined = judge.update(_Window(3.0, 0.5), _Window(1.0, 0.5))
    assert combined.posture is Posture.OK


def test_one_angle_unknown_is_not_reported_as_normal():
    """另一項正常不代表整體正常，少了一半的資訊就是判斷不出來。"""
    judge = PostureJudge()
    _, _, combined = judge.update(_Window(3.0, 0.5), _Window(None, None))
    assert combined.posture is Posture.UNKNOWN


def test_the_two_angles_keep_separate_hysteresis():
    """門檻、方向、量測誤差都不同，共用一個狀態機會把兩者的遲滯綁在一起。"""
    judge = PostureJudge()
    judge.update(_Window(3.0, 0.5), _Window(1.0, 0.5))
    ca, sym, _ = judge.update(_Window(15.0, 0.5), _Window(1.0, 0.5))
    assert ca.posture is Posture.OVER
    assert sym.posture is Posture.OK


def test_it_works_with_a_real_rolling_window():
    """替身測的是邏輯，這一條確認接口對得上真的 RollingAngle。"""
    judge = PostureJudge()
    ca_window, sym_window = RollingAngle(4), RollingAngle(4)
    for _ in range(4):
        ca_window.add(25.0)
        sym_window.add(0.5)
    _, _, combined = judge.update(ca_window, sym_window)
    assert combined.posture is Posture.OVER


def test_a_half_filled_real_window_is_unknown():
    judge = PostureJudge()
    ca_window, sym_window = RollingAngle(30), RollingAngle(30)
    ca_window.add(25.0)
    sym_window.add(0.5)
    _, _, combined = judge.update(ca_window, sym_window)
    assert combined.posture is Posture.UNKNOWN


def test_a_window_of_values_older_than_the_window_itself_is_stale():
    """偵測連續失敗到視窗長度，裡面每個值都比一個視窗的跨度更舊。

    不處理的話，受試者離開座位之後裝置會對著空椅子繼續回報上一個狀態。
    """
    assert windows_are_stale(29, 30) is False
    assert windows_are_stale(30, 30) is True


def test_a_window_of_zero_is_never_stale():
    """避免除零式的邊界：視窗長度 0 本來就不該出現，但不能因此判成過期。"""
    assert windows_are_stale(5, 0) is False


def test_clearing_the_window_puts_the_judgement_back_to_unknown():
    """清掉視窗之後不能再回報上一個狀態，這是過期處理要達成的效果。"""
    judge = PostureJudge()
    ca_window, sym_window = RollingAngle(4), RollingAngle(4)
    for _ in range(4):
        ca_window.add(25.0)
        sym_window.add(0.5)
    assert judge.update(ca_window, sym_window)[2].posture is Posture.OVER
    ca_window.clear()
    sym_window.clear()
    assert judge.update(ca_window, sym_window)[2].posture is Posture.UNKNOWN
