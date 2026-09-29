"""live 迴圈遇到硬體與偵測問題時的行為。

這些都來自 2026-09-29 的實機量測：相機中途斷線、以及同一個原因連續略過。
"""
from types import SimpleNamespace

import pytest

from geometry.pipeline import rejection_advice
from posture_loader import load_posture_module


def _posture():
    return load_posture_module("posture_live")


def test_a_dropped_frame_does_not_stop_the_measurement():
    """偶爾掉一幀是正常的，停下來的話二十分鐘的量測會被一次打嗝毀掉。"""
    posture = _posture()
    args = SimpleNamespace(camera=1)
    assert posture._camera_lost(args, 1) is None
    assert posture._camera_lost(args, posture._MAX_CONSECUTIVE_READ_FAILURES - 1) is None


def test_a_camera_that_is_gone_stops_the_measurement():
    """2026-09-29 實機：USB 中途斷線，迴圈全速空轉 110 秒寫了 162485 筆空記錄。

    相機不見了的話重試多少次都一樣，而重試迴圈會把有效資料埋在後面。
    """
    posture = _posture()
    args = SimpleNamespace(camera=1)
    message = posture._camera_lost(args, posture._MAX_CONSECUTIVE_READ_FAILURES)
    assert message is not None
    # 當下最想知道的是資料還在不在
    assert "已經寫下的資料是完整的" in message


def test_a_read_failure_is_a_different_kind_of_error_from_no_person():
    """兩者該有的反應相反：偵測失敗值得繼續跑，相機不見了不值得。"""
    from calibration.capture import CameraReadError

    assert issubclass(CameraReadError, RuntimeError)
    assert not isinstance(RuntimeError("左眼沒有偵測到人"), CameraReadError)


def test_the_retry_pause_keeps_the_log_from_running_away():
    """重試之間不停一下的話，掉線的相機會用滿磁碟。

    上限是這兩個數字的乘積：25 次 × 0.1 秒，約兩秒半，之後就停止量測。
    """
    posture = _posture()
    assert posture._READ_RETRY_PAUSE_S > 0
    worst = posture._MAX_CONSECUTIVE_READ_FAILURES
    assert worst * posture._READ_RETRY_PAUSE_S < 10.0


def _watcher(after=3):
    return _posture()._StuckWatcher(after=after)


def test_a_repeated_reason_eventually_says_what_to_do():
    """逐幀那一行只說哪裡不對，而且會被下一幀蓋掉。"""
    watcher = _watcher()
    reason = "right_shoulder 的垂直視差 8.9px，左右配對錯了"
    assert watcher.saw(reason) is None
    assert watcher.saw(reason) is None
    told = watcher.saw(reason)
    assert told is not None
    assert "連續 3 幀" in told
    assert "兩邊肩膀都完整露出來" in told


def test_the_hint_is_printed_once_per_run_not_every_frame():
    """反覆印會把它變成跟原本那一行一樣的背景雜訊。"""
    watcher = _watcher()
    reason = "right_shoulder 的垂直視差 8.9px，左右配對錯了"
    told = [watcher.saw(reason) for _ in range(20)]
    assert sum(1 for t in told if t is not None) == 1


def test_the_same_problem_with_different_numbers_is_one_run():
    """「8.9px」與「8.4px」是同一件事，分開算的話提示永遠不會觸發。"""
    watcher = _watcher()
    for px in (8.9, 8.4, 9.2):
        told = watcher.saw(f"right_shoulder 的垂直視差 {px}px，左右配對錯了")
    assert told is not None


def test_a_good_frame_resets_the_run():
    """零星的失誤不該累積成提示。"""
    watcher = _watcher()
    reason = "left_shoulder 的垂直視差 8.4px，左右配對錯了"
    watcher.saw(reason)
    watcher.saw(reason)
    watcher.saw(None)
    assert watcher.saw(reason) is None
    assert watcher.saw(reason) is None
    assert watcher.saw(reason) is not None


def test_switching_to_a_different_problem_starts_over():
    watcher = _watcher()
    watcher.saw("深度 104mm 落在合理範圍外")
    watcher.saw("深度 108mm 落在合理範圍外")
    assert watcher.saw("左右只有 3 個共同關鍵點") is None


def test_shoulder_mispairing_is_told_apart_from_other_disparity_problems():
    """肩膀配錯是遮擋，要動的是模組角度；其他點配錯多半是標定。"""
    shoulder = rejection_advice("right_shoulder 的垂直視差 8.9px，左右配對錯了")
    other = rejection_advice("right_wrist 的垂直視差 30px，左右配對錯了")
    assert "遠側肩膀" in shoulder
    assert "標定" in other
    assert shoulder != other


def test_every_rejection_reason_the_pipeline_produces_has_advice():
    """原因說的是哪裡不對，建議說的是怎麼辦。少一個就是印了一句沒有下一步的話。"""
    for reason in (
        "深度 104mm 落在桌前坐姿的合理範圍外（200~3000mm）",
        "深度 -5138mm 是負的，左右配對接反了",
        "left_shoulder 的垂直視差 8.4px，左右配對錯了",
        "左右只有 3 個共同關鍵點",
        "左眼沒有偵測到人。確認受試者在畫面內、光線足夠",
        "角度用到的 left_shoulder 貼在畫面邊緣，真實位置在畫面外",
    ):
        assert rejection_advice(reason), reason


def test_an_unknown_reason_says_nothing_rather_than_guessing():
    assert rejection_advice("某個沒見過的原因") == ""


@pytest.mark.parametrize("after", [1, 5, 30])
def test_the_threshold_is_respected(after):
    watcher = _watcher(after=after)
    reason = "深度 104mm 落在合理範圍外"
    told = [watcher.saw(reason) for _ in range(after + 2)]
    assert told.index(next(t for t in told if t is not None)) == after - 1


def test_the_snapshot_name_says_when_what_and_the_verdict():
    """要能不開 CSV 就看出這一張是什麼時候、量到多少、判成什麼。"""
    posture = _posture()
    name = posture.snapshot_name(42, 87.3, 24.9, "超標")
    assert name == "000042_t087.3s_ca+024.9_over.png"


def test_snapshot_names_sort_in_time_order():
    """檔案總管按名字排序，補零之後才與時間順序一致。"""
    posture = _posture()
    names = [posture.snapshot_name(i, i * 10.0, 5.0, "正常") for i in (2, 10, 100)]
    assert names == sorted(names)


def test_a_missing_angle_is_not_written_as_zero():
    """0 是合法的角度值，拿它表示算不出來會讓檔名說謊。"""
    posture = _posture()
    assert "cana" in posture.snapshot_name(1, 1.0, None, None)


def test_the_sign_survives_because_it_is_the_whole_point():
    """θ_CA 的符號就是前傾與後仰的差別。"""
    posture = _posture()
    assert "ca-012.0" in posture.snapshot_name(1, 1.0, -12.0, "正常")
    assert "ca+012.0" in posture.snapshot_name(1, 1.0, 12.0, "正常")


def test_snapshots_are_off_unless_a_folder_is_given():
    posture = _posture()
    assert not posture._Snapshots(None, 10.0).enabled
    assert not posture._Snapshots(__import__("pathlib").Path("x"), 0).enabled


def test_snapshots_are_taken_on_the_interval_not_every_frame(tmp_path):
    """每幀都存的話，兩分鐘的量測會產生六百張，沒有人會去看。"""
    import numpy as np

    posture = _posture()
    shots = posture._Snapshots(tmp_path, 10.0)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    saved = [shots.maybe_save(i, i * 1.0, frame, None, 5.0, "正常")
             for i in range(25)]
    taken = [s for s in saved if s is not None]
    assert len(taken) == 3, "0、10、20 秒各一張"
    assert len(list(tmp_path.glob("*.png"))) == 3


def test_the_end_summary_covers_the_whole_session_not_the_last_window():
    """2026-09-29 實機：跑了一百秒，摘要印的是最後 30 幀（約六秒）的平均。

    標題寫的是整段，數字卻只有結尾那一小段，兩者對不起來。
    """
    posture = _posture()
    session = posture._Session()
    for value in [2.0] * 200 + [20.0] * 30:
        session.add((value, 1.0))
    assert len(session.ca) == 230
    import numpy as np
    assert np.mean(session.ca) == pytest.approx(4.35, abs=0.01)


def test_the_session_keeps_the_two_angles_apart():
    """θ_CA 算得出來而 θ_sym 算不出來的幀是常見的，不能讓它們錯位。"""
    posture = _posture()
    session = posture._Session()
    session.add((5.0, None))
    session.add((6.0, 1.0))
    session.add((None, 2.0))
    assert session.ca == [5.0, 6.0]
    assert session.sym == [1.0, 2.0]
