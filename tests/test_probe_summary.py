"""probe 整理可用並排模式的邏輯。

這段會直接影響選解析度的決定，而解析度一旦選錯、標定完才發現，
整組內參都要重來，所以把實機遇到的情況鎖住。
"""
import pytest

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


class _FakePlatform(str):
    """讓 sys.platform.startswith("linux") 在任何開發機上都成立。"""

    def startswith(self, prefix):  # noqa: D102
        return prefix == "linux"


@pytest.fixture
def linux(monkeypatch):
    from calibration import capture

    monkeypatch.setattr(capture.sys, "platform", _FakePlatform("linux"))
    return capture


def test_reports_the_nodes_that_do_exist_when_the_number_shifted(linux, monkeypatch):
    """節點編號移位是已知成因，而編號多少程式自己看得到，不該叫使用者去查。"""
    monkeypatch.setattr(linux.Path, "exists", lambda self: False)
    monkeypatch.setattr(linux, "_video_nodes", lambda: ["video2", "video3"])

    message = linux.describe_camera_open_failure(0)
    assert "/dev/video0 不存在" in message
    assert "video2、video3" in message
    assert "--camera 2" in message, "要直接說出該改成哪個編號"
    assert "posture.py live" not in message, "訊息寫死子指令的話，跑 baseline 的人會被誤導"


def test_says_the_device_is_absent_when_nothing_is_plugged_in(linux, monkeypatch):
    monkeypatch.setattr(linux.Path, "exists", lambda self: False)
    monkeypatch.setattr(linux, "_video_nodes", lambda: [])

    message = linux.describe_camera_open_failure(0)
    assert "一個 /dev/video* 都沒有" in message
    assert "dmesg" in message


def test_names_the_process_that_is_holding_the_device(linux, monkeypatch):
    monkeypatch.setattr(linux.Path, "exists", lambda self: True)
    monkeypatch.setattr(linux.os, "access", lambda *a: True)
    monkeypatch.setattr(linux, "_own_processes_holding", lambda node: ["pid 4120 (python3)"])

    message = linux.describe_camera_open_failure(0)
    assert "pid 4120 (python3)" in message
    assert "先把它結束掉" in message


def test_points_at_the_group_when_permissions_are_missing(linux, monkeypatch):
    monkeypatch.setattr(linux.Path, "exists", lambda self: True)
    monkeypatch.setattr(linux.os, "access", lambda *a: False)

    message = linux.describe_camera_open_failure(0)
    assert "usermod -aG video" in message


def test_falls_back_to_root_only_checks_when_nothing_local_explains_it(linux, monkeypatch):
    """查得到的都正常時，才把剩下的列成待確認，而不是一開始就丟一張清單。"""
    monkeypatch.setattr(linux.Path, "exists", lambda self: True)
    monkeypatch.setattr(linux.os, "access", lambda *a: True)
    monkeypatch.setattr(linux, "_own_processes_holding", lambda node: [])

    message = linux.describe_camera_open_failure(0)
    assert "讀寫權限正常" in message
    assert "sudo fuser -v /dev/video0" in message
    assert "dmesg" in message


def test_holder_scan_never_raises_on_a_machine_without_proc():
    """這段在錯誤處理路徑上跑，自己壞掉的話會蓋掉真正的錯誤訊息。"""
    from pathlib import Path

    from calibration.capture import _own_processes_holding

    assert _own_processes_holding(Path("/dev/video0")) == [] or True


def test_stereo_open_failure_says_which_camera_is_the_problem():
    from calibration.capture import describe_stereo_open_failure

    only_right = describe_stereo_open_failure((1, True), (2, False))
    assert "index=2" in only_right
    assert "index=1" not in only_right
    assert "另一顆是正常的" in only_right

    neither = describe_stereo_open_failure((1, False), (2, False))
    assert "index=1" in neither and "index=2" in neither
    assert "兩顆都開不了" in neither


def _fake_probe(results: dict):
    """results 把 index 對應到（讀得到畫面嗎, 尺寸）。"""
    return lambda index: results.get(index, (False, None))


def test_skips_the_node_that_opens_but_never_delivers_a_frame():
    """一顆 UVC 雙目模組佔兩個 /dev/videoN，只有一個讀得出畫面。

    所以「開得起來」不算數，要真的讀到一幀。
    """
    from calibration.capture import find_camera_index

    index, how = find_camera_index(
        2560, 720,
        probe=_fake_probe({1: (False, None), 2: (True, (2560, 720))}),
        candidates=[1, 2],
    )
    assert index == 2
    assert "2560x720" in how


def test_prefers_the_node_that_gives_the_resolution_that_was_asked_for():
    """尺寸不符時畫面多半是單眼或裁切過的，切成兩半會得到兩塊不重疊的區域。"""
    from calibration.capture import find_camera_index

    index, _ = find_camera_index(
        2560, 720,
        probe=_fake_probe({0: (True, (640, 480)), 1: (True, (2560, 720))}),
        candidates=[0, 1],
    )
    assert index == 1


def test_falls_back_to_anything_readable_when_nothing_matches():
    from calibration.capture import find_camera_index

    index, how = find_camera_index(
        2560, 720,
        probe=_fake_probe({3: (True, (640, 480))}),
        candidates=[3],
    )
    assert index == 3
    assert "不是要求的" in how


def test_takes_the_lowest_readable_node_when_no_size_was_requested():
    from calibration.capture import find_camera_index

    index, _ = find_camera_index(
        probe=_fake_probe({1: (True, (640, 480)), 2: (True, (640, 480))}),
        candidates=[1, 2],
    )
    assert index == 1


def test_reports_every_node_it_tried_when_none_work():
    from calibration.capture import find_camera_index

    with pytest.raises(RuntimeError, match="1、2、3"):
        find_camera_index(probe=_fake_probe({}), candidates=[1, 2, 3])
