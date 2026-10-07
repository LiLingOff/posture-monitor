"""視窗的測試。不需要顯示器：cv2 的四個呼叫換成假的，比照 test_capture_guards。"""
from __future__ import annotations

import cv2
import pytest

from view import window
from view.text import TextPainter
from view_fakes import view

def ascii_painter() -> TextPainter:
    """每次給一份新的。TextPainter 的說明訊息只回傳一次，共用一份的話
    測試就會互相影響，而那種失敗只在特定順序下出現。"""
    return TextPainter(font_path="沒有這個字型.ttf")


class FakeCv2:
    """只實作視窗用得到的那幾個。"""

    # 真的例外類別，不然 window.py 的 except cv2.error 攔不到。
    error = cv2.error
    WINDOW_AUTOSIZE = cv2.WINDOW_AUTOSIZE

    def __init__(self, key: int = 255, fails: str | None = None):
        self.key = key
        self.fails = fails
        self.shown = 0
        self.waits = 0
        self.destroyed = 0
        self.named = 0
        self.last = None

    def _maybe_fail(self, name: str) -> None:
        if self.fails == name:
            raise cv2.error(f"假的 {name} 失敗")

    def namedWindow(self, title, flags=0):
        self.named += 1
        self._maybe_fail("namedWindow")

    def imshow(self, title, image):
        self.shown += 1
        self.last = image
        self._maybe_fail("imshow")

    def waitKey(self, delay):
        self.waits += 1
        return self.key

    def destroyAllWindows(self):
        self.destroyed += 1
        self._maybe_fail("destroyAllWindows")


@pytest.fixture
def fake(monkeypatch):
    stub = FakeCv2()
    monkeypatch.setattr(window, "cv2", stub)
    return stub


def _window(fake_cv2=None, **kwargs) -> window.LiveWindow:
    w = window.LiveWindow(painter=ascii_painter(), **kwargs)
    w.open()
    return w


# ---- 不要視窗的時候 ----------------------------------------------------

def test_nothing_is_drawn_unless_a_window_was_asked_for(fake):
    display = window.open_display(False)
    assert display.show(view()) is None
    assert not display.stopped
    assert fake.shown == 0 and fake.named == 0


def test_the_null_display_answers_every_call_the_loop_makes():
    display = window.NO_DISPLAY
    display.begin()
    assert display.show(view()) is None
    assert display.hold(view()) is None
    assert not display.take_baseline_request()
    display.close()
    assert not display.stopped


# ---- 按鍵 --------------------------------------------------------------

@pytest.mark.parametrize("key", [ord("q"), 27])
def test_q_and_escape_end_the_segment(fake, key):
    fake.key = key
    w = _window()
    w.show(view())
    assert w.stopped


@pytest.mark.parametrize("key", [ord("a"), 255])
def test_an_ordinary_key_does_not_end_the_segment(fake, key):
    """255 是沒有按鍵時 waitKey 回傳的 -1 & 0xFF。"""
    fake.key = key
    w = _window()
    w.show(view())
    assert not w.stopped


def test_a_modifier_bit_does_not_hide_the_quit_key(fake):
    """這正是要 & 0xFF 的原因：某些平台會把修飾鍵疊在高位元上。"""
    fake.key = 0x10071
    w = _window()
    w.show(view())
    assert w.stopped


def test_c_asks_for_a_baseline_rather_than_ending_the_segment(fake):
    fake.key = ord("c")
    w = _window()
    w.show(view())
    assert not w.stopped
    assert w.take_baseline_request()
    # 取走之後要清掉，同一次按鍵不能觸發兩次。
    assert not w.take_baseline_request()


def test_the_stop_flag_is_cleared_when_the_next_segment_starts(fake):
    """不清的話，study 按過一次 q 之後剩下的姿勢會全部立刻收工。"""
    fake.key = ord("q")
    w = _window()
    w.show(view())
    assert w.stopped
    w.begin()
    assert not w.stopped


# ---- 沒有 GUI 後端 -----------------------------------------------------

def test_a_missing_backend_does_not_stop_the_measurement(monkeypatch):
    stub = FakeCv2(fails="namedWindow")
    monkeypatch.setattr(window, "cv2", stub)
    w = window.LiveWindow(painter=ascii_painter())
    w.open()
    assert w.show(view()) is not None
    assert not w.stopped
    assert stub.shown == 0


def test_the_missing_backend_is_reported_once_and_names_the_flag(monkeypatch):
    stub = FakeCv2(fails="imshow")
    monkeypatch.setattr(window, "cv2", stub)
    w = window.LiveWindow(painter=ascii_painter())
    w.open()
    # 每幀最多吐一句，不然會蓋掉量測本身那一行原地更新的輸出。
    notices = [n for n in (w.show(view()) for _ in range(6)) if n is not None]
    assert any("視窗" in n for n in notices)
    # 只有開不了視窗這一句。面板本來就是英文，沒有中日韓字型不是退路。
    assert len(notices) == 1, notices
    assert w.show(view()) is None


def test_no_further_cv2_calls_once_the_backend_is_known_to_be_missing(monkeypatch):
    stub = FakeCv2(fails="imshow")
    monkeypatch.setattr(window, "cv2", stub)
    w = window.LiveWindow(painter=ascii_painter())
    w.open()
    for _ in range(50):
        w.show(view())
    assert stub.shown == 1


def test_a_session_without_a_display_never_touches_cv2(monkeypatch):
    """Qt 後端在沒有 DISPLAY 時可能直接中止行程，那是 try/except 攔不到的。"""
    monkeypatch.setattr(window.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    stub = FakeCv2()
    monkeypatch.setattr(window, "cv2", stub)

    display = window.open_display(True)
    assert isinstance(display, window.NullDisplay)
    notice = display.show(view())
    assert notice is not None and "DISPLAY" in notice
    assert display.show(view()) is None
    assert stub.named == 0 and stub.shown == 0


def test_a_desktop_session_is_not_treated_as_headless(monkeypatch):
    monkeypatch.setattr(window.sys, "platform", "linux")
    monkeypatch.setenv("DISPLAY", ":0")
    assert window.headless_reason() is None


# ---- 沒有資料的那一幀 --------------------------------------------------

def test_waitkey_is_still_pumped_on_a_frame_with_nothing_to_draw(fake):
    """相機掉線時這條路要走兩秒半。不打點的話畫面凍住而且按鍵沒反應。"""
    w = _window()
    w.show(view())
    before = fake.waits
    w.show(None)
    assert fake.waits == before + 1


def test_a_frame_with_no_data_is_marked_rather_than_left_looking_live(fake):
    w = _window()
    w.show(view())
    live = fake.last.copy()
    w.show(None)
    assert not (fake.last == live).all()


def test_nothing_is_shown_when_there_was_never_a_frame(fake):
    w = _window()
    w.show(None)
    assert fake.shown == 0


# ---- 收掉 --------------------------------------------------------------

def test_the_window_is_destroyed_once_at_the_end_not_between_segments(fake):
    w = _window()
    w.begin()
    w.show(view())
    w.begin()
    w.show(view())
    w.close()
    assert fake.destroyed == 1


def test_close_is_safe_when_the_window_was_never_shown(monkeypatch):
    """某些無頭版本的 destroyAllWindows 會拋例外。"""
    stub = FakeCv2(fails="namedWindow")
    monkeypatch.setattr(window, "cv2", stub)
    w = window.LiveWindow(painter=ascii_painter())
    w.open()
    w.close()
    assert stub.destroyed == 0


def test_hold_waits_for_a_key(fake):
    w = _window()
    w.hold(view())
    assert fake.shown == 1 and fake.waits == 1
