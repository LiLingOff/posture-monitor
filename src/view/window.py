"""即時視窗：開、畫、收按鍵、收掉。

`show()` **回傳**要印的字而不是自己印。量測迴圈用 `_print_line` 管著一行原地
更新的輸出，旁邊亂印會把它打散；repo 既有的答案是 `StuckWatcher.saw()` 回傳
字串交給 `_tell`，這裡照著做。

無頭環境要走得通。部署目標是 Jetson，而 Jetson 多半是 SSH 進去的。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2

from . import overlay
from .text import TextPainter

QUIT_KEYS = (ord("q"), 27)          # q 與 ESC
BASELINE_KEY = ord("c")
# 沒有按鍵時 waitKey 回傳 -1，& 0xFF 之後是 255。
NO_KEY = 255


class NullDisplay:
    """不開視窗時用的替身。

    有這個東西，量測迴圈就不必到處寫 `if display is not None`。
    """

    stopped = False

    def __init__(self, notice: str | None = None):
        self._notice = notice
        self._told = False

    def begin(self) -> None:
        pass

    def take_baseline_request(self) -> bool:
        return False

    def show(self, view) -> str | None:
        if self._notice is None or self._told:
            return None
        self._told = True
        return self._notice

    def hold(self, view) -> str | None:
        return self.show(view)

    def close(self) -> None:
        pass


# 當預設參數用。notice 是 None，所以它永遠不會被改到。
NO_DISPLAY = NullDisplay()


class LiveWindow:
    """一個 cv2 視窗。"""

    def __init__(self, width: int = 1280, eyes: str = "both",
                 painter: TextPainter | None = None,
                 title: str = "posture monitor"):
        # 標題用 ASCII：OpenCV 的 Windows 後端會把非 ASCII 的標題弄壞。
        self._title = title
        self._width = width
        self._eyes = eyes
        self._painter = TextPainter() if painter is None else painter
        self.stopped = False
        self._dead = False
        self._opened = False
        self._last = None
        self._baseline_requested = False
        self._pending: list[str] = []
        notice = self._painter.notice()
        if notice is not None:
            self._pending.append(notice)

    def begin(self) -> None:
        """開始新的一段。

        視窗的壽命是行程不是段落：`study` 每個姿勢呼叫一次錄製迴圈，不逐段清掉
        結束旗標的話，按過一次 q 之後剩下的姿勢會全部立刻收工。
        """
        self.stopped = False
        self._baseline_requested = False

    def take_baseline_request(self) -> bool:
        """有沒有按過 c。取走之後清掉，同一次按鍵不會觸發兩次。"""
        requested, self._baseline_requested = self._baseline_requested, False
        return requested

    def show(self, view) -> str | None:
        """畫一幀並收按鍵。view 是 None 表示這一幀沒有資料。"""
        if self._dead:
            return self._take_notice()

        if view is None:
            if self._last is None:
                return self._take_notice()
            image = overlay.mark_stale(self._last, self._painter)
        else:
            image = overlay.compose(view, self._width, self._painter, self._eyes)
            self._last = image

        try:
            cv2.imshow(self._title, image)
            key = cv2.waitKey(1) & 0xFF
        except cv2.error as exc:
            self._give_up(exc)
            return self._take_notice()

        if key in QUIT_KEYS:
            self.stopped = True
        elif key == BASELINE_KEY:
            self._baseline_requested = True
        return self._take_notice()

    def hold(self, view) -> str | None:
        """畫一幀然後等按鍵。`once` 用：畫面是靜止的，不必一直重畫。"""
        if self._dead:
            return self._take_notice()
        try:
            cv2.imshow(self._title, overlay.compose(
                view, self._width, self._painter, self._eyes))
            cv2.waitKey(0)
        except cv2.error as exc:
            self._give_up(exc)
        return self._take_notice()

    def close(self) -> None:
        # 沒開過就不要呼叫，某些無頭版本的 destroyAllWindows 會拋例外。
        if not self._opened or self._dead:
            return
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        self._opened = False

    def open(self) -> None:
        """先試著把視窗建起來，失敗就退回不顯示。

        這裡刻意用 `namedWindow`，與 `calibration/capture.py` 隱式建立視窗的
        寫法不同：後端是在 namedWindow 初始化的，在這裡試一次就能在啟動當下
        知道結果，而不是量到第 90 秒才發現畫不出來。
        """
        try:
            cv2.namedWindow(self._title, cv2.WINDOW_AUTOSIZE)
            self._opened = True
        except cv2.error as exc:
            self._give_up(exc)

    def _give_up(self, exc) -> None:
        """訊息排進佇列而不是直接回傳，開視窗與畫第一幀是兩個時機。"""
        self._dead = True
        self._pending.append(
            f"開不了視窗（{str(exc).strip().splitlines()[-1][:80]}），"
            f"改用終端機顯示，量測照常。"
        )

    def _take_notice(self) -> str | None:
        return self._pending.pop(0) if self._pending else None


def headless_reason() -> str | None:
    """這台機器現在能不能開視窗。不能的話回傳原因。

    Linux 上沒有 DISPLAY 也沒有 WAYLAND_DISPLAY 時連 cv2 都不要碰：Qt 後端在
    這種情況下可能直接 qFatal 中止整個行程，那是 Python 的 try/except 攔不到的。
    """
    if sys.platform in ("win32", "darwin"):
        return None
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return None
    return ("這個工作階段沒有顯示器（DISPLAY 沒有設定），不開視窗，量測照常。"
            "要看畫面的話從桌面登入，或用 ssh -X")


def open_display(enabled: bool, width: int = 1280, eyes: str = "both",
                 font: Path | str | None = None):
    """要視窗就開一個，不要或開不起來就給一個不做事的替身。"""
    if not enabled:
        return NO_DISPLAY
    reason = headless_reason()
    if reason is not None:
        return NullDisplay(reason)
    window = LiveWindow(width=width, eyes=eyes, painter=TextPainter(font))
    window.open()
    return window
