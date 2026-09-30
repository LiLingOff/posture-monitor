"""在畫面上寫中文。

`cv2.putText` 用的 Hershey 字型只有 ASCII，中文會整排變成問號，所以
`calibration/capture.py` 的拍攝畫面一律寫英文。這裡要的是中文面板，只能走
Pillow。

兩件事決定了這個模組的樣子：

**只轉一塊帶狀區域。** numpy 與 PIL 之間來回轉換要複製整張圖，每幀都做太浪費。
面板是畫面下方一條約 120px 高的帶子，只轉那一塊，成本低一個量級。

**找不到 Pillow 或找不到字型時退回英文。** 視窗本來就是選配，不該因為一台機器
沒裝字型就量不了。所以每段文字都要同時給中文與英文兩種寫法，退回時用後者，
並且說一次原因。
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

import cv2
import numpy as np

# 由上而下試。Windows 是開發機、Noto 是 Jetson 上 JetPack 的預設中日韓字型。
_FONT_CANDIDATES = (
    "C:/Windows/Fonts/msjh.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKtc-Regular.otf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
)


@dataclass(frozen=True)
class TextItem:
    """一段要畫的字。

    `ascii_text` 不是可選的：沒有字型時整個面板都靠它，臨時把中文轉成拼音或
    直接畫問號都不算退路。留空的話那一段字在退回模式下會整段消失，而消失的
    正好是狀態詞這種最不能少的東西。
    """

    x: int
    y: int
    text: str
    ascii_text: str
    colour: tuple[int, int, int]
    size: int = 20
    bold: bool = False


def find_font(explicit: Path | str | None = None) -> Path | None:
    """找一份畫得出中文的字型。找不到回傳 None。"""
    if explicit is not None:
        path = Path(explicit)
        return path if path.is_file() else None
    for candidate in _FONT_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            return path
    return _fc_match()


def _fc_match() -> Path | None:
    """最後一招：問系統自己的字型設定。只有 Linux 有 fontconfig。"""
    if sys.platform == "win32":
        return None
    try:
        out = subprocess.run(
            ["fc-match", "-f", "%{file}", "sans:lang=zh-tw"],
            capture_output=True, text=True, timeout=3.0, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    path = Path(out.stdout.strip())
    return path if out.stdout.strip() and path.is_file() else None


class TextPainter:
    """畫字，畫不了中文就畫英文。"""

    def __init__(self, font_path: Path | str | None = None):
        self._fonts: dict[int, object] = {}
        self._path: Path | None = None
        self._why: str | None = None
        self._told = False

        if find_spec("PIL") is None:
            self._why = ("沒有安裝 Pillow，畫面上的文字改用英文。"
                         "要中文的話 pip install Pillow")
            return

        path = find_font(font_path)
        if path is None:
            self._why = ("找不到中日韓字型，畫面上的文字改用英文。"
                         "用 --font 指定一份，或安裝 Noto Sans CJK")
            return
        self._path = path

    @property
    def cjk(self) -> bool:
        """畫得出中文嗎。"""
        return self._path is not None

    def notice(self) -> str | None:
        """退回英文的原因，只回傳一次。

        每幀都印的話會蓋掉量測本身的輸出，而這是開場就確定、之後不會變的事。
        """
        if self._why is None or self._told:
            return None
        self._told = True
        return self._why

    def paint(self, canvas: np.ndarray, items) -> np.ndarray:
        """把幾段字畫到 canvas 上，就地修改並回傳它。

        一次收下全部，因為 Pillow 那條路每呼叫一次就要來回轉換一整塊影像。
        """
        items = list(items)
        if not items:
            return canvas
        if not self.cjk:
            return self._paint_hershey(canvas, items)
        return self._paint_pillow(canvas, items)

    def _paint_hershey(self, canvas: np.ndarray, items) -> np.ndarray:
        for item in items:
            # Hershey 的字高約為 scale 的 22px，換算成要求的像素高度。
            scale = item.size / 22.0
            cv2.putText(canvas, item.ascii_text, (item.x, item.y + item.size),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, item.colour,
                        2 if item.bold else 1, cv2.LINE_AA)
        return canvas

    def _paint_pillow(self, canvas: np.ndarray, items) -> np.ndarray:
        from PIL import Image, ImageDraw

        # PIL 是 RGB，cv2 是 BGR。顏色在 TextItem 裡照 cv2 的慣例寫，所以這裡
        # 轉影像也轉顏色，兩邊才對得起來。
        image = Image.fromarray(canvas[:, :, ::-1])
        draw = ImageDraw.Draw(image)
        for item in items:
            draw.text((item.x, item.y), item.text, font=self._font(item.size),
                      fill=tuple(int(c) for c in item.colour[::-1]))
        canvas[:, :, :] = np.asarray(image)[:, :, ::-1]
        return canvas

    def _font(self, size: int):
        if size not in self._fonts:
            from PIL import ImageFont

            self._fonts[size] = ImageFont.truetype(str(self._path), size)
        return self._fonts[size]

    def width(self, text: str, ascii_text: str, size: int) -> int:
        """一段字畫出來有多寬，用來排版。"""
        if not self.cjk:
            # 用 cv2 自己量。先前照字數估，粗體大字會少算一截，於是狀態詞會被
            # 下一段字蓋住，而狀態詞就在最前面。
            (w, _), _ = cv2.getTextSize(ascii_text, cv2.FONT_HERSHEY_SIMPLEX,
                                        size / 22.0, 2)
            return int(w)
        font = self._font(size)
        return int(font.getlength(text))
