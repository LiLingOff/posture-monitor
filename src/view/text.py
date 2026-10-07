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

**選哪一種語言與用哪個算繪器是兩件事。** `lang` 決定畫 `text` 還是 `ascii_text`，
字型找不找得到決定走 Pillow 還是 Hershey。英文配 Pillow 比 Hershey 好看得多，
所以有字型的時候即使選英文也走 Pillow;先前兩件事綁在一起，想看英文就只能
把字型藏起來。
"""
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

import cv2
import numpy as np

# `θ_{CA}` 會畫成 θ 加一個下標的 CA。報告書寫的是真正的下標，而 Unicode 的
# 小型大寫（ꜱʏᴍ）在微軟正黑體裡整排缺字,實測過，會變成一排豆腐。自己排
# 下標不挑字型：小一級、往下挪一點，用的還是本來就有的那幾個字母。
_SUBSCRIPT = re.compile(r"_\{([^}]*)\}")
_SUB_SIZE = 0.72        # 下標的字級倍率
# 下標往下挪多少，單位是主字級。挪太多會掉進下一列,列高是照主字級算的，
# 下標的下緣不在裡面。0.16 在 20px 的字上是 3px。
_SUB_DROP = 0.16

# 由上而下試。Windows 是開發機、Noto 是 Jetson 上 JetPack 的預設中日韓字型。
_FONT_CANDIDATES = (
    "C:/Windows/Fonts/msjh.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKtc-Regular.otf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
)

# 同一份字型的粗體檔。字型檔名怎麼對應是各家自己的慣例，所以列成表。
# 找不到粗體就用描邊假粗,那在中文上會把筆畫之間的空隙填掉，所以只當退路。
_BOLD_OF = {
    "msjh.ttc": "msjhbd.ttc",
    "msyh.ttc": "msyhbd.ttc",
    "NotoSansCJK-Regular.ttc": "NotoSansCJK-Bold.ttc",
    "NotoSansCJKtc-Regular.otf": "NotoSansCJKtc-Bold.otf",
}


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


def hershey_can_draw(char: str) -> bool:
    """Hershey 畫不畫得出這個字。

    OpenCV 5 的 putText 自己會畫 Unicode，連中文都畫得出來；4.x 的 Hershey 只有
    ASCII，其餘每個位元組畫一個問號。Jetson 上是哪一版不一定，所以問，不要猜。

    比對的是「同樣長度的問號」而不是單一個問號：一個 θ 是兩個 UTF-8 位元組，
    畫不出來時是兩個問號。
    """
    def mask(text: str) -> bytes:
        canvas = np.zeros((48, 240), np.uint8)
        cv2.putText(canvas, text, (4, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, 255, 2, cv2.LINE_AA)
        return canvas.tobytes()

    return mask(char) != mask("?" * len(char.encode("utf-8")))


# 畫不出非 ASCII 時，這些字各自換成純 ASCII 的寫法。
_ASCII_INSTEAD = (("θ", "theta"), ("±", "+-"))


def to_ascii_symbols(text: str) -> str:
    """把面板上的符號換成純 ASCII 的寫法。

    判斷放在這裡而不是排版那邊：能不能畫是算繪器才知道的事，而排版只負責
    決定要寫什麼。兩側都寫 θ 與 ±，換台機器截圖才對得起來。
    """
    for symbol, plain in _ASCII_INSTEAD:
        text = text.replace(symbol, plain)
    return text


def find_bold_font(regular: Path | None) -> Path | None:
    """同一份字型的粗體檔，找不到回傳 None。"""
    if regular is None:
        return None
    bold = _BOLD_OF.get(regular.name)
    if bold is None:
        return None
    candidate = regular.with_name(bold)
    return candidate if candidate.is_file() else None


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
    """畫字。lang 決定畫哪一種，字型決定怎麼畫。"""

    def __init__(self, font_path: Path | str | None = None, lang: str = "en"):
        self._fonts: dict[tuple[int, bool], object] = {}
        self._bold_path: Path | None = None
        self._path: Path | None = None
        self._why: str | None = None
        self._told = False
        # 預設英文：面板上的字多半是指標名稱與按鍵提示，英文短、不必擔心字型，
        # 而中文那一側隨時可以用 --display-lang zh 叫出來。
        self._lang = lang
        # 問一次就好，每幀都畫一張小圖去比太浪費。
        # OpenCV 4.x 的 Hershey 只有 ASCII，5 以後自己會畫 Unicode。θ 與 ±
        # 同屬非 ASCII，一起過關或一起不過，所以問一個就夠。
        self._hershey_unicode = hershey_can_draw("θ")

        if find_spec("PIL") is None:
            self._why = ("沒有安裝 Pillow，畫面上的文字改用英文。"
                         "要中文的話 pip install Pillow")
            return

        path = find_font(font_path)
        self._bold_path = find_bold_font(path)
        if path is None:
            self._why = ("找不到中日韓字型，畫面上的文字改用英文。"
                         "用 --font 指定一份，或安裝 Noto Sans CJK")
            return
        self._path = path

    @property
    def cjk(self) -> bool:
        """畫得出中文嗎。"""
        return self._path is not None

    @property
    def chinese(self) -> bool:
        """這一次要畫中文嗎。要中文而且畫得出來才算。"""
        return self._lang == "zh" and self.cjk

    def _pick(self, item) -> str:
        text = item.text if self.chinese else item.ascii_text
        if self.chinese or self._hershey_unicode:
            return text
        return to_ascii_symbols(text)

    @staticmethod
    def _runs(text: str):
        """把 `θ_{CA}` 拆成 [(θ, False), (CA, True)]。True 代表下標。"""
        runs, at = [], 0
        for m in _SUBSCRIPT.finditer(text):
            if m.start() > at:
                runs.append((text[at:m.start()], False))
            runs.append((m.group(1), True))
            at = m.end()
        if at < len(text):
            runs.append((text[at:], False))
        return runs or [("", False)]

    def notice(self) -> str | None:
        """沒能畫成中文的原因，只回傳一次。

        每幀都印的話會蓋掉量測本身的輸出，而這是開場就確定、之後不會變的事。
        本來就選英文的時候不必說，那不是退路，是指定的。
        """
        if self._why is None or self._told or self._lang != "zh":
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
        # 算繪器跟著語言走，不跟著字型走。英文配 Hershey 的筆畫比 TTF 的
        # Latin 粗，站在旁邊瞄比較清楚,而那正是這個視窗的用途。
        if not self.chinese:
            return self._paint_hershey(canvas, items)
        return self._paint_pillow(canvas, items)

    def _paint_hershey(self, canvas: np.ndarray, items) -> np.ndarray:
        for item in items:
            x = item.x
            for run, sub in self._runs(self._pick(item)):
                # Hershey 的字高約為 scale 的 22px，換算成要求的像素高度。
                size = item.size * (_SUB_SIZE if sub else 1.0)
                baseline = item.y + item.size + (item.size * _SUB_DROP if sub else 0)
                cv2.putText(canvas, run, (int(x), int(baseline)),
                            cv2.FONT_HERSHEY_SIMPLEX, size / 22.0, item.colour,
                            2 if item.bold else 1, cv2.LINE_AA)
                x += self._hershey_width(run, size, item.bold)
        return canvas

    @staticmethod
    def _hershey_width(text: str, size: float, bold: bool) -> int:
        (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                    size / 22.0, 2 if bold else 1)
        return int(w)

    def _paint_pillow(self, canvas: np.ndarray, items) -> np.ndarray:
        from PIL import Image, ImageDraw

        # PIL 是 RGB，cv2 是 BGR。顏色在 TextItem 裡照 cv2 的慣例寫，所以這裡
        # 轉影像也轉顏色，兩邊才對得起來。
        image = Image.fromarray(canvas[:, :, ::-1])
        draw = ImageDraw.Draw(image)
        for item in items:
            fill = tuple(int(c) for c in item.colour[::-1])
            x = item.x
            for run, sub in self._runs(self._pick(item)):
                size = int(round(item.size * _SUB_SIZE)) if sub else item.size
                y = item.y + (int(round(item.size * _SUB_DROP)) if sub else 0)
                font = self._font(size, item.bold)
                draw.text((x, y), run, font=font, fill=fill,
                          stroke_width=self._weight(item), stroke_fill=fill)
                x += font.getlength(run)
        canvas[:, :, :] = np.asarray(image)[:, :, ::-1]
        return canvas

    def _weight(self, item) -> int:
        """沒有粗體字型檔時，描一圈同色的邊假裝粗體。

        Pillow 這條路先前整個忽略 `bold`，於是狀態詞與警告跟旁邊的數字一樣細，
        而那兩個就是要讓人一眼看到的東西（Hershey 那條路一直都有）。

        描邊只當退路。中文的筆畫本來就密，描一圈會把筆畫之間的空隙填掉，
        「肩部垂直位移」會糊成一團,有真正的粗體字型就用它。
        """
        return 1 if item.bold and self._bold_path is None else 0

    def _font(self, size: int, bold: bool = False):
        bold = bold and self._bold_path is not None
        key = (size, bold)
        if key not in self._fonts:
            from PIL import ImageFont

            path = self._bold_path if bold else self._path
            self._fonts[key] = ImageFont.truetype(str(path), size)
        return self._fonts[key]

    def width(self, text: str, ascii_text: str, size: int,
              bold: bool = False) -> int:
        """一段字畫出來有多寬，用來排版。

        下標要分開量：它用的字級比較小，照主字級算會多留一截空白。
        """
        if not self.chinese:
            # 用 cv2 自己量。先前照字數估，粗體大字會少算一截，於是狀態詞會被
            # 下一段字蓋住，而狀態詞就在最前面。
            picked = (ascii_text if self._hershey_unicode
                      else to_ascii_symbols(ascii_text))
            return sum(self._hershey_width(run, size * (_SUB_SIZE if sub else 1.0), bold)
                       for run, sub in self._runs(picked))
        total = 0.0
        for run, sub in self._runs(text):
            run_size = int(round(size * _SUB_SIZE)) if sub else size
            total += self._font(run_size, bold).getlength(run)
        return int(total)
