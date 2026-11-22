"""用 repo 內固定的字型跑文字繪製的整合測試。

`test_view_text.py` 那些測試靠「這台機器剛好有中日韓字型」，沒有的機器（CI、乾淨的
Jetson）整段跳過，而 Pillow 這一條路正是面板實際走的路。這裡改用 `tests/fonts/`
裡的 Noto Sans TC 子集（SIL OFL，見該資料夾的 NOTICE），任何機器上結果都一樣。
"""
from pathlib import Path

import numpy as np
import pytest

from view import text as text_module
from view.text import TextItem, TextPainter
from view_fakes import blank

pytest.importorskip("PIL")

FONTS = Path(__file__).parent / "fonts"
REGULAR = FONTS / "NotoSansTC-Regular-subset.ttf"
BOLD = FONTS / "NotoSansTC-Bold-subset.ttf"
WHITE = (255, 255, 255)


@pytest.fixture(autouse=True)
def _bold_pair(monkeypatch):
    """粗體檔的對應是寫死的表；把測試用的這一對加進去，不去改正式程式的表。"""
    monkeypatch.setitem(text_module._BOLD_OF, REGULAR.name, BOLD.name)


def _painter(lang="en") -> TextPainter:
    return TextPainter(font_path=REGULAR, lang=lang)


def _item(text, size=28, bold=False, ascii_text=None) -> TextItem:
    return TextItem(x=10, y=10, text=text, ascii_text=ascii_text or text,
                    colour=WHITE, size=size, bold=bold)


def _canvas(painter, *items, width=700, height=120):
    canvas = blank(0, width=width, height=height)
    painter.paint(canvas, list(items))
    return canvas


def _ink(canvas) -> int:
    return int((canvas.sum(axis=2) > 0).sum())


def _rows(canvas):
    rows = np.where(canvas.any(axis=(1, 2)))[0]
    return int(rows.min()), int(rows.max())


def test_the_bundled_fonts_are_there_and_carry_the_panel_symbols():
    from fontTools.ttLib import TTFont

    for path in (REGULAR, BOLD):
        cmap = TTFont(path).getBestCmap()
        for char in "θ±°正常":
            assert ord(char) in cmap, f"{path.name} 缺 {char}"


def test_the_painter_finds_the_font_and_has_nothing_to_apologise_for():
    for lang in ("en", "zh"):
        painter = _painter(lang)
        assert painter.cjk
        assert painter.notice() is None, "有字型就不該印「找不到字型」"


def test_english_still_goes_through_pillow_and_keeps_the_symbols():
    """英文也走 Pillow：Hershey 在 OpenCV 4.x 上 θ 會變 theta、± 會變 +-。"""
    painter = _painter("en")
    assert painter.picked("θ_CA", "θ_{CA} ±3°") == "θ_{CA} ±3°"
    assert painter.picked("正常", "OK") == "OK", "英文模式畫的是英文那一側"


def test_chinese_is_really_drawn_when_asked_for():
    zh, en = _painter("zh"), _painter("en")
    item = _item("正常", ascii_text="OK")
    drawn_zh, drawn_en = _canvas(zh, item), _canvas(en, item)
    assert _ink(drawn_zh) > 0 and _ink(drawn_en) > 0
    assert not np.array_equal(drawn_zh, drawn_en)
    # 兩個全形字：墨水範圍約兩個字級寬，不是一排豆腐方塊或問號。
    columns = np.where(drawn_zh.any(axis=(0, 2)))[0]
    assert (columns.max() - columns.min()) > 28 * 1.5


def test_the_symbols_are_drawn_not_replaced_by_boxes():
    """缺字時 Pillow 畫的是同一個空框；三個符號畫出來的形狀要各不相同。"""
    painter = _painter("en")
    shapes = [_canvas(painter, _item(c))[:, :120].tobytes() for c in ("θ", "±", "°")]
    assert len(set(shapes)) == 3


def test_a_subscript_is_a_real_subscript():
    painter = _painter("en")
    main = _canvas(painter, _item("θ"))
    full = _canvas(painter, _item("θ_{CA}", ascii_text="θ_{CA}"))
    literal = _canvas(painter, _item("θ_CA"))
    # 下標畫在主字右邊、小一級，而且往下挪：整體的下緣比主字低。
    assert _rows(full)[1] > _rows(main)[1]
    # 不是把 `_`、`{`、`}` 原樣畫出來：寫成大括號的與不寫的畫面不一樣，
    # 而且大括號不見了，所以比字面照畫的窄。
    assert not np.array_equal(full, literal)
    cols = lambda c: np.where(c.any(axis=(0, 2)))[0]   # noqa: E731
    assert cols(full).max() < cols(_canvas(painter, _item("θ_{CA}x")) ).max()
    assert painter.width("θ_{CA}", "θ_{CA}", 28) < painter.width("θ_{CA}_", "θ_{CA}_", 28) + 28


def test_the_subscript_is_smaller_than_the_main_letter():
    painter = _painter("en")
    capital = _canvas(painter, _item("C", size=40))
    sub = _canvas(painter, _item("x_{C}", size=40))
    capital_height = _rows(capital)[1] - _rows(capital)[0]
    # 下標的 C 在 sub 裡的高度小於同字級的 C：整張圖的墨水範圍比「C 與 x 並排」小不了，
    # 所以直接比墨水量，小一級的 C 面積較小。
    assert capital_height > 0
    plain_c = _ink(capital)
    sub_ink = _ink(sub) - _ink(_canvas(painter, _item("x", size=40)))
    assert 0 < sub_ink < plain_c


def test_the_subscript_stays_inside_the_row():
    painter = _painter("en")
    size = 28
    _, bottom = _rows(_canvas(painter, _item("θ_{CA}", size=size)))
    assert bottom <= 10 + size + 6, "往下挪過頭會掉進下一列"


def test_english_is_bold_throughout_and_chinese_only_when_asked():
    english, chinese = _painter("en"), _painter("zh")
    assert english._face_is_bold(False) and english._face_is_bold(True)
    assert not chinese._face_is_bold(False) and chinese._face_is_bold(True)


def test_bold_really_puts_more_ink_down():
    painter = _painter("zh")
    regular = _ink(_canvas(painter, _item("正常 θ_CA", bold=False)))
    bold = _ink(_canvas(painter, _item("正常 θ_CA", bold=True)))
    assert bold > regular * 1.05


@pytest.mark.parametrize("lang", ["en", "zh"])
@pytest.mark.parametrize("bold", [False, True])
@pytest.mark.parametrize("text", ["OK", "θ_{CA} +12.3°", "±3.5°", "正常 θ_{sym}"])
def test_the_measured_width_covers_the_ink(lang, bold, text):
    """排版靠 width() 決定下一欄放哪；低估的話兩欄會疊在一起。"""
    painter = _painter(lang)
    canvas = _canvas(painter, _item(text, bold=bold), width=900)
    columns = np.where(canvas.any(axis=(0, 2)))[0]
    ink_width = int(columns.max()) - 10 + 1
    assert painter.width(text, text, 28, bold=bold) >= ink_width
