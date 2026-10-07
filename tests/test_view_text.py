"""畫中文字，以及畫不了的時候怎麼辦。"""
from __future__ import annotations

import numpy as np
import pytest

from view.text import TextItem, TextPainter, find_font
from view_fakes import blank, painted


def _item(text="正常", ascii_text="OK", size=30) -> TextItem:
    return TextItem(x=5, y=5, text=text, ascii_text=ascii_text,
                    colour=(255, 255, 255), size=size)


def test_a_missing_font_falls_back_to_english_instead_of_failing():
    """視窗是選配的，不該因為一台機器沒裝字型就量不了。"""
    painter = TextPainter(font_path="沒有這個字型.ttf")
    assert not painter.cjk
    canvas = blank(0, width=300, height=60)
    painter.paint(canvas, [_item()])
    assert painted(canvas, 0)


def test_the_fallback_says_why_and_says_it_once():
    """每幀都印會蓋掉量測輸出，而這是開場就確定、之後不會變的事。"""
    painter = TextPainter(font_path="沒有這個字型.ttf", lang="zh")
    first = painter.notice()
    assert first is not None and "英文" in first
    assert painter.notice() is None


def test_asking_for_english_is_not_a_fallback_and_says_nothing():
    """英文是預設值。沒找到中日韓字型不是退路，是本來就不需要。"""
    assert TextPainter(font_path="沒有這個字型.ttf").notice() is None


def test_english_is_the_default():
    painter = TextPainter(font_path="沒有這個字型.ttf")
    assert not painter.chinese
    assert TextPainter(font_path="沒有這個字型.ttf", lang="zh").chinese is False


def test_a_painter_that_found_a_font_has_nothing_to_say():
    if find_font() is None:
        pytest.skip("這台機器沒有中日韓字型")
    assert TextPainter().notice() is None


def test_chinese_is_actually_drawn_when_a_font_is_available():
    if find_font() is None:
        pytest.skip("這台機器沒有中日韓字型")
    painter = TextPainter()
    assert painter.cjk
    canvas = blank(0, width=300, height=60)
    painter.paint(canvas, [_item()])
    assert painted(canvas, 0)


def test_the_two_modes_draw_different_things():
    """退回模式畫的是英文，不是把中文畫成一排問號。"""
    if find_font() is None:
        pytest.skip("這台機器沒有中日韓字型")
    cjk, ascii_only = blank(0, width=300, height=60), blank(0, width=300, height=60)
    TextPainter().paint(cjk, [_item()])
    TextPainter(font_path="無").paint(ascii_only, [_item()])
    assert not np.array_equal(cjk, ascii_only)


def test_nothing_to_draw_leaves_the_canvas_alone():
    canvas = blank(0, width=300, height=60)
    TextPainter(font_path="無").paint(canvas, [])
    assert not painted(canvas, 0)


def test_the_width_grows_with_the_text():
    """排版靠這個。估太短的話後一段會蓋住前一段，而狀態詞正好在最前面。"""
    painter = TextPainter(font_path="無")
    short = painter.width("正常", "OK", 30)
    long = painter.width("正常但是很長", "OK BUT MUCH LONGER", 30)
    assert 0 < short < long


def test_the_width_grows_with_the_size():
    painter = TextPainter(font_path="無")
    assert painter.width("正常", "OK", 20) < painter.width("正常", "OK", 40)


def test_an_explicit_font_that_does_not_exist_is_not_silently_replaced():
    """指定了字型卻被換掉的話，畫面上的字跟指定的不一樣而且沒有人會發現。"""
    assert find_font("沒有這個字型.ttf") is None


def test_bold_uses_the_bold_font_file_when_there_is_one():
    """描邊假粗會把中文筆畫之間的空隙填掉，「肩部垂直位移」糊成一團。"""
    from view.text import find_bold_font
    painter = TextPainter(lang="zh")
    if find_font() is None or find_bold_font(find_font()) is None:
        pytest.skip("這台機器沒有成對的一般／粗體字型")
    item = TextItem(x=0, y=0, text="超標", ascii_text="OVER", colour=(0, 0, 0), bold=True)
    assert painter._weight(item) == 0        # 不描邊
    assert painter._font(20, bold=True) is not painter._font(20, bold=False)


def test_without_a_bold_file_bold_falls_back_to_a_stroke():
    painter = TextPainter(font_path="沒有這個字型.ttf")
    item = TextItem(x=0, y=0, text="超標", ascii_text="OVER", colour=(0, 0, 0), bold=True)
    assert painter._weight(item) == 1


def test_a_bold_label_is_wider_than_the_same_label_regular():
    """排版照量到的寬度擺下一欄，粗體量錯的話後面整排會疊在一起。"""
    if find_font() is None:
        pytest.skip("這台機器沒有中日韓字型")
    painter = TextPainter()
    assert (painter.width("超標", "OVER", 34, bold=True)
            >= painter.width("超標", "OVER", 34, bold=False))
