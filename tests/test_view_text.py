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
    """英文那一邊畫的是英文，不是把中文畫成一排問號。"""
    if find_font() is None:
        pytest.skip("這台機器沒有中日韓字型")
    cjk, english = blank(0, width=300, height=60), blank(0, width=300, height=60)
    TextPainter(lang="zh").paint(cjk, [_item()])
    TextPainter().paint(english, [_item()])
    assert not np.array_equal(cjk, english)


def test_english_goes_through_hershey_even_when_a_font_exists():
    """算繪器跟著語言走。英文配 Hershey 的筆畫比 TTF 的 Latin 粗，
    站在旁邊瞄比較清楚,而那正是這個視窗的用途。"""
    if find_font() is None:
        pytest.skip("這台機器沒有中日韓字型")
    with_font, without = blank(0, width=300, height=60), blank(0, width=300, height=60)
    TextPainter().paint(with_font, [_item()])
    TextPainter(font_path="無").paint(without, [_item()])
    assert np.array_equal(with_font, without)


def test_a_subscript_is_drawn_smaller_and_lower():
    """報告書寫的是真正的下標。Unicode 的小型大寫在微軟正黑體裡整排缺字。"""
    plain, sub = blank(0, width=300, height=60), blank(0, width=300, height=60)
    item = TextItem(x=5, y=5, text="θ_{CA}", ascii_text="θ_{CA}",
                    colour=(255, 255, 255), size=28)
    TextPainter(font_path="無").paint(sub, [item])
    TextPainter(font_path="無").paint(
        plain, [TextItem(x=5, y=5, text="θCA", ascii_text="θCA",
                         colour=(255, 255, 255), size=28)])
    assert painted(sub, 0)
    assert not np.array_equal(plain, sub)
    # 下標在主字的下半部：上緣那幾列只有 θ，不會有 CA。
    assert sub[5:12].sum() < plain[5:12].sum()


def test_the_subscript_markup_is_split_into_runs():
    runs = TextPainter._runs("θ_{CA} +19.6°")
    assert runs == [("θ", False), ("CA", True), (" +19.6°", False)]
    assert TextPainter._runs("距離 587mm") == [("距離 587mm", False)]


def test_theta_becomes_the_word_when_the_renderer_cannot_draw_it():
    """OpenCV 4 的 Hershey 只有 ASCII，θ 會變成兩個問號。能不能畫是算繪器
    才知道的事，所以問它，不要猜。"""
    from view.text import to_ascii_symbols
    assert to_ascii_symbols("θ_{CA} ± 0.8") == "theta_{CA} +- 0.8"
    assert to_ascii_symbols("Distance 587mm") == "Distance 587mm"


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


def test_plus_minus_survives_when_the_renderer_can_draw_it():
    """± 與 θ 同屬非 ASCII，一起過關或一起不過。"""
    painter = TextPainter(font_path="無")
    item = TextItem(x=0, y=0, text="± 0.8", ascii_text="± 0.8", colour=(1, 1, 1))
    picked = painter._pick(item)
    assert picked == ("± 0.8" if painter._hershey_unicode else "+- 0.8")
