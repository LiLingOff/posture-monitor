"""終端機寬度計算。

中日韓字元佔兩欄而 len() 只算一個，所以排版與裁切都要按顯示寬度來。
這個檔案的存在理由是同一個錯誤犯過兩次：先是報告表格的中文標題排不齊，
後來是 live 那一行「裁」完仍然折行。
"""
import pytest

from geometry.terminal import cell, display_width, truncate


def test_cjk_characters_count_as_two_columns():
    assert display_width("abc") == 3
    assert display_width("關鍵點") == 6
    assert display_width("θ_CA") == 4, "希臘字母屬於寬度未定，按一欄算"


def test_cell_pads_by_display_width_not_character_count():
    assert display_width(cell("關鍵點", 16)) == 16
    assert display_width(cell("right_ear", 16)) == 16


def test_cell_leaves_longer_text_alone():
    assert cell("一二三四五", 4) == "一二三四五"


def test_truncate_measures_in_columns():
    """text[:width] 留下來的字數雖然對，佔的欄數是兩倍，照樣會折行。"""
    text = "略過：深度 74mm 落在桌前坐姿的合理範圍外"
    cut = truncate(text, 20)
    assert display_width(cut) <= 20
    assert len(cut) < 20, "純用字元數裁的話這裡會等於 20"


def test_truncate_never_splits_a_wide_character_in_half():
    assert display_width(truncate("一二三", 5)) == 4


def test_truncate_leaves_short_text_alone():
    assert truncate("abc", 10) == "abc"


@pytest.mark.parametrize("text", ["", "a", "中", "θ_CA +12.3±1.7(單幀 -2.7)"])
def test_truncating_to_its_own_width_is_a_no_op(text):
    assert truncate(text, display_width(text)) == text


def test_a_mixed_heading_line_is_measured_by_display_width():
    """中英混排的標題用字元數估寬度會算錯，橫線就短一截。

    「逐受試者的 θ_CA」是 5 個中文字（各佔兩欄）加空白與 θ_CA：顯示寬度 15，
    而 len() 是 10，乘二得 20。差的那 5 欄就是線短掉的長度。

    θ 的 east_asian_width 是 Ambiguous，這裡當一欄算，那是多數終端機的行為。
    """
    from geometry.terminal import display_width

    assert display_width("逐受試者的 θ_CA") == 15
    assert len("逐受試者的 θ_CA") == 10


def test_headings_in_both_reports_come_out_the_same_length():
    """兩個報表的區段標題要對齊，不然同一份輸出看起來像兩個程式拼的。"""
    from geometry.cohort_report import _heading
    from geometry.terminal import display_width

    widths = {display_width(_heading(t))
              for t in ("逐段", "逐受試者的 θ_CA", "姿勢之間的差距")}
    assert len(widths) == 1, f"標題長度不一致：{widths}"


# ---- 值與誤差 ----------------------------------------------------------

def test_the_unit_goes_on_the_last_number_only():
    """+19.61 ± 0.80°。每個數字後面都掛一個單位讀起來像兩個不相干的量，
    而那是同一個量的中心與寬度。"""
    from geometry.terminal import with_error
    assert with_error(19.61, 0.8) == "+19.61 ± 0.80°"
    assert with_error(-7.52, 0.32) == "-7.52 ± 0.32°"


def test_without_an_error_the_unit_stays_on_the_value():
    from geometry.terminal import with_error
    assert with_error(19.61, None) == "+19.61°"


def test_the_unit_is_not_always_degrees():
    from geometry.terminal import with_error
    assert with_error(18.0, 2.0, "mm", 1) == "+18.0 ± 2.0mm"


def test_a_missing_value_is_a_dash_not_a_zero():
    from geometry.terminal import with_error
    assert with_error(None, None) == "—"


def test_the_sign_can_be_turned_off():
    """門檻與散佈這類本來就是正的，加號只是雜訊。"""
    from geometry.terminal import with_error
    assert with_error(10.0, None, signed=False) == "10.00°"
