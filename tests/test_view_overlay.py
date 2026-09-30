"""疊圖的測試。不開視窗，斷言看的是像素。"""
from __future__ import annotations

import numpy as np
import pytest

from geometry.judgement import Posture
from view import overlay
from view.text import TextPainter
from view_fakes import SEATED, blank, painted, person, view

# 退回英文的畫字器。面板的測試要能在沒有中日韓字型的機器上跑。
ASCII = TextPainter(font_path="沒有這個字型.ttf")


def _midpoint(a: str, b: str) -> tuple[float, float]:
    ax, ay = SEATED[a]
    bx, by = SEATED[b]
    return (ax + bx) / 2, (ay + by) / 2


# ---- 骨架 --------------------------------------------------------------

def test_a_bone_is_drawn_between_two_detected_keypoints():
    canvas = blank()
    overlay.draw_bones(canvas, person(), 1.0)
    x, y = _midpoint("right_shoulder", "right_elbow")
    assert canvas[int(y), int(x)].tolist() != [40, 40, 40]


def test_a_bone_with_one_missing_end_is_not_drawn():
    """NaN 轉 int 會變成一個極大的整數，cv2 會拿它畫一條穿過整張圖的線。"""
    canvas = blank()
    overlay.draw_bones(canvas, person(only=("right_shoulder",)), 1.0)
    assert not painted(canvas, 40)


def test_no_keypoints_at_all_does_not_raise():
    canvas = blank()
    overlay.draw_bones(canvas, person(only=()), 1.0)
    overlay.draw_keypoints(canvas, person(only=()), 1.0)
    assert not painted(canvas, 40)


def test_a_missing_person_does_not_raise():
    canvas = blank()
    overlay.draw_bones(canvas, None, 1.0)
    overlay.draw_keypoints(canvas, None, 1.0)
    overlay.label_keypoints(canvas, None, 1.0)
    assert not painted(canvas, 40)


def test_points_are_scaled_with_the_frame():
    canvas = blank(width=320, height=240)
    overlay.draw_bones(canvas, person(), 0.5)
    x, y = _midpoint("right_shoulder", "right_elbow")
    assert canvas[int(y * 0.5), int(x * 0.5)].tolist() != [40, 40, 40]


# ---- 量角度的那幾段 ----------------------------------------------------

@pytest.mark.parametrize("side", ["left", "right"])
def test_the_theta_ca_segment_follows_the_side_the_measurement_used(side):
    """側邊寫死的話會有一半的幀標錯，而標錯的地方看起來跟標對一模一樣。"""
    canvas = blank()
    drawn = overlay.draw_segment(canvas, person(), f"{side}_ear", f"{side}_shoulder",
                                 1.0, (0, 0, 255))
    assert drawn
    x, y = _midpoint(f"{side}_ear", f"{side}_shoulder")
    assert canvas[int(y), int(x)].tolist() != [40, 40, 40]

    other = "left" if side == "right" else "right"
    ox, oy = _midpoint(f"{other}_ear", f"{other}_shoulder")
    assert canvas[int(oy), int(ox)].tolist() == [40, 40, 40]


def test_a_segment_with_a_missing_end_reports_that_it_drew_nothing():
    canvas = blank()
    assert not overlay.draw_segment(canvas, person(only=("right_ear",)),
                                    "right_ear", "right_shoulder", 1.0, (0, 0, 255))
    assert not painted(canvas, 40)


def test_the_shoulder_line_is_drawn_without_a_theta_ca_side():
    canvas = blank()
    assert overlay.draw_segment(canvas, person(), "left_shoulder", "right_shoulder",
                                1.0, (0, 0, 255))


# ---- 標記 --------------------------------------------------------------

def test_a_mispaired_keypoint_is_marked():
    canvas = blank()
    overlay.draw_keypoints(canvas, person(), 1.0, mispaired=("right_shoulder",))
    x, y = SEATED["right_shoulder"]
    assert overlay.MISPAIRED in [tuple(int(v) for v in canvas[y + dy, x + dx])
                                 for dy in range(-8, 9) for dx in range(-8, 9)]


def test_an_edge_keypoint_is_marked():
    canvas = blank()
    overlay.draw_keypoints(canvas, person(), 1.0, edge=("left_wrist",))
    x, y = SEATED["left_wrist"]
    assert overlay.EDGE in [tuple(int(v) for v in canvas[y + dy, x + dx])
                            for dy in range(-8, 9) for dx in range(-8, 9)]


def test_a_point_that_is_both_pinned_and_mispaired_is_marked_as_mispaired():
    """順序寫死，否則這兩條測試會變成依賴繪製順序。配對錯誤比貼邊嚴重：
    那一幀的角度是錯的，不是缺的。"""
    canvas = blank()
    overlay.draw_keypoints(canvas, person(), 1.0,
                           edge=("right_shoulder",), mispaired=("right_shoulder",))
    x, y = SEATED["right_shoulder"]
    nearby = [tuple(int(v) for v in canvas[y + dy, x + dx])
              for dy in range(-8, 9) for dx in range(-8, 9)]
    assert overlay.MISPAIRED in nearby
    assert overlay.EDGE not in nearby


# ---- 並排與對極輔助線 --------------------------------------------------

def test_the_two_eyes_are_composed_side_by_side_each_with_its_own_frame():
    """並排的用意全靠這一條：兩半要各自來自自己那一隻眼。"""
    out = overlay.compose(view(), 1280, ASCII)
    picture = out[:out.shape[0] - overlay.PANEL_HEIGHT]
    assert out.shape[1] == 1280
    # 取左右兩半的角落，那裡不會被骨架蓋到。
    assert picture[5, 5].tolist() == [40, 40, 40]
    assert picture[5, 1275].tolist() == [90, 90, 90]


def test_only_the_left_eye_is_drawn_when_asked_for_one():
    out = overlay.compose(view(), 1280, ASCII, eyes="left")
    picture = out[:out.shape[0] - overlay.PANEL_HEIGHT]
    assert picture[5, 5].tolist() == [40, 40, 40]
    assert picture[5, 1275].tolist() == [40, 40, 40]


def test_each_eye_gets_its_own_keypoints():
    """兩眼畫同一組點的話，配對錯誤在畫面上就消失了，而那正是要看的東西。"""
    moved = person(right_shoulder=(0.0, 60.0))
    out = overlay.compose(view(right=moved), 1280, ASCII, eyes="both")
    x, y = SEATED["right_shoulder"]
    half = out.shape[1] // 2
    scale = half / 640
    left_half, right_half = out[:, :half], out[:, half:]
    # 左眼的點在原本的高度，右眼的往下 60px。畫同一組點的話這兩條就都不成立。
    assert left_half[int(y * scale), int(x * scale)].tolist() != [40, 40, 40]
    assert left_half[int((y + 60) * scale), int(x * scale)].tolist() == [40, 40, 40]
    assert right_half[int((y + 60) * scale), int(x * scale)].tolist() != [90, 90, 90]


def test_the_epipolar_guide_spans_both_eyes():
    """判別配對錯誤靠的是垂直視差，而並排的版面讓垂直方向最難比。"""
    canvas = blank(fill=40, width=1280, height=480)
    overlay.draw_epipolar_guides(canvas, person(), person(),
                                 ("left_shoulder", "right_shoulder"), 1.0)
    y = SEATED["right_shoulder"][1]
    assert canvas[y, 5].tolist() != [40, 40, 40]
    assert canvas[y, 1275].tolist() != [40, 40, 40]


def test_a_mispaired_guide_line_is_the_alarm_colour():
    canvas = blank(fill=40, width=1280, height=480)
    overlay.draw_epipolar_guides(canvas, person(), person(), ("right_shoulder",),
                                 1.0, mispaired=("right_shoulder",))
    plain = blank(fill=40, width=1280, height=480)
    overlay.draw_epipolar_guides(plain, person(), person(), ("right_shoulder",), 1.0)
    y = SEATED["right_shoulder"][1]
    # LINE_AA 會混色，所以比的是「哪一條比較紅」而不是精確的顏色值。
    assert int(canvas[y, 640][2]) > int(plain[y, 640][2]) + 100


def test_drawing_does_not_modify_the_camera_frame():
    """同一張 frames[0] 接著要交給 Snapshots。就地畫下去會把即時疊圖混進
    事後查證用的證據裡。"""
    left, right = blank(40), blank(90)
    before_left, before_right = left.copy(), right.copy()
    overlay.compose(view(left_bgr=left, right_bgr=right), 1280, ASCII)
    assert np.array_equal(left, before_left)
    assert np.array_equal(right, before_right)


def test_a_window_narrower_than_the_minimum_is_widened():
    out = overlay.compose(view(), 200, ASCII)
    assert out.shape[1] == overlay.MIN_WIDTH


# ---- 面板 --------------------------------------------------------------

def _texts(v) -> list[str]:
    return [cell[0] for row in overlay.panel_rows(v) for cell in row]


def _ascii_texts(v) -> list[str]:
    return [cell[1] for row in overlay.panel_rows(v) for cell in row]


def test_every_posture_has_a_word_and_a_colour():
    for posture in Posture:
        assert posture in overlay.STATE_WORDS
        assert posture in overlay.STATE_COLOURS


def test_the_state_word_comes_first_and_is_bigger_than_the_numbers():
    rows = overlay.panel_rows(view(state=Posture.OVER))
    word, ascii_word, colour, size, bold = rows[0][0]
    assert word == "超標" and ascii_word == "OVER"
    assert colour == overlay.STATE_COLOURS[Posture.OVER]
    assert bold
    assert size > rows[0][1][3]


def test_a_missing_angle_is_a_dash_not_a_zero():
    """0 是合法的角度。寫成 0 的話「量不到」與「剛好是零」長得一樣。"""
    texts = " ".join(_texts(view(theta_ca_mean_deg=None)))
    assert "—" in texts
    assert "0.0" not in texts


def test_the_panel_is_ascii_only_in_the_fallback():
    """沒有字型時整個面板靠英文。中文會被 Hershey 畫成一排問號。"""
    v = view(state=Posture.OVER, turned_deg=25.0,
             skip_reason="right_shoulder 的垂直視差 43.3px，左右配對錯了",
             keys=(("c 重新歸零", "c zero"), ("q 離開", "q quit")),
             theta_ca_mean_deg=19.6,
             theta_ca_error_deg=0.8, distance_mm=662.0, rejected=12,
             remaining_s=45.0)
    for text in _ascii_texts(v):
        assert text.isascii(), text


@pytest.mark.parametrize("keys", [
    (("q 離開", "q quit"),),
    (("任意鍵關閉", "press any key"),),
    (("c 重新歸零", "c zero"), ("q 離開", "q quit")),
])
def test_every_key_hint_the_cli_passes_has_an_ascii_form(keys):
    """先前只存中文、畫英文時從第一個字猜，於是 once 的「任意鍵關閉」在退回
    模式下整段變成問號。這裡列的是 posture.py 真正傳進來的那幾組。"""
    for text in _ascii_texts(view(keys=keys)):
        assert text.isascii(), text


def test_turned_is_the_change_from_the_baseline_not_the_azimuth():
    """這個數字的全部價值就在於它是個差值。"""
    texts = " ".join(_texts(view(turned_deg=32.0)))
    assert "32" in texts


def test_turned_is_blank_without_a_baseline():
    texts = " ".join(_texts(view(turned_deg=None)))
    assert "轉身 —" in texts


def test_a_big_turn_is_as_loud_as_the_state_word():
    """這個視窗本身就是引人轉頭去看的東西，提示比狀態詞小等於沒提示。"""
    small = overlay.panel_rows(view(turned_deg=2.0))[1][1]
    big = overlay.panel_rows(view(turned_deg=40.0))[1][1]
    assert big[3] == overlay.STATE_SIZE > small[3]
    assert big[2] == overlay.MISPAIRED
    assert "請轉回正面" in big[0]


def test_a_skipped_frame_still_shows_the_distance_and_the_reason():
    """調整架設位置時正是略過最多的時候，那幾個數字不能跟著消失。"""
    texts = " ".join(_texts(view(distance_mm=662.0,
                                 skip_reason="左右只有 3 個共同關鍵點")))
    assert "662" in texts
    assert "共同關鍵點" in texts


@pytest.mark.parametrize("reason,tag", [
    ("深度 104mm 落在桌前坐姿的合理範圍外（200~3000mm）", "DEPTH"),
    ("深度 -5138mm 是負的，左右配對接反了", "SWAPPED"),
    ("left_shoulder 的垂直視差 8.4px，左右配對錯了", "MISPAIR"),
    ("左右只有 3 個共同關鍵點", "FEW POINTS"),
    ("左眼沒有偵測到人。確認受試者在畫面內、光線足夠", "NO PERSON"),
    ("角度用到的 left_shoulder 貼在畫面邊緣，真實位置在畫面外", "EDGE"),
    ("沒有可用的耳朵或肩膀深度", "NO DEPTH"),
    ("讀取相機影格失敗", "NO FRAME"),
])
def test_every_rejection_reason_has_its_own_ascii_tag(reason, tag):
    assert overlay.reason_tag(reason) == tag


def test_an_unknown_reason_gets_a_generic_tag_rather_than_a_guess():
    assert overlay.reason_tag("某個沒見過的原因") == "SKIP"
    assert overlay.reason_tag(None) == ""


def test_the_stale_band_says_there_is_no_data():
    """看起來還活著的凍結畫面是這個專案一再吃虧的那一類。"""
    image = overlay.compose(view(), 1280, ASCII)
    marked = overlay.mark_stale(image, ASCII)
    assert marked.shape == image.shape
    assert not np.array_equal(marked[:44], image[:44])
    assert np.array_equal(marked[44:], image[44:])


def test_keypoint_labels_are_only_for_saved_frames():
    """即時畫面兩眼各佔一半，十八個標籤會疊成一團。"""
    canvas = blank()
    overlay.label_keypoints(canvas, person(), 1.0)
    assert painted(canvas, 40)
