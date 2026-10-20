"""尺度驗證：算出來的 mm 是不是真的 mm。

合成測試證明數學對、標定驗算證明參數自洽,但整組參數等比例縮放的話那兩項
都照樣過關，而每一個距離都會錯同一個倍率。這個檔案測的就是「抓不抓得到那個
倍率」，所以這裡刻意把板子縮放過再餵進去。
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from calibration.charuco import CharucoBoardSpec
from charuco_synthetic import (frontal_to_object_homography,
                               make_charuco_frontal_image,
                               synthesize_charuco_view)
from geometry.scale_check import (adjacent_pairs, check_scale,
                                  format_scale_check)
from geometry_synthetic import make_synthetic_stereo_calibration
from posture_loader import load_posture_module

SQUARES_X, SQUARES_Y = 6, 5
SQUARE_MM = 25.0


def _flat_board(square_mm: float = SQUARE_MM, missing=()) -> dict:
    """一塊攤平在 z=600 的板子，角點剛好隔 square_mm。"""
    cols, rows = SQUARES_X - 1, SQUARES_Y - 1
    points = {}
    for corner in range(cols * rows):
        col, row = corner % cols, corner // cols
        if corner in missing:
            continue
        points[corner] = np.array([col * square_mm, row * square_mm, 600.0])
    return points


# ---- 哪些角點算相鄰 ----------------------------------------------------

def test_a_full_board_has_both_directions():
    """內部角點是 (x-1)×(y-1) 個，橫向 (cols-1)×rows 段、縱向 cols×(rows-1) 段。"""
    cols, rows = SQUARES_X - 1, SQUARES_Y - 1
    pairs = adjacent_pairs(range(cols * rows), SQUARES_X, SQUARES_Y)
    horizontal = [p for p in pairs if p[2]]
    assert len(horizontal) == (cols - 1) * rows
    assert len(pairs) - len(horizontal) == cols * (rows - 1)


def test_corners_on_opposite_edges_are_not_neighbours():
    """編號只差 1 不代表相鄰,一列的最後一個與下一列的第一個差 1，但它們
    在板子的兩端。不擋掉的話會多出一段橫跨整塊板子的假距離。"""
    cols = SQUARES_X - 1
    pairs = adjacent_pairs(range(cols * 2), SQUARES_X, SQUARES_Y)
    assert (cols - 1, cols, True) not in pairs


def test_a_missing_corner_takes_its_pairs_with_it():
    """板子常常只看得到一部分，缺的角點不該變成一段 NaN。"""
    full = adjacent_pairs(_flat_board(), SQUARES_X, SQUARES_Y)
    holed = adjacent_pairs(_flat_board(missing=(7,)), SQUARES_X, SQUARES_Y)
    assert len(holed) < len(full)
    assert all(7 not in (a, b) for a, b, _ in holed)


# ---- 量出來的尺度 ------------------------------------------------------

def test_a_board_that_is_exactly_right_reports_no_error():
    check = check_scale([_flat_board()], SQUARES_X, SQUARES_Y, SQUARE_MM)
    assert check.measured_mm == pytest.approx(SQUARE_MM)
    assert check.relative_error == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("factor", [0.95, 1.05, 1.20])
def test_a_scaled_board_is_caught_with_the_right_sign_and_size(factor):
    """整組參數等比例縮放時，重投影誤差一樣漂亮,這一項是唯一會叫的。"""
    check = check_scale([_flat_board(SQUARE_MM * factor)],
                        SQUARES_X, SQUARES_Y, SQUARE_MM)
    assert check.relative_error == pytest.approx(factor - 1.0, abs=1e-9)


def test_several_frames_all_go_into_the_same_bag():
    """單幀的角點偵測本身有雜訊，而要看的是系統性的偏差。"""
    one = check_scale([_flat_board()], SQUARES_X, SQUARES_Y, SQUARE_MM)
    three = check_scale([_flat_board()] * 3, SQUARES_X, SQUARES_Y, SQUARE_MM, frames=3)
    assert three.corner_pairs == one.corner_pairs * 3
    assert three.measured_mm == pytest.approx(one.measured_mm)


def test_nothing_detected_is_not_a_crash():
    """板子沒入鏡、反光、只有一眼看得到,這些都會走到這裡。"""
    check = check_scale([{}, {}], SQUARES_X, SQUARES_Y, SQUARE_MM, frames=2)
    assert check.corner_pairs == 0
    assert check.measured_mm is None
    assert "一對相鄰角點都沒量到" in " ".join(format_scale_check(check))


def test_one_pair_has_no_spread_to_report():
    """一段距離算不出散佈。回傳 None 而不是 0,0 是「完全沒有誤差」。"""
    check = check_scale([{0: np.zeros(3), 1: np.array([SQUARE_MM, 0.0, 600.0])}],
                        SQUARES_X, SQUARES_Y, SQUARE_MM)
    assert check.corner_pairs == 1
    assert check.spread_mm is None and check.error_mm is None


def test_the_two_directions_are_counted_apart():
    """尺度錯會讓兩邊一起錯。只有一邊不對的話問題在校正，不在尺度。"""
    points = _flat_board()
    cols = SQUARES_X - 1
    for corner, point in points.items():       # 只把縱向拉長
        point[1] *= 1.10
    check = check_scale([points], SQUARES_X, SQUARES_Y, SQUARE_MM)
    assert check.horizontal.mean() == pytest.approx(SQUARE_MM)
    assert check.vertical.mean() == pytest.approx(SQUARE_MM * 1.10)
    assert check.axes_differ_mm == pytest.approx(SQUARE_MM * 0.10)
    assert cols > 1


# ---- 報告本身 ----------------------------------------------------------

def test_a_pass_says_the_geometry_can_be_trusted_so_far():
    text = " ".join(format_scale_check(
        check_scale([_flat_board()], SQUARES_X, SQUARES_Y, SQUARE_MM)))
    assert "尺度正確" in text
    assert "沒有經過姿態模型" in text


def test_a_fail_gives_the_number_to_multiply_by():
    """知道偏了幾個百分比還不夠，要直接說該乘多少才對。"""
    text = " ".join(format_scale_check(
        check_scale([_flat_board(SQUARE_MM * 1.10)],
                    SQUARES_X, SQUARES_Y, SQUARE_MM)))
    assert "+10.00%" in text
    assert "0.9091" in text
    assert "θ_CA 與 θ_sym 是角度" in text


def test_a_fail_points_at_the_most_likely_cause():
    """標定時填的方格邊長與板子實際的邊長不符,這是唯一會讓尺度整組偏掉
    而其他檢查都不叫的原因。"""
    text = " ".join(format_scale_check(
        check_scale([_flat_board(SQUARE_MM * 1.10)],
                    SQUARES_X, SQUARES_Y, SQUARE_MM)))
    assert "--square-size-mm" in text


# ---- 整條路徑：合成影像 → 偵測 → 三角測量 → 尺度 ----------------------

def _synthetic_pair(spec: CharucoBoardSpec, baseline_mm: float = 60.0,
                    distance_mm: float = 600.0, size=(1280, 720)):
    """同一塊板子在左右兩眼的畫面，外加對應的標定。"""
    K = np.array([[900.0, 0.0, size[0] / 2],
                  [0.0, 900.0, size[1] / 2],
                  [0.0, 0.0, 1.0]])
    frontal = make_charuco_frontal_image(spec)
    to_object = frontal_to_object_homography(frontal, spec)
    # 板子擺在畫面中央偏左上，整塊都進得了兩眼的視野。
    centre_x = -spec.squares_x * spec.square_size_mm / 2
    centre_y = -spec.squares_y * spec.square_size_mm / 2
    rvec = np.zeros(3)
    frames = []
    for shift in (0.0, -baseline_mm):
        tvec = np.array([centre_x + shift, centre_y, distance_mm])
        gray = synthesize_charuco_view(frontal, to_object, K, rvec, tvec, size)
        frames.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    return make_synthetic_stereo_calibration(K, baseline_mm, size), frames


def test_the_whole_path_recovers_the_square_size_from_two_views():
    """偵測、按編號配對、三角測量、量距離,整條串起來才證明得了尺度。"""
    spec = CharucoBoardSpec(squares_x=SQUARES_X, squares_y=SQUARES_Y,
                            square_size_mm=SQUARE_MM, marker_size_mm=18.0)
    calib, (left, right) = _synthetic_pair(spec)
    posture = load_posture_module()
    corners = posture._board_corners_3d(calib, spec.build_board(), left, right)
    assert len(corners) >= (SQUARES_X - 1) * (SQUARES_Y - 1) - 2

    check = check_scale([corners], SQUARES_X, SQUARES_Y, SQUARE_MM)
    assert check.measured_mm == pytest.approx(SQUARE_MM, rel=0.01)
    assert abs(check.relative_error) < 0.01


def test_only_corners_both_eyes_saw_are_used():
    """靠編號配對，不靠位置。一眼看不到的角點不該被湊成一對。"""
    spec = CharucoBoardSpec(squares_x=SQUARES_X, squares_y=SQUARES_Y,
                            square_size_mm=SQUARE_MM, marker_size_mm=18.0)
    calib, (left, right) = _synthetic_pair(spec)
    posture = load_posture_module()
    blank = np.full_like(right, 200)
    assert posture._board_corners_3d(calib, spec.build_board(), left, blank) == {}


def test_the_report_has_no_markdown_in_it():
    """報告是印在終端機上的，星號不會變粗體，只會原樣印出來。
    指導語那邊踩過同一個坑（2026-10-01）。"""
    for check in (check_scale([_flat_board()], SQUARES_X, SQUARES_Y, SQUARE_MM),
                  check_scale([_flat_board(SQUARE_MM * 1.1)], SQUARES_X, SQUARES_Y, SQUARE_MM),
                  check_scale([{}], SQUARES_X, SQUARES_Y, SQUARE_MM)):
        for line in format_scale_check(check):
            for mark in ("**", "__", "`"):
                assert mark not in line, line
