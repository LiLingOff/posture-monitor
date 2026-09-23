"""雙目標定結果的合理性驗算。

RMS 低只代表標定在內部自洽，說不出參數物理上合不合理。
這些檢查是拿結果去對照「剛性雙目模組應該長什麼樣」，
用來抓出收斂到自洽但錯誤的解那種情況。
"""
import numpy as np
import pytest

from calibration.inspect_stereo import (format_report, rotation_angle_deg,
                                        sanity_checks, translation_axis_ratio)
from geometry_synthetic import make_synthetic_stereo_calibration


# 實機那組的量級：每眼 1280x720、基線約 60mm
K = np.array(
    [
        [1000.0, 0.0, 640.0],
        [0.0, 1000.0, 360.0],
        [0.0, 0.0, 1.0],
    ]
)


def _calib(baseline_mm: float = 60.0, rms: float = 0.37):
    calib = make_synthetic_stereo_calibration(K, baseline_mm, image_size=(1280, 720))
    calib.rms_error = rms
    return calib


def _by_name(checks):
    return {c.name: c for c in checks}


def test_clean_synthetic_calibration_passes_everything():
    checks = sanity_checks(_calib())
    failed = [c.name for c in checks if not c.passed]
    assert failed == [], failed


def test_rotation_angle_of_identity_is_zero():
    assert rotation_angle_deg(np.eye(3)) == pytest.approx(0.0, abs=1e-9)


def test_rotation_angle_of_known_rotation():
    import cv2

    R, _ = cv2.Rodrigues(np.array([0.0, np.radians(3.0), 0.0]))
    assert rotation_angle_deg(R) == pytest.approx(3.0, abs=1e-6)


def test_translation_axis_ratio():
    assert translation_axis_ratio(np.array([[-60.0], [0.0], [0.0]])) == pytest.approx(1.0)
    assert translation_axis_ratio(np.array([[0.0], [60.0], [0.0]])) == pytest.approx(0.0)
    assert translation_axis_ratio(np.zeros((3, 1))) == 0.0


def test_flags_rms_over_target():
    checks = _by_name(sanity_checks(_calib(rms=0.9)))
    assert checks["RMS 重投影誤差"].passed is False


def test_flags_tilted_cameras():
    """剛性模組的兩顆鏡頭不該差好幾度，差很多代表外參解錯了。"""
    import cv2

    calib = _calib()
    calib.R, _ = cv2.Rodrigues(np.array([0.0, np.radians(12.0), 0.0]))
    checks = _by_name(sanity_checks(calib))
    assert checks["兩相機夾角"].passed is False


def test_flags_translation_not_along_x():
    """左右並排的模組，平移應該幾乎只有 X 分量。"""
    calib = _calib()
    calib.T = np.array([[10.0], [59.0], [0.0]])
    checks = _by_name(sanity_checks(calib))
    assert checks["平移方向"].passed is False


def test_flags_p2_inconsistent_with_baseline():
    """P2 第四欄串起內參、外參與校正，對不上代表某一段接錯了。"""
    calib = _calib()
    calib.P2 = calib.P2.copy()
    calib.P2[0, 3] *= 1.5
    checks = _by_name(sanity_checks(calib))
    assert checks["P2 與基線一致"].passed is False


def test_flags_off_centre_principal_point():
    calib = _calib()
    calib.camera_matrix_left = calib.camera_matrix_left.copy()
    calib.camera_matrix_left[0, 2] = 5.0  # 光心跑到畫面最左邊
    checks = _by_name(sanity_checks(calib))
    assert checks["左光心位置"].passed is False


def test_report_mentions_the_square_size_trap_when_all_pass():
    """全通過時要提醒基線還得跟規格書對照，因為尺度錯了這些檢查都抓不到。"""
    report = format_report(_calib())
    assert "square-size-mm" in report
    assert "規格書" in report


def test_report_lists_failures_when_something_is_off():
    calib = _calib(rms=0.9)
    report = format_report(calib)
    assert "不符預期" in report
    assert "0.9000 px" in report


def test_flags_swapped_left_right_eyes():
    """左右眼顛倒時 T_x 會變正號，而其他所有檢查都照樣通過。

    這是唯一抓得到這件事的項目：RMS、夾角、平移比例、P2 一致性
    在顛倒的情況下數值完全一樣，看不出任何異常。
    """
    calib = _calib()
    calib.T = -np.asarray(calib.T)  # 左右對調
    checks = _by_name(sanity_checks(calib))
    assert checks["左右眼順序"].passed is False
    assert "swap-lr" in checks["左右眼順序"].detail
    for name in ("RMS 重投影誤差", "兩相機夾角", "平移方向", "左右焦距一致"):
        assert checks[name].passed is True, f"{name} 抓不到左右顛倒，本來就不該抓到"


def test_correct_order_reports_second_camera_on_the_right():
    checks = _by_name(sanity_checks(_calib()))
    assert checks["左右眼順序"].passed is True
    assert "右側" in checks["左右眼順序"].detail
