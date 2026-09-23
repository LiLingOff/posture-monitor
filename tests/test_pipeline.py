"""端到端量測：從左右兩組關鍵點到角度與品質指標。

不需要相機或模型——直接餵已知的3D姿勢投影出來的關鍵點。
"""
import numpy as np
import pytest

from geometry.pipeline import (format_measurement, measure_posture,
                               plausibility_warnings)
from geometry_synthetic import make_synthetic_stereo_calibration, project_point
from pose.keypoints import PersonKeypoints
from pose.topology import COCO18_KEYPOINT_NAMES, NUM_KEYPOINTS

K = np.array([[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]])
BASELINE_MM = 60.0
T_LEFT = np.zeros((3, 1))
T_RIGHT = np.array([[-BASELINE_MM], [0.0], [0.0]])


def _seated_pose(theta_ca_deg: float = 15.0, shoulder_tilt_deg: float = 0.0) -> dict:
    """距離相機600mm的坐姿，耳朵相對肩膀前傾theta_ca度。

    相機架在正面、受試者面向相機，所以「往前伸」是朝相機的方向，也就是-Z
    （OpenCV慣例下+Z是遠離相機）。垂直向上是-Y。
    """
    ear_len = 120.0
    dy = -ear_len * np.cos(np.radians(theta_ca_deg))
    dz = -ear_len * np.sin(np.radians(theta_ca_deg))

    # 面向相機：解剖右肩在影像左半邊(X為負)、左肩在右半邊(X為正)
    half = 180.0
    tilt = np.radians(shoulder_tilt_deg)
    right_sho = np.array([-half * np.cos(tilt), -half * np.sin(tilt), 600.0])
    left_sho = np.array([half * np.cos(tilt), half * np.sin(tilt), 600.0])
    return {
        "right_shoulder": right_sho,
        "left_shoulder": left_sho,
        "right_ear": right_sho + np.array([40.0, dy, dz]),
        "left_ear": left_sho + np.array([-40.0, dy, dz]),
        "neck": (right_sho + left_sho) / 2 + np.array([0.0, -60.0, 0.0]),
    }


def _project(pose: dict) -> tuple[PersonKeypoints, PersonKeypoints]:
    left = np.full((NUM_KEYPOINTS, 2), np.nan, np.float32)
    right = np.full((NUM_KEYPOINTS, 2), np.nan, np.float32)
    conf = np.zeros(NUM_KEYPOINTS, np.float32)
    for name, p in pose.items():
        i = COCO18_KEYPOINT_NAMES.index(name)
        left[i] = project_point(K, np.eye(3), T_LEFT, p)
        right[i] = project_point(K, np.eye(3), T_RIGHT, p)
        conf[i] = 0.9
    return PersonKeypoints(left, conf), PersonKeypoints(right, conf.copy())


def _calib():
    return make_synthetic_stereo_calibration(K, BASELINE_MM)


def test_recovers_the_known_forward_head_angle():
    left, right = _project(_seated_pose(theta_ca_deg=15.0))
    m = measure_posture(_calib(), left, right)
    assert m.theta_ca_deg == pytest.approx(15.0, abs=0.5)


def test_recovers_the_known_shoulder_tilt():
    left, right = _project(_seated_pose(shoulder_tilt_deg=6.0))
    m = measure_posture(_calib(), left, right)
    assert abs(m.theta_sym_deg) == pytest.approx(6.0, abs=0.5)


def test_depth_range_matches_the_scene():
    left, right = _project(_seated_pose())
    m = measure_posture(_calib(), left, right)
    low, high = m.depth_range_mm
    assert 550.0 < low < 650.0 and 600.0 < high < 700.0


def test_vertical_disparity_is_near_zero_for_correct_correspondences():
    """校正後對極線是水平的，配對正確時左右y幾乎相同。"""
    left, right = _project(_seated_pose())
    m = measure_posture(_calib(), left, right)
    assert m.max_abs_vertical_disparity_px < 0.01
    assert plausibility_warnings(m) == []


def test_flags_mismatched_correspondences():
    """左右配對到不同部位時垂直視差會變大。

    三角測量不會因此報錯——它只會取兩條不相交視線的最近點，
    照樣給出一個數量級正常的3D座標，所以這是唯一的線索。
    """
    left, right = _project(_seated_pose())
    shifted = right.points.copy()
    shifted[COCO18_KEYPOINT_NAMES.index("right_ear"), 1] += 25.0
    m = measure_posture(_calib(), left, PersonKeypoints(shifted, right.confidences))

    assert m.max_abs_vertical_disparity_px > 20.0
    assert any("垂直視差" in w for w in plausibility_warnings(m))


def test_flags_implausible_depth():
    """尺度錯了的話深度會整體偏掉，但每個數字看起來都很正常。"""
    pose = {k: v * 8.0 for k, v in _seated_pose().items()}
    left, right = _project(pose)
    m = measure_posture(_calib(), left, right)
    warnings = plausibility_warnings(m)
    assert any("超出桌前坐姿的合理區間" in w for w in warnings)
    assert any("right_shoulder" in w for w in warnings), "要指名是哪個點，否則無從查起"


def test_missing_keypoints_do_not_abort_the_other_angle():
    """少了耳朵時θ_CA算不出來，θ_sym仍然要有結果。"""
    pose = _seated_pose()
    pose.pop("right_ear")
    pose.pop("left_ear")
    left, right = _project(pose)
    m = measure_posture(_calib(), left, right)

    assert m.theta_ca_deg is None
    assert m.theta_sym_deg is not None
    assert any("theta_ca" in e for e in m.angle_errors)


def test_confidence_filter_excludes_low_scoring_points():
    left, right = _project(_seated_pose())
    conf = left.confidences.copy()
    conf[COCO18_KEYPOINT_NAMES.index("right_ear")] = 0.1
    m = measure_posture(_calib(), PersonKeypoints(left.points, conf), right,
                        min_confidence=0.5)
    assert np.isnan(m.keypoints_3d.points[COCO18_KEYPOINT_NAMES.index("right_ear")]).all()


def test_report_is_readable_and_names_the_angles():
    left, right = _project(_seated_pose())
    m = measure_posture(_calib(), left, right)
    report = format_measurement(m, left, right)
    assert "θ_CA" in report and "θ_sym" in report
    assert "right_shoulder" in report
    assert "深度範圍" in report


def test_report_survives_a_frame_with_nothing_detected():
    empty = PersonKeypoints(
        np.full((NUM_KEYPOINTS, 2), np.nan, np.float32), np.zeros(NUM_KEYPOINTS, np.float32)
    )
    m = measure_posture(_calib(), empty, empty)
    assert m.shared_count == 0
    assert m.depth_range_mm is None
    report = format_measurement(m, empty, empty)
    assert "算不出來" in report


def test_angle_precision_grows_with_the_square_of_distance():
    """深度誤差 σ_Z = Z²·σ_d/(fx·B)，所以距離加倍、誤差變四倍。

    這決定了整個系統在多遠還能用，也是使用者唯一能立刻改變的因素。
    """
    from geometry.pipeline import estimate_theta_ca_precision_deg

    calib = _calib()
    near = estimate_theta_ca_precision_deg(calib, 500.0)
    far = estimate_theta_ca_precision_deg(calib, 1000.0)
    assert far == pytest.approx(4.0 * near, rel=1e-6)


def test_warns_when_single_frame_error_exceeds_the_decision_threshold():
    """2026-09-21 實機那次坐到 1900mm，單幀誤差約 50°，遠超過 10° 的判定門檻。"""
    from geometry.pipeline import estimate_theta_ca_precision_deg

    pose = {k: np.array([v[0], v[1], v[2] * 3.1]) for k, v in _seated_pose().items()}
    left, right = _project(pose)
    m = measure_posture(_calib(), left, right)

    assert m.theta_ca_precision_deg > 10.0
    assert any("單幀誤差" in w for w in plausibility_warnings(m))


def test_precision_is_reported_even_when_it_is_acceptable():
    """數字要一直在，使用者才知道現在的角度可以信到什麼程度。"""
    left, right = _project(_seated_pose())
    m = measure_posture(_calib(), left, right)
    assert m.theta_ca_precision_deg is not None
    assert "單幀誤差" in format_measurement(m, left, right)


def test_precision_ignores_a_stray_keypoint_far_from_the_subject():
    """精度換算要以耳朵與肩膀的深度為準，不能被離群點帶走。

    原本取的是所有關鍵點深度的最大最小值中點。只要有一個關節配對錯誤、
    三角測量跑到三公尺外，那個中點就會遠離受試者實際位置，
    印出來的誤差跟著虛報，使用者會以為坐近一點沒有用。
    """
    pose = _seated_pose()
    clean_left, clean_right = _project(pose)
    clean = measure_posture(_calib(), clean_left, clean_right)

    pose_with_outlier = dict(pose)
    pose_with_outlier["right_wrist"] = np.array([0.0, 0.0, 2800.0])
    left, right = _project(pose_with_outlier)
    polluted = measure_posture(_calib(), left, right)

    assert polluted.reference_depth_mm == pytest.approx(clean.reference_depth_mm, rel=1e-9)
    assert polluted.theta_ca_precision_deg == pytest.approx(
        clean.theta_ca_precision_deg, rel=1e-9
    )


def test_reference_depth_falls_back_to_the_median_when_ears_and_shoulders_are_missing():
    from geometry.pipeline import reference_depth_mm

    pose = {"neck": np.array([0.0, 0.0, 700.0]), "nose": np.array([0.0, -50.0, 680.0])}
    left, right = _project(pose)
    m = measure_posture(_calib(), left, right)
    assert reference_depth_mm(m.keypoints_3d) == pytest.approx(690.0, abs=1.0)


def test_report_columns_line_up_when_labels_are_chinese():
    """中文標題佔兩欄，用 f-string 的 :<16 會把表頭排短，欄位對不齊。"""
    from geometry.terminal import display_width as _display_width

    left, right = _project(_seated_pose())
    report = format_measurement(measure_posture(_calib(), left, right), left, right)
    lines = report.splitlines()

    header = next(ln for ln in lines if ln.startswith("關鍵點"))
    row = next(ln for ln in lines if ln.startswith("right_ear"))
    assert _display_width(header) == _display_width(row)


def test_a_bad_limb_does_not_read_as_a_calibration_problem():
    """2026-09-22 實機：四個角度點的 Δy 都 <3px，整體最大卻是 38px。

    自底向上的關聯常把手腕這類點在左右影像連到不同位置，而角度根本沒用到它們。
    兩者混在一起報的話，標定明明已經夠準，看到的還是一行「垂直視差 38px」，
    會讓人白白再重拍一次標定板。
    """
    left, right = _project(_seated_pose())
    broken = right.points.copy()
    broken[COCO18_KEYPOINT_NAMES.index("right_wrist")] = [200.0, 40.0]
    left_pts = left.points.copy()
    left_pts[COCO18_KEYPOINT_NAMES.index("right_wrist")] = [260.0, 300.0]
    conf = np.full(NUM_KEYPOINTS, 0.9, np.float32)
    m = measure_posture(
        _calib(), PersonKeypoints(left_pts, conf), PersonKeypoints(broken, conf)
    )

    warnings = plausibility_warnings(m)
    assert m.max_abs_vertical_disparity_px > 3.0
    disparity_warnings = [w for w in warnings if "垂直視差" in w]
    assert len(disparity_warnings) == 1
    assert "right_wrist" in disparity_warnings[0]
    assert "不影響這一次的角度" in disparity_warnings[0]


def test_a_bad_angle_keypoint_is_reported_as_untrustworthy():
    """同樣的偏移落在角度用到的點上，講法要完全不同。"""
    left, right = _project(_seated_pose())
    shifted = right.points.copy()
    shifted[COCO18_KEYPOINT_NAMES.index("right_shoulder"), 1] += 25.0
    m = measure_posture(_calib(), left, PersonKeypoints(shifted, right.confidences))

    warning = next(w for w in plausibility_warnings(m) if "垂直視差" in w)
    assert "right_shoulder" in warning
    assert "角度不可信" in warning


def test_negative_depth_is_named_as_a_pairing_error_not_a_scale_error():
    """深度為負代表點在相機後方，跟 --square-size-mm 填錯是兩回事。"""
    left, right = _project(_seated_pose())
    swapped = right.points.copy()
    i = COCO18_KEYPOINT_NAMES.index("right_wrist")
    swapped[i] = [900.0, 300.0]
    left_pts = left.points.copy()
    left_pts[i] = [100.0, 300.0]
    conf = np.full(NUM_KEYPOINTS, 0.9, np.float32)
    m = measure_posture(
        _calib(), PersonKeypoints(left_pts, conf), PersonKeypoints(swapped, conf)
    )

    assert m.keypoints_3d.points[i, 2] < 0
    warning = next(w for w in plausibility_warnings(m) if "合理區間" in w)
    assert "相機後方" in warning
    assert "square-size-mm" not in warning


def test_side_mounting_is_an_order_of_magnitude_better_for_theta_ca():
    """θ_CA 量的是往前伸多少，而「往前」落在哪個軸由相機方位決定。

    正面時它完全落在深度軸（σ_Z 隨距離平方成長），側面時落在影像平面
    （σ_X 只隨距離線性成長），兩者差 Z/B 倍。620mm 配 60mm 基線就是 10 倍。
    這個差距比坐近一點能得到的改善大得多，卻一度完全沒有反映在估計值裡。
    """
    from geometry.pipeline import estimate_theta_ca_precision_deg

    calib = _calib()
    frontal = estimate_theta_ca_precision_deg(calib, 620.0, azimuth_deg=0.0)
    side = estimate_theta_ca_precision_deg(calib, 620.0, azimuth_deg=90.0)
    assert frontal / side > 8.0
    # 中間的方位角要落在兩者之間，而且靠近正面那側改善有限
    oblique = estimate_theta_ca_precision_deg(calib, 620.0, azimuth_deg=30.0)
    assert side < oblique < frontal
    assert oblique > 0.8 * frontal, "30 度只換到很小的改善，不該看起來像解決了問題"


def test_disparity_noise_is_larger_than_keypoint_noise():
    """視差是左右兩次像素量測的差，所以 σ_d = √2·σ_px。

    早期版本把關鍵點雜訊直接當成視差雜訊，整個精度估計低估了 √2 倍——
    600mm 正面算出 ±5° 而實際接近 ±7°，對 10° 的門檻來說是有意義的差別。
    蒙地卡羅（3000 次、1px 雜訊、620mm、正面）量到 ±10.0°，解析式給 ±10.8°。
    """
    from geometry.pipeline import estimate_theta_ca_precision_deg

    calib = _calib()
    fx, baseline = float(calib.P1[0, 0]), calib.baseline_mm
    depth = 620.0
    naive = np.degrees(np.sqrt(2) * depth**2 * 1.0 / (fx * baseline) / 170.0)
    assert estimate_theta_ca_precision_deg(calib, depth) == pytest.approx(
        np.sqrt(2) * naive, rel=1e-9
    )


def test_azimuth_is_measured_from_the_shoulders():
    from geometry.pipeline import camera_azimuth_deg

    assert camera_azimuth_deg(measure_posture(_calib(), *_project(_seated_pose())).keypoints_3d) \
        == pytest.approx(0.0, abs=0.5)


def test_a_bad_left_ear_does_not_discredit_the_angles():
    """left_ear 在預設參數下沒有任何角度用到——θ_CA 取右側、θ_sym 取雙肩。

    把它放進「角度用到的點」會讓它配對錯誤時誤報成角度不可信，
    而實際上兩個角度都不受影響。
    """
    left, right = _project(_seated_pose())
    broken = right.points.copy()
    broken[COCO18_KEYPOINT_NAMES.index("left_ear"), 1] += 40.0
    m = measure_posture(_calib(), left, PersonKeypoints(broken, right.confidences))

    warning = next(w for w in plausibility_warnings(m) if "垂直視差" in w)
    assert "left_ear" in warning
    assert "角度不可信" not in warning
    assert m.theta_ca_deg is not None and m.theta_sym_deg is not None


def test_depth_warning_survives_a_rename_of_the_display_format():
    """判斷深度為負要看數值，不能去剖析自己格式化出來的字串。"""
    from geometry.pipeline import _implausible_depths

    left, right = _project(_seated_pose())
    m = measure_posture(_calib(), left, right)
    assert _implausible_depths(m) == []


def test_worst_disparity_with_an_empty_selection_returns_nothing():
    """空的名單代表沒有要看的點，不該退回全部 18 點。"""
    from geometry.pipeline import _worst_disparity

    left, right = _project(_seated_pose())
    m = measure_posture(_calib(), left, right)
    assert _worst_disparity(m, ()) is None
    assert _worst_disparity(m) is not None


def _at_depth(scale: float):
    pose = {k: np.array([v[0], v[1], v[2] * scale]) for k, v in _seated_pose().items()}
    return _project(pose)


def test_rejects_the_frame_that_reports_the_best_error_for_the_worst_data():
    """2026-09-23 實機錄到距離 74mm 配誤差 ±0.1°。

    誤差公式只看距離，距離愈近算出來愈小，所以偵測失誤跑到近處時，
    印出來的誤差數字反而最好看。這種幀一定要在進入平均前擋掉——
    靠數值大小是擋不住的，因為它的數值全部落在正常範圍。
    """
    from geometry.pipeline import unusable_reason

    left, right = _at_depth(74.0 / 600.0)
    m = measure_posture(_calib(), left, right)

    assert m.theta_ca_precision_deg < 1.0, "誤差確實看起來很漂亮"
    reason = unusable_reason(m)
    assert reason is not None and "深度" in reason


def test_rejects_points_triangulated_behind_the_camera():
    """深度 −291mm：實機錄到的另一種失誤，θ_CA 跟著印出 −169°。"""
    from geometry.pipeline import unusable_reason

    left, right = _project(_seated_pose())
    swapped = right.points.copy()
    for name in ("right_ear", "right_shoulder", "left_shoulder", "left_ear"):
        swapped[COCO18_KEYPOINT_NAMES.index(name), 0] += 400.0
    m = measure_posture(_calib(), left, PersonKeypoints(swapped, right.confidences))

    assert unusable_reason(m) is not None


def test_accepts_an_ordinary_seated_frame():
    from geometry.pipeline import unusable_reason

    left, right = _project(_seated_pose())
    assert unusable_reason(measure_posture(_calib(), left, right)) is None


def test_rejection_reason_names_what_was_wrong():
    """略過的理由要講得出來，否則使用者只會看到幀數一直被吃掉。"""
    from geometry.pipeline import unusable_reason

    left, right = _project(_seated_pose())
    broken = right.points.copy()
    broken[COCO18_KEYPOINT_NAMES.index("right_shoulder"), 1] += 30.0
    reason = unusable_reason(measure_posture(_calib(), left, PersonKeypoints(broken, right.confidences)))
    assert reason is not None and "right_shoulder" in reason
