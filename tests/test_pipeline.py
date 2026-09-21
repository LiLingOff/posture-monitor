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
    照樣給出一個看似合理的3D座標，所以這是唯一的線索。
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
    assert any("深度範圍" in w for w in plausibility_warnings(m))


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
