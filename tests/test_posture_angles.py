import numpy as np
import pytest

from geometry.keypoints3d import PersonKeypoints3D
from geometry.posture_angles import theta_ca, theta_ka, theta_sym
from pose.topology import NUM_KEYPOINTS, keypoint_index


def _make_keypoints3d(overrides: dict[str, np.ndarray]) -> PersonKeypoints3D:
    points = np.full((NUM_KEYPOINTS, 3), np.nan, dtype=np.float64)
    for name, value in overrides.items():
        points[keypoint_index(name)] = value
    return PersonKeypoints3D(points=points)


def test_theta_ca_zero_when_ear_directly_above_shoulder():
    shoulder = np.array([0.0, 0.0, 0.0])
    ear = shoulder + np.array([0.0, -100.0, 0.0])  # 正上方(Y向下為正，正上方是-Y)，無深度偏移
    kp = _make_keypoints3d({"right_shoulder": shoulder, "right_ear": ear})
    assert theta_ca(kp) == pytest.approx(0.0, abs=1e-6)


def test_theta_ca_known_tilt_angle():
    shoulder = np.array([0.0, 0.0, 0.0])
    angle_deg = 20.0
    rad = np.radians(angle_deg)
    ear = shoulder + 100.0 * np.array([0.0, -np.cos(rad), -np.sin(rad)])
    kp = _make_keypoints3d({"right_shoulder": shoulder, "right_ear": ear})
    assert theta_ca(kp) == pytest.approx(angle_deg, abs=1e-6)


def test_theta_ca_uses_left_side_when_requested():
    shoulder = np.array([0.0, 0.0, 0.0])
    ear = shoulder + np.array([0.0, -100.0, 0.0])
    kp = _make_keypoints3d({"left_shoulder": shoulder, "left_ear": ear})
    assert theta_ca(kp, side="left") == pytest.approx(0.0, abs=1e-6)


def test_theta_ca_raises_when_ear_missing():
    kp = _make_keypoints3d({"right_shoulder": np.array([0.0, 0.0, 0.0])})
    with pytest.raises(ValueError):
        theta_ca(kp)


def test_theta_sym_zero_when_shoulders_level():
    left = np.array([-50.0, 0.0, 0.0])
    right = np.array([50.0, 0.0, 0.0])
    kp = _make_keypoints3d({"left_shoulder": left, "right_shoulder": right})
    assert theta_sym(kp) == pytest.approx(0.0, abs=1e-6)


def test_theta_sym_known_tilt_angle():
    left = np.array([-50.0, 0.0, 0.0])
    angle_deg = 10.0
    rad = np.radians(angle_deg)
    # 右肩比左肩高(Y較小)，偏移量對應angle_deg
    right = left + 100.0 * np.array([np.cos(rad), -np.sin(rad), 0.0])
    kp = _make_keypoints3d({"left_shoulder": left, "right_shoulder": right})
    assert theta_sym(kp) == pytest.approx(-angle_deg, abs=1e-6)


def test_theta_sym_raises_when_shoulder_missing():
    kp = _make_keypoints3d({"left_shoulder": np.array([0.0, 0.0, 0.0])})
    with pytest.raises(ValueError):
        theta_sym(kp)


def test_theta_ca_raises_when_keypoints_coincide():
    """耳朵與肩膀被計算到同一個3D點時要拋出例外。

    這種情況原本會回傳0度，也就是完美姿勢，異常的資料反而永遠不會觸發警示。
    """
    same_point = np.array([0.0, 0.0, 500.0])
    kp = _make_keypoints3d({"right_shoulder": same_point, "right_ear": same_point.copy()})
    with pytest.raises(ValueError):
        theta_ca(kp)


def test_theta_ka_not_implemented():
    kp = _make_keypoints3d({})
    with pytest.raises(NotImplementedError):
        theta_ka(kp)
