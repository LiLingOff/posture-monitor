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
    # 受試者面向相機：解剖學上的右肩出現在影像左半邊(X較小)，左肩在右半邊(X較大)。
    # 擺反的話 left-right 會指向 -X，算出來永遠接近 ±180° 而不是 0°。
    right = np.array([-50.0, 0.0, 0.0])
    left = np.array([50.0, 0.0, 0.0])
    kp = _make_keypoints3d({"left_shoulder": left, "right_shoulder": right})
    assert theta_sym(kp) == pytest.approx(0.0, abs=1e-6)


def test_theta_sym_known_tilt_angle():
    right = np.array([-50.0, 0.0, 0.0])
    angle_deg = 10.0
    rad = np.radians(angle_deg)
    # 從右肩往左肩，左肩比右肩低(Y較大) -> 右肩較高 -> 正值
    left = right + 100.0 * np.array([np.cos(rad), np.sin(rad), 0.0])
    kp = _make_keypoints3d({"left_shoulder": left, "right_shoulder": right})
    assert theta_sym(kp) == pytest.approx(angle_deg, abs=1e-6)


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


def test_theta_ca_sign_forward_is_positive():
    """頭往前伸（朝相機，-Z）為正，往後仰為負。

    這個方向性是用atan2而非arccos的唯一理由：前作以 >10° 判定頭部前傾，
    正負號搞反的話判定會整個顛倒——後仰被當成前傾，而數值大小完全一樣，
    看不出任何異常。
    """
    shoulder = np.array([150.0, 0.0, 600.0])
    lean = 120.0 * np.sin(np.radians(15.0))
    drop = 120.0 * np.cos(np.radians(15.0))

    forward = _make_keypoints3d({
        "right_shoulder": shoulder,
        "right_ear": shoulder + np.array([0.0, -drop, -lean]),
    })
    backward = _make_keypoints3d({
        "right_shoulder": shoulder,
        "right_ear": shoulder + np.array([0.0, -drop, +lean]),
    })

    assert theta_ca(forward) == pytest.approx(15.0, abs=1e-3)
    assert theta_ca(backward) == pytest.approx(-15.0, abs=1e-3)


def test_theta_sym_sign_right_shoulder_higher_is_positive():
    """右肩較高為正、左肩較高為負（Y軸向下，較高代表Y較小）。

    這個方向與前作的 arctan[(y_L-y_R)/(x_L-x_R)] 一致。
    擺位前提是受試者面向相機：解剖右肩在影像左半邊(X較小)、左肩在右半邊(X較大)。
    這個前提一度在程式與測試裡同時弄反，導致兩邊互相印證、實機資料才抓出來。
    """
    # 兩肩各偏移drop，但夾角看的是兩肩連線這個向量，
    # 兩倍的分子與分母會抵銷，所以傾角仍是6度而不是12度。
    drop = 180.0 * np.sin(np.radians(6.0))
    span = 180.0 * np.cos(np.radians(6.0))

    right_higher = _make_keypoints3d({
        "right_shoulder": np.array([-span, -drop, 600.0]),
        "left_shoulder": np.array([span, +drop, 600.0]),
    })
    left_higher = _make_keypoints3d({
        "right_shoulder": np.array([-span, +drop, 600.0]),
        "left_shoulder": np.array([span, -drop, 600.0]),
    })

    assert theta_sym(right_higher) == pytest.approx(6.0, abs=1e-3)
    assert theta_sym(left_higher) == pytest.approx(-6.0, abs=1e-3)


def test_theta_sym_is_near_zero_not_180_for_a_person_facing_the_camera():
    """用實機量到的肩膀座標當回歸案例。

    2026-09-21 在 Jetson 上量到 right_shoulder x=219、left_shoulder x=418，
    當時程式取 right-left，算出 +157°。符號方向錯的話這個測試會再紅一次。
    """
    kp = _make_keypoints3d({
        "right_shoulder": np.array([219.2, 316.2, 588.9]),
        "left_shoulder": np.array([417.8, 233.3, 434.8]),
    })
    value = theta_sym(kp)
    assert abs(value) < 45.0, f"算出 {value:.1f}°，接近±180°代表左右肩向量取反了"
