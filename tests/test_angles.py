import numpy as np
import pytest

from geometry.angles import angle_between_vectors, project_onto_plane, signed_angle_in_plane


def test_angle_between_vectors_known_30_degrees():
    a = np.array([1.0, 0.0, 0.0])
    theta = np.radians(30.0)
    b = np.array([np.cos(theta), np.sin(theta), 0.0])
    assert angle_between_vectors(a, b) == pytest.approx(30.0, abs=1e-6)


def test_angle_between_vectors_orthogonal_is_90():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert angle_between_vectors(a, b) == pytest.approx(90.0, abs=1e-6)


def test_angle_between_vectors_identical_is_zero():
    a = np.array([3.0, -2.0, 1.0])
    assert angle_between_vectors(a, a) == pytest.approx(0.0, abs=1e-6)


def test_project_onto_plane_removes_normal_component():
    vector = np.array([1.0, 2.0, 3.0])
    normal = np.array([0.0, 0.0, 1.0])
    projected = project_onto_plane(vector, normal)
    assert np.allclose(projected, [1.0, 2.0, 0.0])


def test_signed_angle_in_plane_positive_rotation():
    plane_normal = np.array([0.0, 0.0, 1.0])
    reference = np.array([1.0, 0.0, 0.0])
    theta = np.radians(30.0)
    vector = np.array([np.cos(theta), np.sin(theta), 0.0])
    assert signed_angle_in_plane(vector, reference, plane_normal) == pytest.approx(30.0, abs=1e-6)


def test_signed_angle_in_plane_negative_rotation():
    plane_normal = np.array([0.0, 0.0, 1.0])
    reference = np.array([1.0, 0.0, 0.0])
    theta = np.radians(30.0)
    vector = np.array([np.cos(theta), -np.sin(theta), 0.0])
    assert signed_angle_in_plane(vector, reference, plane_normal) == pytest.approx(-30.0, abs=1e-6)


def test_angle_between_vectors_raises_on_zero_length():
    """零長度向量無法定義角度。回傳0度會被解讀成完全沒有偏移，也就是最理想的姿勢，
    但實際情況是兩個關鍵點重疊、資料異常。"""
    with pytest.raises(ValueError):
        angle_between_vectors(np.zeros(3), np.array([1.0, 0.0, 0.0]))


def test_signed_angle_in_plane_raises_on_zero_length():
    with pytest.raises(ValueError):
        signed_angle_in_plane(np.zeros(3), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))


def test_signed_angle_in_plane_raises_when_vector_perpendicular_to_plane():
    """向量完全垂直於該平面時投影後長度為0，角度沒有意義。"""
    plane_normal = np.array([0.0, 0.0, 1.0])
    reference = np.array([1.0, 0.0, 0.0])
    vector = np.array([0.0, 0.0, 100.0])  # 整個落在法線方向上
    with pytest.raises(ValueError):
        signed_angle_in_plane(vector, reference, plane_normal)


def test_signed_angle_in_plane_ignores_out_of_plane_component():
    plane_normal = np.array([0.0, 0.0, 1.0])
    reference = np.array([1.0, 0.0, 0.0])
    theta = np.radians(30.0)
    # 加入一個Z分量，不應影響投影到XY平面後算出來的角度
    vector = np.array([np.cos(theta), np.sin(theta), 999.0])
    assert signed_angle_in_plane(vector, reference, plane_normal) == pytest.approx(30.0, abs=1e-6)


def test_signed_angle_independent_of_plane_normal_length():
    """法向量長度不該影響角度。

    外積沒有先正規化法向量的話，atan2的分子會被|plane_normal|縮放、分母不會，
    同一組幾何傳[1,0,0]得到-45度、傳[2,0,0]卻得到-63.4度。
    """
    vector = np.array([0.0, -1.0, 1.0])
    reference = np.array([0.0, -1.0, 0.0])

    baseline = signed_angle_in_plane(vector, reference, np.array([1.0, 0.0, 0.0]))
    assert baseline == pytest.approx(-45.0)

    for scale in (0.5, 2.0, 37.0):
        scaled = signed_angle_in_plane(vector, reference, np.array([scale, 0.0, 0.0]))
        assert scaled == pytest.approx(baseline), f"法向量乘{scale}倍後角度就變了"


def test_zero_plane_normal_raises():
    """零向量定義不出平面，回傳nan會一路往下游傳。"""
    with pytest.raises(ValueError):
        signed_angle_in_plane(
            np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.zeros(3)
        )
