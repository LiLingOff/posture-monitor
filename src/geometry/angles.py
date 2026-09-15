"""通用角度數學工具，跟研究專用的角度名稱無關。"""
from __future__ import annotations

import numpy as np


def angle_between_vectors(a: np.ndarray, b: np.ndarray) -> float:
    """兩個3D向量夾角（度，0~180，無號）。"""
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def project_onto_plane(vector: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """向量投影到以plane_normal為法向量的平面上。"""
    normal_unit = plane_normal / np.linalg.norm(plane_normal)
    return vector - np.dot(vector, normal_unit) * normal_unit


def signed_angle_in_plane(
    vector: np.ndarray, reference_axis: np.ndarray, plane_normal: np.ndarray
) -> float:
    """向量投影到指定平面後，相對參考軸的帶號夾角（度，-180~180）。

    用atan2(垂直分量, 平行分量)算，而不是acos(內積)，才能區分「往哪個方向偏」。
    """
    v = project_onto_plane(vector, plane_normal)
    ref = project_onto_plane(reference_axis, plane_normal)
    perp = np.cross(plane_normal, ref)
    return float(np.degrees(np.arctan2(np.dot(v, perp), np.dot(v, ref))))
