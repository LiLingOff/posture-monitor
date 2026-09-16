"""通用角度數學工具，跟研究專用的角度名稱無關。"""
from __future__ import annotations

import numpy as np

# 長度小於這個值就當成退化向量。單位是mm，0.001mm遠小於任何真實的關節間距，
# 會落在這個範圍代表兩個關鍵點被算到同一個位置，是偵測或三角測量出問題。
_MIN_VECTOR_NORM = 1e-3


def _check_not_degenerate(vector: np.ndarray, name: str) -> float:
    norm = float(np.linalg.norm(vector))
    if norm < _MIN_VECTOR_NORM:
        raise ValueError(f"{name}長度趨近0（{norm:.3e}），兩個端點重疊，無法定義角度")
    return norm


def angle_between_vectors(a: np.ndarray, b: np.ndarray) -> float:
    """兩個3D向量夾角（度，0~180，無號）。"""
    norm_a = _check_not_degenerate(a, "向量a")
    norm_b = _check_not_degenerate(b, "向量b")
    cos_theta = np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0)
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
    _check_not_degenerate(vector, "輸入向量")
    v = project_onto_plane(vector, plane_normal)
    # 向量幾乎垂直於該平面時，投影後趨近0，這時的角度沒有意義；
    # 靜靜回傳0度會被誤讀成「完全沒有偏移」，也就是最理想的姿勢
    _check_not_degenerate(v, "向量投影到平面後")
    ref = project_onto_plane(reference_axis, plane_normal)
    _check_not_degenerate(ref, "參考軸投影到平面後")
    perp = np.cross(plane_normal, ref)
    return float(np.degrees(np.arctan2(np.dot(v, perp), np.dot(v, ref))))
