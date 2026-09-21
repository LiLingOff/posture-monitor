"""坐姿角度計算。θ_CA、θ_sym依前作公式的3D版本實作；θ_KA公式未定，尚未實作。

假設雙目校正後(rectified)座標系採OpenCV慣例：X右、Y下、Z深度(遠離相機方向)，
相機架在受試者正面、大致水平、無明顯翻滾角——這跟原始實驗步驟文件「記錄殘餘垂直角度差
作為後續分析共變量」的做法一致（不做座標系旋轉校正，殘差留給統計分析處理）。

個人校正基準（θ_offset）、判定門檻（10°/5°/20px）屬於執行期監測邏輯，不在這裡處理；
這裡只計算目前這一幀的原始角度。
"""
from __future__ import annotations

import numpy as np

from .angles import signed_angle_in_plane
from .keypoints3d import PersonKeypoints3D

_VERTICAL_AXIS = np.array([0.0, -1.0, 0.0])  # Y向下為正，所以正上方是-Y
_LATERAL_AXIS = np.array([1.0, 0.0, 0.0])  # 左右方向，矢狀面的法向量
_DEPTH_AXIS = np.array([0.0, 0.0, 1.0])  # 前後(深度)方向，冠狀面的法向量

_MISSING_DEFINITION_MSG = "尚未取得θ_KA的精確定義（哪些關節點、參考平面/軸、正負號慣例），無法實作"


def theta_ca(keypoints_3d: PersonKeypoints3D, side: str = "right") -> float:
    """頸椎前傾角：肩膀→耳朵向量(ear - shoulder)投影到矢狀面後相對垂直軸的帶號夾角。

    正負號：耳朵在肩膀正上方是0°，**頭往前伸（朝相機方向）為正**，往後仰為負。
    前作以 >10° 判定頭部前傾姿勢，對應的就是正值這一側。
    這個方向性是用atan2而非arccos的唯一理由，判定邏輯要靠它區分前傾與後仰。

    前作用單一45°相機、靠sin(45°)補償透視壓縮才能計算這個角度；
    改用3D三角測量後直接以真實深度計算矢狀面投影，不再需要這個補償係數。
    side預設right，對應前作側面相機架在使用者右側的設定。
    """
    ear = keypoints_3d.get(f"{side}_ear")
    shoulder = keypoints_3d.get(f"{side}_shoulder")
    if ear is None or shoulder is None:
        raise ValueError(f"{side}_ear或{side}_shoulder未偵測到，無法計算theta_ca")
    return signed_angle_in_plane(ear - shoulder, _VERTICAL_AXIS, plane_normal=_LATERAL_AXIS)


def theta_sym(keypoints_3d: PersonKeypoints3D) -> float:
    """肩膀水平角：左右肩連線投影到冠狀面後相對水平軸的帶號夾角。

    正負號：雙肩等高是0°，**左肩較高為正、右肩較高為負**
    （Y軸向下，所以較高代表Y較小）。前作以絕對值 >5° 判定聳肩或脊椎側彎傾向，
    但左右哪一邊高在臨床上是不同的事，所以這裡保留方向。
    """
    left = keypoints_3d.get("left_shoulder")
    right = keypoints_3d.get("right_shoulder")
    if left is None or right is None:
        raise ValueError("left_shoulder或right_shoulder未偵測到，無法計算theta_sym")
    return signed_angle_in_plane(right - left, _LATERAL_AXIS, plane_normal=_DEPTH_AXIS)


def theta_ka(keypoints_3d: PersonKeypoints3D) -> float:
    raise NotImplementedError(_MISSING_DEFINITION_MSG)
