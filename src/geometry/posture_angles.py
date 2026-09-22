"""坐姿角度計算。θ_CA、θ_sym依前作公式的3D版本實作；θ_KA公式未定，尚未實作。

假設雙目校正後(rectified)座標系採OpenCV慣例：X右、Y下、Z深度(遠離相機方向)。
唯一的架設要求是**相機大致水平、沒有明顯翻滾角**，因為程式直接把相機的 Y 軸
當作重力方向。這跟原始實驗步驟文件「記錄殘餘垂直角度差作為後續分析共變量」的
做法一致（不做座標系旋轉校正，殘差留給統計分析處理）。

**方位角不受限制。** 矢狀面與冠狀面是從受試者自己的雙肩連線算出來的，
不是拿相機的 X 軸當左右方向，所以雙目模組架在正面、側面或任何斜角都一樣。
這是這個專案取代前作 45° 假設的完整形式——只補償深度壓縮而仍用相機軸定義解剖平面的話，
方位角會以另一個係數的形式重新混進角度裡（詳見 theta_sym 的說明）。

個人校正基準（θ_offset）、判定門檻（10°/5°/20px）屬於執行期監測邏輯，不在這裡處理；
這裡只計算目前這一幀的原始角度。
"""
from __future__ import annotations

import numpy as np

from .angles import signed_angle_in_plane
from .keypoints3d import PersonKeypoints3D

_VERTICAL_AXIS = np.array([0.0, -1.0, 0.0])  # Y向下為正，所以正上方是-Y
# 雙肩取不到時退回相機軸。這等同於假設受試者正對相機，只在無從判斷時才使用。
_CAMERA_LATERAL_AXIS = np.array([1.0, 0.0, 0.0])
_CAMERA_DEPTH_AXIS = np.array([0.0, 0.0, 1.0])
# 雙肩連線的水平投影短於這個長度就無法定義方向。正常肩寬約350mm，
# 只有兩點被算到同一位置、或身體幾乎躺平時才會落到這個範圍。
_MIN_SHOULDER_SPAN_MM = 20.0

_MISSING_DEFINITION_MSG = "尚未取得θ_KA的精確定義（哪些關節點、參考平面/軸、正負號慣例），無法實作"


def anatomical_axes(keypoints_3d: PersonKeypoints3D) -> tuple[np.ndarray, np.ndarray]:
    """從雙肩算出受試者自己的左右軸與前後軸，回傳 (lateral, backward)。

    lateral 是雙肩連線的水平分量（由右肩指向左肩），backward 與它和垂直軸都垂直、
    指向受試者背後。前者是矢狀面的法向量，後者是冠狀面的法向量。

    為什麼要這樣算，而不是直接用相機的 X 軸與 Z 軸：相機架在斜角時，
    雙肩連線在相機 X 軸上的分量會被 cos(方位角) 壓縮，而垂直分量不會，
    算出來的 θ_sym 就被放大 1/cos(方位角) 倍——45° 斜角會多報 41%。
    這跟前作除以 sin(45°) 是同一類的係數，只是換了個位置。
    三角測量拿到真實3D座標之後，這個係數可以真正消掉，不必只是換一個估計值。

    垂直分量會先去掉：受試者聳肩或側傾時矢狀面不應該跟著傾斜。
    """
    left = keypoints_3d.get("left_shoulder")
    right = keypoints_3d.get("right_shoulder")
    if left is None or right is None:
        return _CAMERA_LATERAL_AXIS, _CAMERA_DEPTH_AXIS

    lateral = left - right
    lateral = lateral - np.dot(lateral, _VERTICAL_AXIS) * _VERTICAL_AXIS
    span = float(np.linalg.norm(lateral))
    if span < _MIN_SHOULDER_SPAN_MM:
        return _CAMERA_LATERAL_AXIS, _CAMERA_DEPTH_AXIS

    lateral = lateral / span
    # 右手定則：受試者正對相機時 lateral 是 +X，這個外積會得到 +Z，
    # 與退回相機軸時的 _CAMERA_DEPTH_AXIS 完全一致，正負號慣例因此不變。
    backward = np.cross(_VERTICAL_AXIS, lateral)
    return lateral, backward


def theta_ca(keypoints_3d: PersonKeypoints3D, side: str = "right") -> float:
    """頸椎前傾角：肩膀→耳朵向量(ear - shoulder)投影到矢狀面後相對垂直軸的帶號夾角。

    正負號：耳朵在肩膀正上方是0°，**頭往前伸（朝相機方向）為正**，往後仰為負。
    前作以 >10° 判定頭部前傾姿勢，對應的就是正值這一側。
    這個方向性是用atan2而非arccos的唯一理由，判定邏輯要靠它區分前傾與後仰。

    矢狀面由受試者自己的雙肩連線定義（見 anatomical_axes），不是相機的 X 軸，
    所以雙目模組架在正面、側面或斜角都得到同一個角度。雙肩有一邊沒偵測到時
    退回相機軸，那等同於假設受試者正對相機。

    前作用單一45°相機、靠sin(45°)補償透視壓縮才能計算這個角度；
    改用3D三角測量後直接以真實深度計算矢狀面投影，不再需要這個補償係數。
    side預設right，對應前作側面相機架在使用者右側的設定。
    """
    ear = keypoints_3d.get(f"{side}_ear")
    shoulder = keypoints_3d.get(f"{side}_shoulder")
    if ear is None or shoulder is None:
        raise ValueError(f"{side}_ear或{side}_shoulder未偵測到，無法計算theta_ca")
    lateral, _ = anatomical_axes(keypoints_3d)
    return signed_angle_in_plane(ear - shoulder, _VERTICAL_AXIS, plane_normal=lateral)


def theta_sym(keypoints_3d: PersonKeypoints3D) -> float:
    """肩膀水平角：左右肩連線投影到冠狀面後相對水平軸的帶號夾角。

    向量取 left - right，不是反過來。受試者面向相機時，解剖學上的右肩會出現在
    影像的左半邊（X較小），左肩在右半邊（X較大），所以 left - right 才會指向 +X、
    雙肩等高時得到0°。取成 right - left 的話會指向 -X，算出來永遠接近 ±180°。
    這個方向與前作的 arctan[(y_L-y_R)/(x_L-x_R)] 一致。

    參考軸用的是雙肩連線的水平分量而非相機 X 軸。相機架在斜角時，
    雙肩連線在相機 X 軸上的分量會被 cos(方位角) 壓縮而垂直分量不會，
    角度就被放大 1/cos(方位角) 倍：30° 斜角多報 15%，45° 多報 41%。

    正負號：雙肩等高是0°，**右肩較高為正、左肩較高為負**
    （Y軸向下，所以較高代表Y較小）。前作以絕對值 >5° 判定聳肩或脊椎側彎傾向，
    但左右哪一邊高在臨床上是不同的事，所以這裡保留方向。
    """
    left = keypoints_3d.get("left_shoulder")
    right = keypoints_3d.get("right_shoulder")
    if left is None or right is None:
        raise ValueError("left_shoulder或right_shoulder未偵測到，無法計算theta_sym")
    lateral, backward = anatomical_axes(keypoints_3d)
    return signed_angle_in_plane(left - right, lateral, plane_normal=backward)


def theta_ka(keypoints_3d: PersonKeypoints3D) -> float:
    raise NotImplementedError(_MISSING_DEFINITION_MSG)
