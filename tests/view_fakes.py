"""疊圖測試用的假資料。不需要相機、模型或顯示器。"""
from __future__ import annotations

import numpy as np

from geometry.judgement import Posture
from pose.keypoints import PersonKeypoints
from pose.topology import NUM_KEYPOINTS, keypoint_index
from view.live_view import LiveView

# 一個坐著的人，座標對應 640x480 的畫面。
SEATED = {
    "nose": (320, 150), "neck": (320, 200),
    "right_shoulder": (260, 205), "right_elbow": (240, 280), "right_wrist": (250, 350),
    "left_shoulder": (380, 205), "left_elbow": (400, 280), "left_wrist": (390, 350),
    "right_hip": (275, 340), "left_hip": (365, 340),
    "right_knee": (270, 420), "left_knee": (365, 420),
    "right_ankle": (268, 470), "left_ankle": (366, 470),
    "right_eye": (305, 140), "left_eye": (335, 140),
    "right_ear": (290, 152), "left_ear": (350, 152),
}


def person(only: tuple[str, ...] | None = None, **moves) -> PersonKeypoints:
    """一個人的關鍵點。only 限定只有這幾點被偵測到，其餘留 NaN。"""
    points = np.full((NUM_KEYPOINTS, 2), np.nan)
    for name, (x, y) in SEATED.items():
        if only is not None and name not in only:
            continue
        dx, dy = moves.get(name, (0.0, 0.0))
        points[keypoint_index(name)] = (x + dx, y + dy)
    confidences = np.where(np.isnan(points[:, 0]), 0.0, 1.0)
    return PersonKeypoints(points=points, confidences=confidences)


def blank(fill: int = 40, width: int = 640, height: int = 480) -> np.ndarray:
    """一張純色畫布。fill 當指紋用，可以認出這一半是從哪張圖來的。"""
    return np.full((height, width, 3), fill, dtype=np.uint8)


def view(**overrides) -> LiveView:
    values = dict(
        left_bgr=blank(40), right_bgr=blank(90),
        left=person(), right=person(),
        state=Posture.OK, window_count=30, window_size=30,
    )
    values.update(overrides)
    return LiveView(**values)


def at(canvas: np.ndarray, x: float, y: float) -> tuple[int, int, int]:
    return tuple(int(v) for v in canvas[int(round(y)), int(round(x))])


def painted(canvas: np.ndarray, background: int) -> bool:
    """這張畫布上有沒有畫過東西。"""
    return bool((canvas != background).any())
