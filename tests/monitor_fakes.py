"""假的量測與基準。狀態機與判定的測試共用，不需要相機或模型。"""
from __future__ import annotations

import numpy as np

from geometry.baseline import PostureBaseline
from geometry.keypoints3d import PersonKeypoints3D
from geometry.pipeline import PostureMeasurement
from pose.topology import NUM_KEYPOINTS, keypoint_index

# 角度與方位角要用到的那幾點。腳踝手腕不影響這幾項測試。
USED = ("left_shoulder", "right_shoulder", "left_ear", "right_ear", "neck",
        "nose", "left_eye", "right_eye")


def measurement(theta_ca: float = 5.0, theta_sym: float = 1.0, usable: bool = True,
                shoulder_height_mm: float | None = -120.0) -> PostureMeasurement:
    """一幀量測。usable=False 會踩到深度的合理範圍檢查而被略過。"""
    points = np.full((NUM_KEYPOINTS, 3), np.nan)
    disparity = np.full(NUM_KEYPOINTS, np.nan)
    for name in USED:
        points[keypoint_index(name)] = (0.0, 0.0, 650.0)
        disparity[keypoint_index(name)] = 0.5
    return PostureMeasurement(
        keypoints_3d=PersonKeypoints3D(points=points),
        vertical_disparity_px=disparity,
        shared_count=len(USED),
        theta_ca_deg=theta_ca, theta_sym_deg=theta_sym,
        theta_ca_precision_deg=0.5,
        reference_depth_mm=650.0 if usable else -900.0,
        camera_azimuth_deg=20.0,
        shoulder_height_mm=shoulder_height_mm,
        theta_ca_side="right",
        edge_keypoints=[], angle_errors=[],
    )


def baseline(**overrides) -> PostureBaseline:
    values = dict(
        subject="chenyue", captured_at="2026-09-30T19:00:00", frames=120,
        rejected=2, duration_s=20.0, theta_ca_deg=3.0, theta_sym_deg=1.0,
        theta_ca_std_deg=3.0, theta_sym_std_deg=0.5,
        theta_ca_standard_error_deg=0.5, theta_sym_standard_error_deg=0.1,
        distance_mm=650.0, azimuth_deg=20.0,
        shoulder_height_mm=-120.0, shoulder_height_std_mm=4.0,
    )
    values.update(overrides)
    return PostureBaseline(**values)
