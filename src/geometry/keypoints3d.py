from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calibration.stereo_calibration import StereoCalibrationResult
from pose.keypoints import PersonKeypoints
from pose.topology import NUM_KEYPOINTS, keypoint_index

from .triangulation import triangulate_points


@dataclass
class PersonKeypoints3D:
    points: np.ndarray  # shape(NUM_KEYPOINTS, 3)，單位mm，未偵測到或未三角測量填NaN

    def __post_init__(self) -> None:
        if self.points.shape != (NUM_KEYPOINTS, 3):
            raise ValueError(f"points形狀應為({NUM_KEYPOINTS}, 3)，實際{self.points.shape}")

    def get(self, name: str) -> np.ndarray | None:
        p = self.points[keypoint_index(name)]
        return None if np.isnan(p).any() else p


def triangulate_person_keypoints(
    calib: StereoCalibrationResult,
    left: PersonKeypoints,
    right: PersonKeypoints,
    min_confidence: float = 0.0,
) -> PersonKeypoints3D:
    """只對左右都有效偵測（非NaN、信心>=門檻）的關節做三角測量，其餘填NaN。"""
    valid = (
        ~np.isnan(left.points).any(axis=1)
        & ~np.isnan(right.points).any(axis=1)
        & (left.confidences >= min_confidence)
        & (right.confidences >= min_confidence)
    )

    points_3d = np.full((NUM_KEYPOINTS, 3), np.nan, dtype=np.float64)
    if valid.any():
        points_3d[valid] = triangulate_points(calib, left.points[valid], right.points[valid])
    return PersonKeypoints3D(points=points_3d)
