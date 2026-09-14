from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .topology import NUM_KEYPOINTS, keypoint_index


@dataclass
class PersonKeypoints:
    points: np.ndarray  # shape (NUM_KEYPOINTS, 2) 像素座標，缺偵測為NaN
    confidences: np.ndarray  # shape (NUM_KEYPOINTS,)，缺偵測為0.0

    def __post_init__(self) -> None:
        if self.points.shape != (NUM_KEYPOINTS, 2):
            raise ValueError(f"points形狀應為({NUM_KEYPOINTS}, 2)，實際{self.points.shape}")
        if self.confidences.shape != (NUM_KEYPOINTS,):
            raise ValueError(f"confidences形狀應為({NUM_KEYPOINTS},)，實際{self.confidences.shape}")

    def get(self, name: str) -> np.ndarray | None:
        p = self.points[keypoint_index(name)]
        return None if np.isnan(p).any() else p


def keypoints_rmse(a: PersonKeypoints, b: PersonKeypoints, min_confidence: float = 0.0) -> float:
    """僅比較兩者皆有效（非NaN、信心>=門檻）的關鍵點座標RMSE，供FP32 vs FP16精度比對用。"""
    valid = (
        ~np.isnan(a.points).any(axis=1)
        & ~np.isnan(b.points).any(axis=1)
        & (a.confidences >= min_confidence)
        & (b.confidences >= min_confidence)
    )
    if not valid.any():
        raise ValueError("沒有共同有效的關鍵點可比較")
    diff = a.points[valid] - b.points[valid]
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
