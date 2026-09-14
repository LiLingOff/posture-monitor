"""測試用假引擎，符合pose.engine.PoseEngine Protocol，不需要torch/GPU。"""
from __future__ import annotations

import time

import numpy as np

from pose.keypoints import PersonKeypoints
from pose.topology import NUM_KEYPOINTS


class FakeEngine:
    def __init__(self, base_points: np.ndarray, noise_std: float = 0.0, latency_s: float = 0.0):
        self._base = base_points
        self._noise_std = noise_std
        self._latency_s = latency_s

    def infer(self, bgr_frame: np.ndarray) -> list[PersonKeypoints]:
        if self._latency_s:
            time.sleep(self._latency_s)
        noisy = self._base + np.random.normal(0, self._noise_std, self._base.shape)
        return [PersonKeypoints(points=noisy.astype(np.float32), confidences=np.ones(NUM_KEYPOINTS, dtype=np.float32))]


def make_base_points(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(50, 400, size=(NUM_KEYPOINTS, 2)).astype(np.float32)
