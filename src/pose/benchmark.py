from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .engine import PoseEngine
from .keypoints import PersonKeypoints, keypoints_rmse

_DEFAULT_RMSE_THRESHOLD_PX = 3.0


@dataclass
class LatencyStats:
    per_frame_ms: list[float]

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.per_frame_ms))

    @property
    def fps(self) -> float:
        return 1000.0 / self.mean_ms if self.mean_ms > 0 else 0.0


def measure_latency(engine: PoseEngine, frames: list[np.ndarray], warmup: int = 3) -> LatencyStats:
    for f in frames[:warmup]:
        engine.infer(f)

    timings = []
    for f in frames[warmup:]:
        t0 = time.perf_counter()
        engine.infer(f)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return LatencyStats(per_frame_ms=timings)


def measure_sequential_multi_camera(
    engine: PoseEngine, frame_sets: list[list[np.ndarray]], warmup: int = 3
) -> LatencyStats:
    """frame_sets[i]是同一時刻要循序推論的畫面清單（例如[front, stereo_left, stereo_right]）。

    對應文件「三相機循序推論延遲基線」，這裡實際是正面+雙目切開後的2路feed。
    """
    for frames in frame_sets[:warmup]:
        for f in frames:
            engine.infer(f)

    timings = []
    for frames in frame_sets[warmup:]:
        t0 = time.perf_counter()
        for f in frames:
            engine.infer(f)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return LatencyStats(per_frame_ms=timings)


def compare_precision_rmse(
    fp32_results: list[PersonKeypoints],
    fp16_results: list[PersonKeypoints],
    threshold_px: float = _DEFAULT_RMSE_THRESHOLD_PX,
) -> tuple[float, bool]:
    """回傳(平均RMSE, 是否通過門檻)。不通過只印警告，不拋例外，比照標定模組風格。"""
    if len(fp32_results) != len(fp16_results):
        raise ValueError("FP32與FP16結果數量不一致，無法逐一比對")

    rmses = [keypoints_rmse(a, b) for a, b in zip(fp32_results, fp16_results)]
    mean_rmse = float(np.mean(rmses))
    passed = mean_rmse <= threshold_px

    if not passed:
        print(f"FP16關鍵點RMSE {mean_rmse:.4f}px 超出可接受門檻 {threshold_px}px，建議保留FP32為備案")

    return mean_rmse, passed


def select_engine_precision(
    fp32_results: list[PersonKeypoints],
    fp16_results: list[PersonKeypoints],
    threshold_px: float = _DEFAULT_RMSE_THRESHOLD_PX,
) -> str:
    """劣化超出門檻時回傳"fp32"，否則"fp16"。"""
    _, passed = compare_precision_rmse(fp32_results, fp16_results, threshold_px)
    return "fp16" if passed else "fp32"
