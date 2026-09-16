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
    if len(frames) <= warmup:
        raise ValueError(f"影格數({len(frames)})必須大於warmup({warmup})，否則沒有資料可以計時")

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
    if len(frame_sets) <= warmup:
        raise ValueError(f"影格組數({len(frame_sets)})必須大於warmup({warmup})，否則沒有資料可以計時")

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

    # 某一幀兩邊沒有共同有效關鍵點時keypoints_rmse會拋例外，略過該幀就好，
    # 不該讓整批比對失敗
    rmses = []
    for a, b in zip(fp32_results, fp16_results):
        try:
            rmses.append(keypoints_rmse(a, b))
        except ValueError:
            continue

    if not rmses:
        raise ValueError("所有影格的FP32/FP16結果都沒有共同有效的關鍵點，無法比對")

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
