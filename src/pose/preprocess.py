"""Lightweight OpenPose 的前處理與座標還原。

前處理照該repo的 demo.py::infer_fast 與 val.py::pad_width 實作：
依高度等比例縮放，再補邊到stride的整數倍。等比例縮放這點很重要——
先前配合trt_pose的寫法是直接拉成正方形，模型看到的人體是變形的。

兩個容易寫錯而且不會報錯的地方：
1. 不做BGR轉RGB。上游是直接把OpenCV讀進來的BGR餵給網路的，跟著做才對得上權重。
2. 正規化是(img - 128) / 256，不是ImageNet的mean/std。

這裡只有純numpy運算，不依賴torch，所以可以在沒有GPU的機器上測。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

_IMG_MEAN = 128.0
_IMG_SCALE = 1.0 / 256.0

DEFAULT_INPUT_HEIGHT = 256
DEFAULT_STRIDE = 8
DEFAULT_UPSAMPLE_RATIO = 4


@dataclass(frozen=True)
class LetterboxInfo:
    """把網路輸出的座標換算回原始畫面所需的資訊。

    scale：原始畫面縮放到網路輸入的倍率。
    pad_top / pad_left：縮放後補在上方與左方的像素數（補邊是置中的）。
    """

    scale: float
    pad_top: int
    pad_left: int


def resize_and_pad(
    bgr_frame: np.ndarray,
    input_height: int = DEFAULT_INPUT_HEIGHT,
    stride: int = DEFAULT_STRIDE,
) -> tuple[np.ndarray, LetterboxInfo]:
    """回傳(HWC float32網路輸入, 座標還原資訊)。"""
    height, width = bgr_frame.shape[:2]
    if height == 0 or width == 0:
        raise ValueError(f"畫面尺寸不合法：{width}x{height}")

    scale = input_height / height
    scaled = cv2.resize(bgr_frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    normalized = (np.asarray(scaled, dtype=np.float32) - _IMG_MEAN) * _IMG_SCALE

    scaled_h, scaled_w = normalized.shape[:2]
    target_h = math.ceil(input_height / stride) * stride
    target_w = math.ceil(max(scaled_w, input_height) / stride) * stride

    pad_top = int(math.floor((target_h - scaled_h) / 2.0))
    pad_left = int(math.floor((target_w - scaled_w) / 2.0))
    pad_bottom = target_h - scaled_h - pad_top
    pad_right = target_w - scaled_w - pad_left

    padded = cv2.copyMakeBorder(
        normalized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )
    return padded, LetterboxInfo(scale=scale, pad_top=pad_top, pad_left=pad_left)


def restore_keypoint_coordinates(
    points: np.ndarray,
    info: LetterboxInfo,
    stride: int = DEFAULT_STRIDE,
    upsample_ratio: int = DEFAULT_UPSAMPLE_RATIO,
) -> np.ndarray:
    """把熱圖座標換算回原始畫面的像素座標。

    熱圖的解析度是網路輸入的 stride/upsample_ratio 倍，所以要先乘回去，
    再扣掉補邊、最後除以縮放倍率。漏掉任何一步都不會報錯，
    只會讓三角測量拿到系統性偏移的2D點。

    points 形狀(N,2)，可含NaN；NaN會原樣傳遞出去。
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points形狀應為(N, 2)，實際{points.shape}")

    factor = stride / upsample_ratio
    restored = np.empty_like(points)
    restored[:, 0] = (points[:, 0] * factor - info.pad_left) / info.scale
    restored[:, 1] = (points[:, 1] * factor - info.pad_top) / info.scale
    return restored
