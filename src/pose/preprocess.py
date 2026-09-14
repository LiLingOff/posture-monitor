from __future__ import annotations

import cv2
import numpy as np

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_and_normalize(bgr_frame: np.ndarray, input_size: int = 224) -> np.ndarray:
    """BGR frame轉成(input_size,input_size,3) float32 RGB，ImageNet normalize。

    回傳HWC numpy陣列；轉成CHW/torch tensor留給engine.py內部做，這裡不碰torch。
    """
    resized = cv2.resize(bgr_frame, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
