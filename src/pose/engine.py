from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .keypoints import PersonKeypoints

_PRECISIONS = ("fp32", "fp16")


@runtime_checkable
class PoseEngine(Protocol):
    def infer(self, bgr_frame: np.ndarray) -> list[PersonKeypoints]: ...


@dataclass
class TrtPoseModelPaths:
    checkpoint: Path  # 例如 data/pose_models/resnet18_baseline_att_224x224_A_epoch_249.pth
    engine_cache: Path  # 例如 data/pose_models/resnet18_fp16.pth（torch2trt狀態快取）
    topology_json: Path  # 例如 data/pose_models/human_pose.json（複製自trt_pose repo）


class TrtPoseEngine:
    """trt_pose + torch2trt包裝。

    所有torch/trt_pose匯入都延遲到方法內部執行，讓沒裝torch的機器仍可
    `import pose.engine`，只有真的instantiate這個class才需要torch/CUDA/trt_pose。
    內部實作（_build_fp32/_build_or_load_trt/_parse）依公開資料設計，
    要到Jetson上對照實際clone下來的trt_pose/torch2trt原始碼調整。
    """

    def __init__(self, paths: TrtPoseModelPaths, precision: str = "fp16", input_size: int = 224):
        if precision not in _PRECISIONS:
            raise ValueError(f"precision必須是{_PRECISIONS}其中之一，收到{precision}")
        self._paths = paths
        self._precision = precision
        self._input_size = input_size
        self._model = None
        self._topology = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import json

        import trt_pose.coco

        self._topology = trt_pose.coco.coco_category_to_topology(
            json.loads(self._paths.topology_json.read_text(encoding="utf-8"))
        )
        if self._precision == "fp16":
            self._model = self._build_or_load_trt()
        else:
            self._model = self._build_fp32()

    def _build_fp32(self):
        import torch
        import trt_pose.models

        num_parts = self._topology["cmap"].shape[0]
        num_links = self._topology["paf"].shape[0] // 2
        model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
        model.load_state_dict(torch.load(self._paths.checkpoint))
        return model

    def _build_or_load_trt(self):
        import torch
        from torch2trt import TRTModule, torch2trt

        if self._paths.engine_cache.exists():
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(self._paths.engine_cache))
            return model_trt

        fp32_model = self._build_fp32()
        dummy = torch.zeros((1, 3, self._input_size, self._input_size)).cuda()
        model_trt = torch2trt(fp32_model, [dummy], fp16_mode=True)
        self._paths.engine_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_trt.state_dict(), self._paths.engine_cache)
        return model_trt

    def infer(self, bgr_frame: np.ndarray) -> list[PersonKeypoints]:
        self._ensure_loaded()
        import torch

        from .preprocess import resize_and_normalize

        arr = resize_and_normalize(bgr_frame, self._input_size)
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).cuda()
        with torch.no_grad():
            cmap, paf = self._model(tensor)
        return self._parse(cmap, paf)

    def _parse(self, cmap, paf) -> list[PersonKeypoints]:
        from .topology import NUM_KEYPOINTS
        from trt_pose.parse_objects import ParseObjects

        parse_objects = ParseObjects(self._topology)
        counts, objects, peaks = parse_objects(cmap, paf)

        results: list[PersonKeypoints] = []
        for i in range(int(counts[0])):
            obj = objects[0][i]
            points = np.full((NUM_KEYPOINTS, 2), np.nan, dtype=np.float32)
            confidences = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
            for j in range(NUM_KEYPOINTS):
                k = int(obj[j])
                if k >= 0:
                    peak = peaks[0][j][k]
                    points[j] = [float(peak[1]), float(peak[0])]
                    confidences[j] = 1.0
            results.append(PersonKeypoints(points=points, confidences=confidences))
        return results
