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
        self._human_pose = None
        self._parse_objects = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import json

        import trt_pose.coco
        from trt_pose.parse_objects import ParseObjects

        # coco_category_to_topology回傳的是tensor不是dict，關節點/連結數量要從原始json拿
        self._human_pose = json.loads(self._paths.topology_json.read_text(encoding="utf-8"))
        self._topology = trt_pose.coco.coco_category_to_topology(self._human_pose)
        # ParseObjects建構成本不低，建一次重複用，不要每幀重建
        self._parse_objects = ParseObjects(self._topology)

        if self._precision == "fp16":
            self._model = self._build_or_load_trt()
        else:
            self._model = self._build_fp32()

    def _build_fp32(self):
        import torch
        import trt_pose.models

        num_parts = len(self._human_pose["keypoints"])
        num_links = len(self._human_pose["skeleton"])
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
        frame_h, frame_w = bgr_frame.shape[:2]
        return self._parse(cmap, paf, frame_w, frame_h)

    def _parse(self, cmap, paf, frame_w: int, frame_h: int) -> list[PersonKeypoints]:
        from .topology import NUM_KEYPOINTS

        counts, objects, peaks = self._parse_objects(cmap, paf)
        cmap_h, cmap_w = int(cmap.shape[2]), int(cmap.shape[3])

        results: list[PersonKeypoints] = []
        for i in range(int(counts[0])):
            obj = objects[0][i]
            points = np.full((NUM_KEYPOINTS, 2), np.nan, dtype=np.float32)
            confidences = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
            for j in range(NUM_KEYPOINTS):
                k = int(obj[j])
                if k < 0:
                    continue
                # peaks是正規化座標(y, x, 值域0~1)，要乘回原始畫面尺寸才是像素座標。
                # 前處理把整張畫面縮成正方形輸入，所以乘原始寬高剛好抵銷這個縮放。
                peak_y, peak_x = float(peaks[0][j][k][0]), float(peaks[0][j][k][1])
                points[j] = [peak_x * frame_w, peak_y * frame_h]
                # 信心度取cmap在該峰值位置的數值，不能寫死1.0——
                # 下游triangulate_person_keypoints要靠這個值做min_confidence過濾
                row = min(max(int(peak_y * cmap_h), 0), cmap_h - 1)
                col = min(max(int(peak_x * cmap_w), 0), cmap_w - 1)
                confidences[j] = float(cmap[0][j][row][col])
            results.append(PersonKeypoints(points=points, confidences=confidences))
        return results
