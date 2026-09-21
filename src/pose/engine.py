from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .keypoints import PersonKeypoints
from .preprocess import (DEFAULT_INPUT_HEIGHT, DEFAULT_STRIDE,
                         DEFAULT_UPSAMPLE_RATIO, resize_and_pad,
                         restore_keypoint_coordinates)
from .topology import NUM_KEYPOINTS, UPSTREAM_KEYPOINT_NAMES

_PRECISIONS = ("fp32", "fp16")
_DEVICES = ("cuda", "cpu")

# 權重直連網址（Intel的伺服器，不需要登入）：
# https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth
_WEIGHTS_URL = (
    "https://download.01.org/opencv/openvino_training_extensions/"
    "models/human_pose_estimation/checkpoint_iter_370000.pth"
)


@runtime_checkable
class PoseEngine(Protocol):
    def infer(self, bgr_frame: np.ndarray) -> list[PersonKeypoints]: ...


@dataclass
class LightweightOpenPoseModelPaths:
    checkpoint: Path  # checkpoint_iter_370000.pth
    engine_cache: Path  # 例如 data/pose_models/lightweight_openpose_fp16.pth（torch2trt狀態快取）
    repo_dir: Path  # clone下來的lightweight-human-pose-estimation.pytorch


class LightweightOpenPoseEngine:
    """Lightweight OpenPose + torch2trt 包裝。

    採用這個模型而非trt_pose的理由：trt_pose的權重掛在Google Drive且長期無法下載，
    而這份權重由Intel的伺服器直接提供。方法學上兩者同屬Bottom-up + PAF，
    這份還是OpenPose本身的最佳化實作，比trt_pose更貼近原研究文件寫的OpenPose。

    所有torch匯入都延遲到方法內部執行，讓沒有安裝torch的機器仍可
    `import pose.engine`，只有實際建立這個class的實例時才需要torch/CUDA。

    上游repo沒有setup.py、不能pip安裝，所以要把clone下來的目錄加進sys.path。
    """

    def __init__(
        self,
        paths: LightweightOpenPoseModelPaths,
        precision: str = "fp16",
        device: str = "cuda",
        input_height: int = DEFAULT_INPUT_HEIGHT,
        stride: int = DEFAULT_STRIDE,
        upsample_ratio: int = DEFAULT_UPSAMPLE_RATIO,
    ):
        if precision not in _PRECISIONS:
            raise ValueError(f"precision必須是{_PRECISIONS}其中之一，收到{precision}")
        if device not in _DEVICES:
            raise ValueError(f"device必須是{_DEVICES}其中之一，收到{device}")
        if precision == "fp16" and device == "cpu":
            raise ValueError(
                "fp16走的是TensorRT，必須在CUDA上執行。CPU請用 --precision fp32"
            )
        self._paths = paths
        self._precision = precision
        self._device = device
        self._input_height = input_height
        self._stride = stride
        self._upsample_ratio = upsample_ratio
        self._model = None
        self._trt_input_shape: tuple[int, int] | None = None

    # ---------- 載入 ----------

    def _add_repo_to_path(self) -> None:
        repo_dir = self._paths.repo_dir.resolve()
        if not (repo_dir / "modules" / "pose.py").is_file():
            raise FileNotFoundError(
                f"{repo_dir} 看起來不是lightweight-human-pose-estimation.pytorch的目錄"
                f"（找不到modules/pose.py）。\n"
                f"git clone https://github.com/Daniil-Osokin/"
                f"lightweight-human-pose-estimation.pytorch"
            )
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))

    def _check_topology_matches(self) -> None:
        """核對模型自己的關鍵點順序與pose/topology.py是否一致。

        兩邊順序不同的話所有關鍵點會整組錯位，而且不會有任何錯誤訊息——
        耳朵的座標被當成肩膀，角度照樣算得出看似合理的數值。
        這次從trt_pose換過來，順序本來就變了（neck從索引17移到索引1），
        所以這個核對比先前更有必要。
        """
        from modules.pose import Pose

        names = tuple(Pose.kpt_names)
        if names == UPSTREAM_KEYPOINT_NAMES:
            return
        raise ValueError(
            f"{self._paths.repo_dir} 的 modules/pose.py 關鍵點順序與 pose/topology.py 不一致，"
            f"索引會整組錯位且不會有任何徵兆。\n"
            f"  模型        : {list(names)}\n"
            f"  topology.py : {list(UPSTREAM_KEYPOINT_NAMES)}\n"
            f"請同步更新 pose/topology.py 的兩個tuple並重跑 tests/test_pose_topology.py"
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # 先把repo接上並核對拓樸：這兩步不需要torch，
        # 擺在前面才能在拓樸對不上時立刻報錯，而不是等模型載入完。
        self._add_repo_to_path()
        self._check_topology_matches()

        if self._precision == "fp16":
            self._model = self._build_or_load_trt()
        else:
            self._model = self._build_fp32()

    def _build_fp32(self):
        import torch
        from models.with_mobilenet import PoseEstimationWithMobileNet
        from modules.load_state import load_state

        if not self._paths.checkpoint.is_file():
            raise FileNotFoundError(
                f"找不到權重 {self._paths.checkpoint}。直接下載（不需要登入）：\n"
                f"  wget {_WEIGHTS_URL}"
            )

        net = PoseEstimationWithMobileNet()
        # 上游的checkpoint是{'state_dict': ...}包一層，要用它自己的load_state拆。
        # weights_only=True擋掉pickle任意執行；這個檔案只有張量，不受影響。
        checkpoint = torch.load(
            self._paths.checkpoint, map_location="cpu", weights_only=True
        )
        load_state(net, checkpoint)
        net = net.eval()
        if self._device == "cpu":
            # 這個模型本來就是為CPU設計的（論文標題就是Real-time ... on CPU），
            # 所以GPU環境還沒弄好時，CPU模式足以驗證整條流程的正確性。
            return net
        self._require_cuda(torch)
        return net.cuda()

    @staticmethod
    def _require_cuda(torch) -> None:
        if torch.cuda.is_available():
            return
        raise RuntimeError(
            "PyTorch 看不到 CUDA。Jetson 上最常見的原因是裝到了一般的 pip PyTorch，"
            "它編譯時對應的 CUDA 比 JetPack 提供的驅動新，而 Jetson 的驅動綁在 JetPack 裡、"
            "不能單獨升級。\n"
            "請改裝 NVIDIA 的 Jetson 專用 wheel（見 README 的關鍵點偵測一節）。\n"
            "在那之前可以先用 --device cpu --precision fp32 驗證流程，"
            "這個模型本來就是為 CPU 設計的。"
        )

    def _build_or_load_trt(self):
        import torch
        from torch2trt import TRTModule, torch2trt

        if self._paths.engine_cache.exists():
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(self._paths.engine_cache))
            return model_trt

        if self._trt_input_shape is None:
            raise RuntimeError(
                "建立TensorRT engine需要固定的輸入尺寸，但還沒有看過任何一幀。"
                "請先用fp32跑一次、或改呼叫warmup(frame)"
            )

        self._require_cuda(torch)
        fp32_model = self._build_fp32()
        h, w = self._trt_input_shape
        dummy = torch.zeros((1, 3, h, w)).cuda()
        model_trt = torch2trt(fp32_model, [dummy], fp16_mode=True)
        self._paths.engine_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_trt.state_dict(), self._paths.engine_cache)
        return model_trt

    def warmup(self, bgr_frame: np.ndarray) -> None:
        """用一幀決定TensorRT engine的輸入尺寸。

        前處理是等比例縮放後補邊，寬度取決於畫面的長寬比，所以輸入尺寸
        由相機解析度唯一決定。TensorRT需要固定shape，這正好成立——
        前提是標定與執行用同一個解析度，那本來就是硬性要求。
        """
        padded, _ = resize_and_pad(bgr_frame, self._input_height, self._stride)
        self._trt_input_shape = (padded.shape[0], padded.shape[1])

    # ---------- 推論 ----------

    def infer(self, bgr_frame: np.ndarray) -> list[PersonKeypoints]:
        if self._trt_input_shape is None:
            self.warmup(bgr_frame)
        self._ensure_loaded()

        import torch

        padded, info = resize_and_pad(bgr_frame, self._input_height, self._stride)
        if (padded.shape[0], padded.shape[1]) != self._trt_input_shape:
            raise ValueError(
                f"這一幀的網路輸入尺寸是{padded.shape[1]}x{padded.shape[0]}，"
                f"與建立engine時的{self._trt_input_shape[1]}x{self._trt_input_shape[0]}不同。"
                f"執行期間不能更換相機解析度"
            )

        tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0).float().to(self._device)
        with torch.no_grad():
            stages_output = self._model(tensor)

        # 上游取的是最後兩個stage輸出：倒數第二個是heatmaps、最後一個是PAFs
        heatmaps = stages_output[-2].squeeze().cpu().numpy().transpose(1, 2, 0)
        pafs = stages_output[-1].squeeze().cpu().numpy().transpose(1, 2, 0)
        return self._parse(heatmaps, pafs, info)

    def _parse(self, heatmaps: np.ndarray, pafs: np.ndarray, info) -> list[PersonKeypoints]:
        import cv2
        from modules.keypoints import extract_keypoints, group_keypoints

        ratio = self._upsample_ratio
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
        pafs = cv2.resize(pafs, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)

        total = 0
        all_keypoints_by_type: list = []
        for kpt_idx in range(NUM_KEYPOINTS):  # 第19個通道是背景，不取
            total += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        if len(all_keypoints) == 0:
            return []

        # all_keypoints每列是(x, y, score, id)，座標仍在熱圖尺度上
        all_keypoints = np.asarray(all_keypoints, dtype=np.float64)
        restored = restore_keypoint_coordinates(
            all_keypoints[:, :2], info, self._stride, self._upsample_ratio
        )

        results: list[PersonKeypoints] = []
        for entry in pose_entries:
            if len(entry) == 0:
                continue
            points = np.full((NUM_KEYPOINTS, 2), np.nan, dtype=np.float32)
            confidences = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
            for kpt_id in range(NUM_KEYPOINTS):
                # 上游用-1表示沒偵測到，我們一律改成NaN
                index = entry[kpt_id]
                if index == -1.0:
                    continue
                index = int(index)
                points[kpt_id] = restored[index]
                # 第2欄是該關鍵點在熱圖上的峰值，下游靠它做min_confidence過濾
                confidences[kpt_id] = float(all_keypoints[index, 2])
            results.append(PersonKeypoints(points=points, confidences=confidences))
        return results
