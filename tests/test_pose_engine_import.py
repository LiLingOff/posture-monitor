"""確認pose.engine在沒有安裝torch與trt_pose的環境下也能import——
torch與trt_pose的import都延遲到TrtPoseEngine的方法內部才執行。
"""
from pathlib import Path

import pytest

from pose.engine import PoseEngine, TrtPoseEngine, TrtPoseModelPaths


def test_trt_pose_model_paths_construction():
    paths = TrtPoseModelPaths(
        checkpoint=Path("a.pth"), engine_cache=Path("b.pth"), topology_json=Path("c.json")
    )
    assert paths.checkpoint == Path("a.pth")


def test_trt_pose_engine_construction_does_not_require_torch():
    paths = TrtPoseModelPaths(Path("a.pth"), Path("b.pth"), Path("c.json"))
    engine = TrtPoseEngine(paths, precision="fp16")
    assert isinstance(engine, PoseEngine)


def test_invalid_precision_rejected():
    paths = TrtPoseModelPaths(Path("a.pth"), Path("b.pth"), Path("c.json"))
    with pytest.raises(ValueError):
        TrtPoseEngine(paths, precision="int8")
