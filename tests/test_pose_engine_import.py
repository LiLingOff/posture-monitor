"""確認pose.engine在沒有安裝torch的環境下也能import——
torch的import都延遲到LightweightOpenPoseEngine的方法內部才執行。
"""
from pathlib import Path

import pytest

from pose.engine import (LightweightOpenPoseEngine,
                         LightweightOpenPoseModelPaths, PoseEngine)


def _paths(tmp_path: Path) -> LightweightOpenPoseModelPaths:
    return LightweightOpenPoseModelPaths(
        checkpoint=tmp_path / "checkpoint_iter_370000.pth",
        engine_cache=tmp_path / "lightweight_openpose_fp16.pth",
        repo_dir=tmp_path / "lightweight-human-pose-estimation.pytorch",
    )


def test_model_paths_construction(tmp_path):
    paths = _paths(tmp_path)
    assert paths.checkpoint.name == "checkpoint_iter_370000.pth"
    assert paths.repo_dir.name == "lightweight-human-pose-estimation.pytorch"


def test_engine_construction_does_not_require_torch(tmp_path):
    engine = LightweightOpenPoseEngine(_paths(tmp_path), precision="fp16")
    assert isinstance(engine, PoseEngine)


def test_invalid_precision_rejected(tmp_path):
    with pytest.raises(ValueError):
        LightweightOpenPoseEngine(_paths(tmp_path), precision="int8")


def test_missing_repo_dir_names_the_clone_command(tmp_path):
    """上游沒有setup.py、不能pip安裝，所以路徑填錯是很常見的狀況，
    訊息要直接給出clone指令。"""
    engine = LightweightOpenPoseEngine(_paths(tmp_path), precision="fp32")
    with pytest.raises(FileNotFoundError) as e:
        engine._add_repo_to_path()
    assert "git clone" in str(e.value)
