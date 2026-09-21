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


def test_cpu_device_accepted(tmp_path):
    """這個模型本來就是為CPU設計的，GPU環境還沒弄好時要能先驗證流程。"""
    engine = LightweightOpenPoseEngine(_paths(tmp_path), precision="fp32", device="cpu")
    assert isinstance(engine, PoseEngine)


def test_fp16_on_cpu_rejected_with_a_useful_message(tmp_path):
    """fp16走TensorRT，CPU上不可能。錯誤訊息要直接指出該改什麼。"""
    with pytest.raises(ValueError) as e:
        LightweightOpenPoseEngine(_paths(tmp_path), precision="fp16", device="cpu")
    assert "fp32" in str(e.value)


def test_invalid_device_rejected(tmp_path):
    with pytest.raises(ValueError):
        LightweightOpenPoseEngine(_paths(tmp_path), precision="fp32", device="mps")


def test_missing_cuda_message_names_the_jetson_wheel_and_the_cpu_fallback():
    """Jetson 上裝到一般 pip PyTorch 是最常見的狀況，訊息要講清楚兩件事：
    要換 NVIDIA 的 wheel，以及在那之前可以先用 CPU。"""
    class _NoCuda:
        class cuda:
            @staticmethod
            def is_available():
                return False

    with pytest.raises(RuntimeError) as e:
        LightweightOpenPoseEngine._require_cuda(_NoCuda)
    message = str(e.value)
    assert "Jetson 專用 wheel" in message
    assert "--device cpu" in message
