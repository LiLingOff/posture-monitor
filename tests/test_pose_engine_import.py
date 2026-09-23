"""確認pose.engine在沒有安裝torch的環境下也能import。
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


def test_engine_cache_path_includes_the_input_size(tmp_path):
    """每個輸入尺寸一個TensorRT engine，快取檔名要帶尺寸才不會互相覆蓋。

    正面相機 2560x720 補邊後是 912x256，雙目單眼 1280x720 是 456x256。
    同一次執行就會同時用到兩種，共用一個檔名的話後者會蓋掉前者。
    """
    engine = LightweightOpenPoseEngine(_paths(tmp_path), precision="fp32")
    assert engine._engine_cache_path((256, 912)).name == "lightweight_openpose_fp16_912x256.pth"
    assert engine._engine_cache_path((256, 456)).name == "lightweight_openpose_fp16_456x256.pth"
    assert engine._engine_cache_path((256, 912)) != engine._engine_cache_path((256, 456))


def test_fp32_reuses_one_model_across_input_sizes(tmp_path, monkeypatch):
    """fp32是全卷積網路，不同尺寸共用同一個模型，不該重建。

    先前把TensorRT的固定尺寸限制套用到fp32上，導致正面畫面與雙目半邊
    混在同一個引擎時直接拋例外。
    """
    engine = LightweightOpenPoseEngine(_paths(tmp_path), precision="fp32", device="cpu")
    monkeypatch.setattr(engine, "_ensure_loaded", lambda: None)

    builds = []

    def _fake_build():
        builds.append(1)
        return object()

    monkeypatch.setattr(engine, "_build_fp32", _fake_build)

    first = engine._model_for((256, 912))
    second = engine._model_for((256, 456))
    assert first is second
    assert len(builds) == 1, "fp32 不該為了不同尺寸重建模型"


def test_fp16_builds_one_engine_per_input_size(tmp_path, monkeypatch):
    engine = LightweightOpenPoseEngine(_paths(tmp_path), precision="fp16")
    monkeypatch.setattr(engine, "_ensure_loaded", lambda: None)

    built = []

    def _fake_build(shape):
        built.append(shape)
        return f"engine{shape}"

    monkeypatch.setattr(engine, "_build_or_load_trt", _fake_build)

    engine._model_for((256, 912))
    engine._model_for((256, 456))
    engine._model_for((256, 912))  # 已經建過，不該再建一次
    assert built == [(256, 912), (256, 456)]
