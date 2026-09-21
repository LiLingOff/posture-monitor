"""載入模型前要核對關鍵點順序。

從trt_pose換到Lightweight OpenPose時順序整個變了（neck從索引17移到索引1），
這正是最危險的那種改動：數量一樣、名稱一樣，只有順序不同，
沿用舊索引不會有任何錯誤訊息，耳朵的座標會被當成肩膀用。

測試不需要torch或GPU：核對只讀上游的modules/pose.py，而且刻意排在torch匯入之前。
用一個假的repo目錄餵不同的kpt_names進去。
"""
import sys
import textwrap
from pathlib import Path

import pytest

from pose.engine import LightweightOpenPoseEngine, LightweightOpenPoseModelPaths
from pose.topology import UPSTREAM_KEYPOINT_NAMES


@pytest.fixture(autouse=True)
def _isolate_sys_modules(monkeypatch):
    """每個測試用自己的假repo，避免modules套件被前一個測試快取住。"""
    for name in list(sys.modules):
        if name == "modules" or name.startswith("modules."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))


def _fake_repo(tmp_path: Path, kpt_names) -> Path:
    repo = tmp_path / "lightweight-human-pose-estimation.pytorch"
    (repo / "modules").mkdir(parents=True)
    (repo / "modules" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "modules" / "pose.py").write_text(
        textwrap.dedent(
            f"""
            class Pose:
                num_kpts = {len(kpt_names)}
                kpt_names = {list(kpt_names)!r}
            """
        ),
        encoding="utf-8",
    )
    return repo


def _engine(tmp_path: Path, kpt_names) -> LightweightOpenPoseEngine:
    paths = LightweightOpenPoseModelPaths(
        checkpoint=tmp_path / "checkpoint_iter_370000.pth",
        engine_cache=tmp_path / "engine.pth",
        repo_dir=_fake_repo(tmp_path, kpt_names),
    )
    return LightweightOpenPoseEngine(paths, precision="fp32")


def test_accepts_matching_order(tmp_path):
    engine = _engine(tmp_path, UPSTREAM_KEYPOINT_NAMES)
    engine._add_repo_to_path()
    engine._check_topology_matches()


def test_rejects_trt_pose_ordering(tmp_path):
    """最實際的誤用：沿用舊模型的排序（neck擺在最後）。"""
    trt_pose_like = [n for n in UPSTREAM_KEYPOINT_NAMES if n != "neck"] + ["neck"]
    engine = _engine(tmp_path, trt_pose_like)
    engine._add_repo_to_path()
    with pytest.raises(ValueError) as e:
        engine._check_topology_matches()
    assert "不一致" in str(e.value)


def test_rejects_swapped_left_right(tmp_path):
    """左右對調不會改變數量，也不會改變名稱集合，只有順序不同。"""
    swapped = list(UPSTREAM_KEYPOINT_NAMES)
    swapped[2], swapped[5] = swapped[5], swapped[2]  # r_sho / l_sho
    engine = _engine(tmp_path, swapped)
    engine._add_repo_to_path()
    with pytest.raises(ValueError):
        engine._check_topology_matches()


def test_error_message_shows_both_lists(tmp_path):
    engine = _engine(tmp_path, ["nose", "neck"])
    engine._add_repo_to_path()
    with pytest.raises(ValueError) as e:
        engine._check_topology_matches()
    message = str(e.value)
    assert "模型" in message and "topology.py" in message
    assert "nose" in message


def test_load_path_runs_the_check_before_importing_torch(tmp_path):
    """核對要真的接在載入流程上，而不只是一個沒人呼叫的方法。

    順序不符時拿到的必須是ValueError；若拿到ImportError，
    代表核對被排在torch匯入之後、或根本沒有被呼叫。
    """
    engine = _engine(tmp_path, ["nose", "neck"])
    with pytest.raises(ValueError):
        engine._ensure_loaded()
