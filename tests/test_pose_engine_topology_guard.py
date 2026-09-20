"""載入模型前要核對關鍵點拓樸。

json的關鍵點順序與pose/topology.py不同時，_parse會用錯的索引去讀模型輸出，
所有關鍵點整組錯位——耳朵的座標被當成肩膀，角度照樣算得出看似合理的數值，
不會有任何錯誤訊息。所以這個核對要在載入模型之前就擋下來。

測試不需要torch或GPU：核對只讀json，而且刻意排在trt_pose匯入之前。
"""
import json
from pathlib import Path

import pytest

from pose.engine import TrtPoseEngine, TrtPoseModelPaths
from pose.topology import COCO18_KEYPOINT_NAMES


def _engine(tmp_path: Path, keypoints) -> TrtPoseEngine:
    topology_json = tmp_path / "human_pose.json"
    topology_json.write_text(
        json.dumps({"keypoints": list(keypoints), "skeleton": []}), encoding="utf-8"
    )
    paths = TrtPoseModelPaths(
        checkpoint=tmp_path / "model.pth",
        engine_cache=tmp_path / "engine.pth",
        topology_json=topology_json,
    )
    return TrtPoseEngine(paths, precision="fp32")


def _engine_with_loaded_json(tmp_path: Path, keypoints) -> TrtPoseEngine:
    """跳過_ensure_loaded、直接驗證核對本身的行為。"""
    engine = _engine(tmp_path, keypoints)
    engine._human_pose = json.loads(engine._paths.topology_json.read_text(encoding="utf-8"))
    return engine


def test_load_path_runs_the_check_before_importing_trt_pose(tmp_path):
    """核對要真的接在載入流程上，而不只是一個沒人呼叫的方法。

    拓樸不符時拿到的必須是ValueError；若拿到ImportError，代表核對被排在
    trt_pose匯入之後、或根本沒有被呼叫。
    """
    swapped = list(COCO18_KEYPOINT_NAMES)
    swapped[3], swapped[4] = swapped[4], swapped[3]  # left_ear / right_ear 對調

    with pytest.raises(ValueError) as e:
        _engine(tmp_path, swapped)._ensure_loaded()
    assert "不一致" in str(e.value)


def test_matching_topology_passes_the_check_and_moves_on(tmp_path):
    """拓樸相符時要通過核對繼續往下走。

    這台開發機沒有trt_pose，所以往下走的表現就是匯入失敗——
    重點在於拿到的不是ValueError，代表核對本身沒有誤擋。
    """
    with pytest.raises(ImportError):
        _engine(tmp_path, COCO18_KEYPOINT_NAMES)._ensure_loaded()


def test_accepts_matching_topology(tmp_path):
    _engine_with_loaded_json(tmp_path, COCO18_KEYPOINT_NAMES)._check_topology_matches()


def test_rejects_different_keypoint_count(tmp_path):
    """COCO 17點模型沒有neck，少一個點。"""
    with pytest.raises(ValueError):
        _engine_with_loaded_json(tmp_path, COCO18_KEYPOINT_NAMES[:-1])._check_topology_matches()


def test_error_message_shows_both_lists(tmp_path):
    """訊息要直接列出兩邊清單，否則只知道不一致、不知道差在哪。"""
    with pytest.raises(ValueError) as e:
        _engine_with_loaded_json(tmp_path, ["nose", "neck"])._check_topology_matches()
    message = str(e.value)
    assert "json" in message and "topology.py" in message
    assert "nose" in message
