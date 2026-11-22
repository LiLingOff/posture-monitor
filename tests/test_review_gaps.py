"""GPT 審查測試檔（2026-11-22）列出的缺漏，逐項核對後仍沒有覆蓋的部分。

其餘幾項（髖被擋不拖累 θ_CA／θ_sym、θ_KA 軸要開了才生效、θ 與 ± 的 ASCII 退路、
批次平均法比 std/√N 大）原本就有測試，不重複。
"""
import dataclasses
import json

import numpy as np
import pytest

from geometry.baseline import PostureBaseline
from geometry.judgement import Posture, PostureJudge
from geometry.pipeline import angle_keypoints
from geometry.smoothing import RollingAngle
from geometry.uncertainty import lag1_autocorrelation, standard_error
from monitor_fakes import baseline as _baseline


def _full(value: float, window: int = 4) -> RollingAngle:
    w = RollingAngle(window)
    for _ in range(window):
        w.add(value)
    return w


# ---- 1. 基準檔向後相容 ---------------------------------------------------

def _saved(tmp_path) -> dict:
    path = tmp_path / "b.json"
    _baseline().save(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _required(field: dataclasses.Field) -> bool:
    return (field.default is dataclasses.MISSING
            and field.default_factory is dataclasses.MISSING)


def test_every_field_with_a_default_may_be_missing_from_an_old_file(tmp_path):
    """後來加的欄位一定要有預設值：現有的基準檔缺那幾欄，讀不進來就沒有地方補。

    逐欄拿掉再讀，之後有人加了新欄位卻忘了預設值，這裡就會抓到。
    """
    data = _saved(tmp_path)
    optional = [f.name for f in dataclasses.fields(PostureBaseline) if not _required(f)]
    assert optional, "至少有 θ_KA、肩高這幾個有預設值的欄位"
    for name in optional:
        trimmed = {k: v for k, v in data.items() if k != name}
        path = tmp_path / f"no_{name}.json"
        path.write_text(json.dumps(trimmed), encoding="utf-8")
        loaded = PostureBaseline.load(path)
        default = next(f.default for f in dataclasses.fields(PostureBaseline) if f.name == name)
        assert getattr(loaded, name) == default, name


def test_every_field_without_a_default_is_required(tmp_path):
    data = _saved(tmp_path)
    for field in dataclasses.fields(PostureBaseline):
        if not _required(field):
            continue
        trimmed = {k: v for k, v in data.items() if k != field.name}
        path = tmp_path / f"missing_{field.name}.json"
        path.write_text(json.dumps(trimmed), encoding="utf-8")
        with pytest.raises(ValueError, match="缺少欄位"):
            PostureBaseline.load(path)


# ---- 2. 肩高軸要 watch_shoulder_drop() 才生效 -----------------------------

def test_the_shoulder_axis_is_inert_until_it_is_switched_on():
    judge = PostureJudge()
    ca, sym = _full(0.0), _full(0.0)
    verdicts = judge.update(ca, sym, _full(80.0))
    assert verdicts[-1].posture is Posture.OK, "沒開肩高軸，肩膀下沉 80mm 也不能影響判定"

    judge.watch_shoulder_drop(10.0)
    assert judge.update(ca, sym, _full(80.0))[-1].posture is Posture.OVER


# ---- 3. θ_KA 的頸與髖不進 angle_keypoints() -------------------------------

@pytest.mark.parametrize("side", [None, "left", "right"])
def test_trunk_keypoints_are_not_part_of_the_core_angle_keypoints(side):
    """髖常被桌面擋住；加進來的話 θ_CA、θ_sym 會跟著整幀被略過。"""
    used = angle_keypoints(side)
    for name in ("left_hip", "right_hip", "neck"):
        assert name not in used


# ---- 5. 誤差估計的邊界 ----------------------------------------------------

def test_empty_and_single_inputs_have_no_error_estimate():
    assert standard_error([]) is None
    assert standard_error([3.0]) is None
    assert lag1_autocorrelation([]) is None
    assert lag1_autocorrelation([1.0, 2.0]) is None


def test_two_samples_still_give_an_error_rather_than_nothing():
    assert standard_error([1.0, 3.0]) == pytest.approx(1.0)


def test_a_nan_sample_is_not_silently_turned_into_a_number():
    """含 NaN 的序列給出 NaN，呼叫端要先濾掉；不能算出一個看起來正常的誤差。"""
    values = np.array([1.0, 2.0, np.nan, 2.0, 1.0, 2.0, 1.5, 2.0, 1.0])
    assert np.isnan(standard_error(values))
