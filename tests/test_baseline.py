"""個人基準（θ_offset）。

判定門檻不能直接套在原始角度上：頸部解剖結構因人而異，同樣坐得端正，
兩個人量到的 θ_CA 可以差十幾度。基準取錯的後果比執行期單幀算錯嚴重：
執行期的雜訊會被平均掉，基準的偏差會固定留在之後每一次判定裡。
"""
import numpy as np
import pytest

from geometry.baseline import BaselineCollector, PostureBaseline
from geometry.pipeline import measure_posture
from geometry_synthetic import make_synthetic_stereo_calibration, project_point
from pose.keypoints import PersonKeypoints
from pose.topology import COCO18_KEYPOINT_NAMES, NUM_KEYPOINTS

K = np.array([[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]])
BASELINE_MM = 60.0


def _pose(theta_ca_deg: float, jitter_px: float = 0.0, seed: int = 0):
    ear_len, half = 120.0, 180.0
    dy = -ear_len * np.cos(np.radians(theta_ca_deg))
    dz = -ear_len * np.sin(np.radians(theta_ca_deg))
    right_sho = np.array([-half, 0.0, 600.0])
    left_sho = np.array([half, 0.0, 600.0])
    pose = {
        "right_shoulder": right_sho,
        "left_shoulder": left_sho,
        "right_ear": right_sho + np.array([40.0, dy, dz]),
        "left_ear": left_sho + np.array([-40.0, dy, dz]),
    }
    rng = np.random.default_rng(seed)
    left = np.full((NUM_KEYPOINTS, 2), np.nan, np.float32)
    right = np.full((NUM_KEYPOINTS, 2), np.nan, np.float32)
    conf = np.zeros(NUM_KEYPOINTS, np.float32)
    for name, p in pose.items():
        i = COCO18_KEYPOINT_NAMES.index(name)
        left[i] = project_point(K, np.eye(3), np.zeros((3, 1)), p) + rng.normal(0, jitter_px, 2)
        right[i] = project_point(
            K, np.eye(3), np.array([[-BASELINE_MM], [0.0], [0.0]]), p
        ) + rng.normal(0, jitter_px, 2)
        conf[i] = 0.9
    return PersonKeypoints(left, conf), PersonKeypoints(right, conf.copy())


def _calib():
    return make_synthetic_stereo_calibration(K, BASELINE_MM)


def _collect(theta_ca_deg: float, frames: int = 60, jitter_px: float = 0.5):
    collector = BaselineCollector()
    calib = _calib()
    for i in range(frames):
        left, right = _pose(theta_ca_deg, jitter_px, seed=i)
        collector.add(measure_posture(calib, left, right))
    return collector


def test_baseline_recovers_the_posture_it_was_shown():
    collector = _collect(12.0)
    baseline = collector.finish("A", 10.0)
    assert baseline.theta_ca_deg == pytest.approx(12.0, abs=2.0)
    assert baseline.frames >= 20


def test_two_people_sitting_equally_upright_get_different_baselines():
    """這就是為什麼門檻不能套在原始角度上。

    兩個人都坐得端正，但頸部結構不同，量到的 θ_CA 差 12 度。
    直接比 10° 的門檻，一個永遠報警、另一個永遠不報。
    """
    a = _collect(4.0).finish("A", 10.0)
    b = _collect(16.0).finish("B", 10.0)
    assert abs(a.theta_ca_deg - b.theta_ca_deg) > 10.0

    # 兩人都往前傾 8 度，扣掉各自的基準之後偏移量應該一致。
    # 容許值取基準本身的標準誤差，那是這個方法固有的殘差。
    leaned_a, _ = a.correct(4.0 + 8.0, 0.0)
    leaned_b, _ = b.correct(16.0 + 8.0, 0.0)
    tolerance = 3 * max(a.theta_ca_standard_error_deg, b.theta_ca_standard_error_deg)
    assert leaned_a == pytest.approx(8.0, abs=tolerance)
    assert leaned_b == pytest.approx(8.0, abs=tolerance)
    assert abs(leaned_a - leaned_b) < tolerance


def test_correct_passes_none_through():
    """角度算不出來時不能變成 -offset，那會被當成一個真實的偏移量。"""
    baseline = _collect(10.0).finish("A", 10.0)
    assert baseline.correct(None, None) == (None, None)


def test_refuses_to_produce_a_baseline_from_too_few_frames():
    """幀數不足時要明講，不能給出一個看起來正常的數字。"""
    collector = _collect(10.0, frames=5)
    with pytest.raises(ValueError, match="至少要"):
        collector.finish("A", 1.0)


def test_warns_when_the_subject_moved_during_the_baseline():
    """實測散佈遠大於理論量測雜訊，多的那部分只可能來自受試者本人。"""
    collector = BaselineCollector()
    calib = _calib()
    rng = np.random.default_rng(7)
    for i in range(60):
        # 每一幀的姿勢都在變，等於受試者一直在動
        left, right = _pose(10.0 + rng.normal(0, 12.0), jitter_px=0.5, seed=i)
        collector.add(measure_posture(calib, left, right))
    baseline = collector.finish("A", 10.0)
    assert any("動了" in w for w in collector.quality_warnings(baseline))


def test_a_still_subject_produces_no_movement_warning():
    collector = _collect(10.0)
    baseline = collector.finish("A", 10.0)
    assert not any("動了" in w for w in collector.quality_warnings(baseline))


def test_round_trips_through_a_file(tmp_path):
    baseline = _collect(10.0).finish("chenyue", 10.0)
    path = tmp_path / "b.json"
    baseline.save(path)
    assert PostureBaseline.load(path) == baseline


def test_rejects_a_baseline_file_missing_fields(tmp_path):
    """欄位少了就是舊版檔案，沿用的話，判定基準會錯而且看不出來。"""
    path = tmp_path / "old.json"
    path.write_text('{"subject": "A", "theta_ca_deg": 5.0}', encoding="utf-8")
    with pytest.raises(ValueError, match="缺少欄位"):
        PostureBaseline.load(path)


def test_describe_names_the_conditions_it_was_taken_under():
    text = _collect(10.0).finish("chenyue", 10.0).describe()
    for expected in ("chenyue", "θ_CA", "距離", "方位角"):
        assert expected in text


def test_refuses_to_overwrite_an_existing_baseline(tmp_path):
    """連續替幾位受試者取基準時很容易忘記換檔名。

    蓋掉的話前一位的判定基準就沒了，而且不會有任何跡象：
    檔案還在，內容卻換成了另一個人的。
    """
    path = tmp_path / "b.json"
    first = _collect(5.0).finish("A", 10.0)
    first.save(path)

    second = _collect(18.0).finish("B", 10.0)
    with pytest.raises(FileExistsError, match="overwrite"):
        second.save(path)
    assert PostureBaseline.load(path).subject == "A"

    second.save(path, overwrite=True)
    assert PostureBaseline.load(path).subject == "B"
