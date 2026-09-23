import numpy as np
import pytest

from pose.benchmark import (
    compare_precision_rmse,
    measure_latency,
    measure_sequential_multi_camera,
    select_engine_precision,
)
from pose.keypoints import PersonKeypoints
from pose.topology import NUM_KEYPOINTS
from pose_fakes import FakeEngine, make_base_points

_DUMMY_FRAME = np.zeros((10, 10, 3), dtype=np.uint8)


def test_measure_latency_reflects_injected_delay():
    engine = FakeEngine(make_base_points(), latency_s=0.01)
    frames = [_DUMMY_FRAME] * 8
    stats = measure_latency(engine, frames, warmup=2)

    assert len(stats.per_frame_ms) == 6
    assert stats.mean_ms == pytest.approx(10.0, abs=5.0)
    assert stats.fps == pytest.approx(1000.0 / stats.mean_ms)


def test_measure_sequential_multi_camera_sums_per_set():
    engine = FakeEngine(make_base_points(), latency_s=0.01)
    frame_sets = [[_DUMMY_FRAME, _DUMMY_FRAME, _DUMMY_FRAME] for _ in range(5)]
    stats = measure_sequential_multi_camera(engine, frame_sets, warmup=1)

    assert len(stats.per_frame_ms) == 4
    # 每組3張，每張約10ms，總延遲應約30ms
    assert stats.mean_ms == pytest.approx(30.0, abs=10.0)


def _kp(points: np.ndarray) -> PersonKeypoints:
    return PersonKeypoints(points=points.astype(np.float32), confidences=np.ones(NUM_KEYPOINTS, dtype=np.float32))


def test_compare_precision_rmse_passes_within_threshold():
    base = make_base_points()
    fp32_results = [_kp(base)]
    fp16_results = [_kp(base + 0.5)]

    mean_rmse, passed = compare_precision_rmse(fp32_results, fp16_results, threshold_px=3.0)
    assert passed
    assert mean_rmse == pytest.approx(0.5 * (2**0.5))


def test_compare_precision_rmse_fails_and_warns(capsys):
    base = make_base_points()
    fp32_results = [_kp(base)]
    fp16_results = [_kp(base + 50.0)]

    mean_rmse, passed = compare_precision_rmse(fp32_results, fp16_results, threshold_px=3.0)
    assert not passed
    captured = capsys.readouterr()
    assert "超出可接受門檻" in captured.out


def test_compare_precision_rmse_raises_on_length_mismatch():
    base = make_base_points()
    with pytest.raises(ValueError):
        compare_precision_rmse([_kp(base)], [_kp(base), _kp(base)])


def test_measure_latency_raises_when_frames_not_more_than_warmup():
    """影格數不足會讓timings是空的，np.mean(空陣列)回傳nan，寧可直接拋出例外也不要印出nan。"""
    engine = FakeEngine(make_base_points())
    with pytest.raises(ValueError):
        measure_latency(engine, [_DUMMY_FRAME] * 3, warmup=3)


def test_measure_sequential_multi_camera_raises_when_sets_not_more_than_warmup():
    engine = FakeEngine(make_base_points())
    with pytest.raises(ValueError):
        measure_sequential_multi_camera(engine, [[_DUMMY_FRAME]] * 2, warmup=2)


def test_compare_precision_rmse_skips_pairs_without_common_keypoints():
    """某一幀兩邊沒有共同有效關鍵點時略過該幀，不應讓整批比對中斷。"""
    base = make_base_points()
    nan_points = np.full((NUM_KEYPOINTS, 2), np.nan, dtype=np.float32)
    blind = PersonKeypoints(points=nan_points, confidences=np.zeros(NUM_KEYPOINTS, dtype=np.float32))

    fp32_results = [_kp(base), blind]
    fp16_results = [_kp(base + 0.5), blind]

    mean_rmse, passed = compare_precision_rmse(fp32_results, fp16_results, threshold_px=3.0)
    assert passed
    assert mean_rmse == pytest.approx(0.5 * (2**0.5))


def test_compare_precision_rmse_raises_when_no_pair_is_usable():
    nan_points = np.full((NUM_KEYPOINTS, 2), np.nan, dtype=np.float32)
    blind = PersonKeypoints(points=nan_points, confidences=np.zeros(NUM_KEYPOINTS, dtype=np.float32))
    with pytest.raises(ValueError):
        compare_precision_rmse([blind], [blind])


def test_select_engine_precision_falls_back_to_fp32():
    base = make_base_points()
    assert select_engine_precision([_kp(base)], [_kp(base + 50.0)]) == "fp32"
    assert select_engine_precision([_kp(base)], [_kp(base + 0.1)]) == "fp16"
