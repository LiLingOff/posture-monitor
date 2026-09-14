import numpy as np
import pytest

from pose.keypoints import PersonKeypoints, keypoints_rmse
from pose.topology import NUM_KEYPOINTS


def _make(points: np.ndarray, confidences: np.ndarray | None = None) -> PersonKeypoints:
    if confidences is None:
        confidences = np.ones(NUM_KEYPOINTS, dtype=np.float32)
    return PersonKeypoints(points=points.astype(np.float32), confidences=confidences.astype(np.float32))


def test_bad_points_shape_raises():
    with pytest.raises(ValueError):
        PersonKeypoints(points=np.zeros((3, 2)), confidences=np.ones(NUM_KEYPOINTS))


def test_bad_confidences_shape_raises():
    with pytest.raises(ValueError):
        PersonKeypoints(points=np.zeros((NUM_KEYPOINTS, 2)), confidences=np.ones(3))


def test_get_returns_none_for_nan():
    points = np.zeros((NUM_KEYPOINTS, 2))
    points[0] = np.nan
    kp = _make(points)
    assert kp.get("nose") is None
    assert kp.get("left_eye") is not None


def test_rmse_zero_for_identical():
    points = np.arange(NUM_KEYPOINTS * 2).reshape(NUM_KEYPOINTS, 2).astype(np.float32)
    a = _make(points)
    b = _make(points.copy())
    assert keypoints_rmse(a, b) == pytest.approx(0.0)


def test_rmse_known_offset():
    points = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
    a = _make(points)
    b = _make(points + np.array([3.0, 4.0]))  # 每點offset(3,4) -> 距離5
    assert keypoints_rmse(a, b) == pytest.approx(5.0)


def test_rmse_raises_when_no_overlap():
    points = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
    confidences_a = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
    confidences_b = np.ones(NUM_KEYPOINTS, dtype=np.float32)
    a = _make(points, confidences_a)
    b = _make(points, confidences_b)
    with pytest.raises(ValueError):
        keypoints_rmse(a, b, min_confidence=0.5)
