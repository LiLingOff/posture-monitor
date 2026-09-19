import numpy as np

from geometry.keypoints3d import PersonKeypoints3D, triangulate_person_keypoints
from geometry_synthetic import make_synthetic_stereo_calibration, project_point
from pose.keypoints import PersonKeypoints
from pose.topology import NUM_KEYPOINTS

K = np.array(
    [
        [700.0, 0.0, 320.0],
        [0.0, 700.0, 240.0],
        [0.0, 0.0, 1.0],
    ]
)
BASELINE_MM = 80.0
R_IDENTITY = np.eye(3)
T_LEFT = np.zeros((3, 1))
T_RIGHT = np.array([[BASELINE_MM], [0.0], [0.0]])


def _make_known_body_points(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-80, 80, size=(NUM_KEYPOINTS, 2))
    z = rng.uniform(400, 700, size=(NUM_KEYPOINTS, 1))
    return np.hstack([xy, z])


def test_bad_shape_raises():
    try:
        PersonKeypoints3D(points=np.zeros((3, 3)))
        assert False, "應拋出ValueError"
    except ValueError:
        pass


def test_get_returns_none_for_nan():
    points = np.zeros((NUM_KEYPOINTS, 3))
    points[0] = np.nan
    kp3d = PersonKeypoints3D(points=points)
    assert kp3d.get("nose") is None
    assert kp3d.get("left_eye") is not None


def test_triangulate_person_keypoints_recovers_known_points():
    calib = make_synthetic_stereo_calibration(K, BASELINE_MM)
    known_3d = _make_known_body_points()

    left_points = np.array([project_point(K, R_IDENTITY, T_LEFT, p) for p in known_3d])
    right_points = np.array([project_point(K, R_IDENTITY, T_RIGHT, p) for p in known_3d])

    left = PersonKeypoints(points=left_points.astype(np.float32), confidences=np.ones(NUM_KEYPOINTS, dtype=np.float32))
    right = PersonKeypoints(points=right_points.astype(np.float32), confidences=np.ones(NUM_KEYPOINTS, dtype=np.float32))

    result = triangulate_person_keypoints(calib, left, right)

    assert np.allclose(result.points, known_3d, atol=0.5)


def test_triangulate_person_keypoints_missing_in_one_side_becomes_nan():
    calib = make_synthetic_stereo_calibration(K, BASELINE_MM)
    known_3d = _make_known_body_points(seed=1)

    left_points = np.array([project_point(K, R_IDENTITY, T_LEFT, p) for p in known_3d])
    right_points = np.array([project_point(K, R_IDENTITY, T_RIGHT, p) for p in known_3d])

    # 索引3只在左側未偵測到，索引7只在右側未偵測到
    left_points[3] = np.nan
    right_points[7] = np.nan

    left = PersonKeypoints(points=left_points.astype(np.float32), confidences=np.ones(NUM_KEYPOINTS, dtype=np.float32))
    right = PersonKeypoints(points=right_points.astype(np.float32), confidences=np.ones(NUM_KEYPOINTS, dtype=np.float32))

    result = triangulate_person_keypoints(calib, left, right)

    assert np.isnan(result.points[3]).all()
    assert np.isnan(result.points[7]).all()
    others = [i for i in range(NUM_KEYPOINTS) if i not in (3, 7)]
    assert np.allclose(result.points[others], known_3d[others], atol=0.5)


def test_triangulate_person_keypoints_respects_min_confidence():
    calib = make_synthetic_stereo_calibration(K, BASELINE_MM)
    known_3d = _make_known_body_points(seed=2)

    left_points = np.array([project_point(K, R_IDENTITY, T_LEFT, p) for p in known_3d])
    right_points = np.array([project_point(K, R_IDENTITY, T_RIGHT, p) for p in known_3d])

    confidences_left = np.ones(NUM_KEYPOINTS, dtype=np.float32)
    confidences_left[5] = 0.1  # 信心度過低，應被門檻排除

    left = PersonKeypoints(points=left_points.astype(np.float32), confidences=confidences_left)
    right = PersonKeypoints(points=right_points.astype(np.float32), confidences=np.ones(NUM_KEYPOINTS, dtype=np.float32))

    result = triangulate_person_keypoints(calib, left, right, min_confidence=0.5)

    assert np.isnan(result.points[5]).all()
