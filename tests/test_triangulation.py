import numpy as np

from geometry.triangulation import triangulate_point, triangulate_points
from geometry_synthetic import make_synthetic_stereo_calibration, project_point

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

# 任意、非共平面的3D點(mm)，z為深度(相機前方)
KNOWN_POINTS_3D = np.array(
    [
        [0.0, 0.0, 500.0],
        [50.0, 30.0, 600.0],
        [-40.0, -20.0, 550.0],
        [10.0, -50.0, 700.0],
        [-25.0, 15.0, 450.0],
    ]
)


def _project_pairs(points_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.array([project_point(K, R_IDENTITY, T_LEFT, p) for p in points_3d])
    right = np.array([project_point(K, R_IDENTITY, T_RIGHT, p) for p in points_3d])
    return left, right


def test_triangulate_points_recovers_known_3d_points():
    calib = make_synthetic_stereo_calibration(K, BASELINE_MM)
    points_left, points_right = _project_pairs(KNOWN_POINTS_3D)

    recovered = triangulate_points(calib, points_left, points_right)

    assert recovered.shape == KNOWN_POINTS_3D.shape
    assert np.allclose(recovered, KNOWN_POINTS_3D, atol=1e-2)


def test_triangulate_point_single():
    calib = make_synthetic_stereo_calibration(K, BASELINE_MM)
    point_3d = KNOWN_POINTS_3D[0]
    left = project_point(K, R_IDENTITY, T_LEFT, point_3d)
    right = project_point(K, R_IDENTITY, T_RIGHT, point_3d)

    recovered = triangulate_point(calib, left, right)

    assert recovered.shape == (3,)
    assert np.allclose(recovered, point_3d, atol=1e-2)


def test_triangulate_points_nan_rows_stay_nan():
    calib = make_synthetic_stereo_calibration(K, BASELINE_MM)
    points_left, points_right = _project_pairs(KNOWN_POINTS_3D)
    points_left[1] = np.nan  # 模擬該點在左相機缺偵測

    recovered = triangulate_points(calib, points_left, points_right)

    assert np.isnan(recovered[1]).all()
    others = [i for i in range(len(KNOWN_POINTS_3D)) if i != 1]
    assert np.allclose(recovered[others], KNOWN_POINTS_3D[others], atol=1e-2)


def test_triangulate_points_empty_input_returns_all_nan():
    calib = make_synthetic_stereo_calibration(K, BASELINE_MM)
    empty = np.full((3, 2), np.nan)

    recovered = triangulate_points(calib, empty, empty)

    assert recovered.shape == (3, 3)
    assert np.isnan(recovered).all()
