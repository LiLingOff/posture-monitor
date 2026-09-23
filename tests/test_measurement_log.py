"""逐幀 CSV 記錄。

找人來坐二十分鐘，結束時終端機只剩最後一行。要做 Kinovea 對標、算準確率、
在報告裡放圖，都需要這份原始資料。
"""
import csv

import numpy as np
import pytest

from geometry.measurement_log import MeasurementLog
from geometry.pipeline import measure_posture
from geometry_synthetic import make_synthetic_stereo_calibration, project_point
from pose.keypoints import PersonKeypoints
from pose.topology import COCO18_KEYPOINT_NAMES, NUM_KEYPOINTS

K = np.array([[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]])


def _measurement():
    pose = {
        "right_shoulder": np.array([-180.0, 0.0, 600.0]),
        "left_shoulder": np.array([180.0, 0.0, 600.0]),
        "right_ear": np.array([-140.0, -116.0, 569.0]),
        "left_ear": np.array([140.0, -116.0, 569.0]),
    }
    left = np.full((NUM_KEYPOINTS, 2), np.nan, np.float32)
    right = np.full((NUM_KEYPOINTS, 2), np.nan, np.float32)
    conf = np.zeros(NUM_KEYPOINTS, np.float32)
    for name, p in pose.items():
        i = COCO18_KEYPOINT_NAMES.index(name)
        left[i] = project_point(K, np.eye(3), np.zeros((3, 1)), p)
        right[i] = project_point(K, np.eye(3), np.array([[-60.0], [0.0], [0.0]]), p)
        conf[i] = 0.9
    calib = make_synthetic_stereo_calibration(K, 60.0)
    return measure_posture(calib, PersonKeypoints(left, conf), PersonKeypoints(right, conf.copy()))


def _rows(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def test_writes_one_row_per_frame_with_the_angles(tmp_path):
    path = tmp_path / "s.csv"
    with MeasurementLog(path, subject="A") as log:
        log.write(_measurement(), corrected=(3.0, -1.0), ca_mean=2.5, sym_mean=-1.2)

    rows = _rows(path)
    assert len(rows) == 1
    assert rows[0]["usable"] == "1"
    assert float(rows[0]["theta_ca_corrected_deg"]) == pytest.approx(3.0)
    assert float(rows[0]["theta_ca_mean_deg"]) == pytest.approx(2.5)
    assert float(rows[0]["distance_mm"]) > 0


def test_records_the_frames_that_were_skipped(tmp_path):
    """略過率本身就是結果。只記成功的幀會讓事後看不出資料有多少缺口。"""
    path = tmp_path / "s.csv"
    with MeasurementLog(path) as log:
        log.write(_measurement())
        log.write(_measurement(), reject_reason="深度 74mm 落在合理範圍外")

    rows = _rows(path)
    assert [r["usable"] for r in rows] == ["1", "0"]
    assert "74mm" in rows[1]["reject_reason"]
    assert [int(r["frame"]) for r in rows] == [1, 2]


def test_keeps_both_the_raw_and_the_corrected_angle(tmp_path):
    """基準日後可能重取，原始值留著才能重算。"""
    path = tmp_path / "s.csv"
    with MeasurementLog(path) as log:
        log.write(_measurement(), corrected=(3.0, -1.0))

    row = _rows(path)[0]
    assert row["theta_ca_deg"] and row["theta_ca_corrected_deg"]
    assert float(row["theta_ca_deg"]) != pytest.approx(float(row["theta_ca_corrected_deg"]))


def test_missing_values_are_blank_not_zero(tmp_path):
    """0 是合法的角度值，拿它表示缺失會讓缺口變成一個看起來正常的量測。"""
    path = tmp_path / "s.csv"
    with MeasurementLog(path) as log:
        log.write(None, reject_reason="讀取相機影格失敗")

    row = _rows(path)[0]
    assert row["theta_ca_deg"] == ""
    assert row["distance_mm"] == ""
    assert row["usable"] == "0"


def test_each_row_is_flushed_so_a_ctrl_c_keeps_the_data(tmp_path):
    """這個程式是用 Ctrl-C 結束的，累積到最後才寫等於什麼都沒記。"""
    path = tmp_path / "s.csv"
    log = MeasurementLog(path)
    log.write(_measurement())
    assert len(_rows(path)) == 1, "還沒 close 就應該讀得到"
    log.close()


def test_subject_is_written_as_a_comment_line(tmp_path):
    path = tmp_path / "s.csv"
    with MeasurementLog(path, subject="chenyue") as log:
        log.write(_measurement())
    assert path.read_text(encoding="utf-8").startswith("# subject=chenyue")
