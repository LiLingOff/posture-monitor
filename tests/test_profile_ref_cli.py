"""profile-ref 子指令：用假引擎與合成圖片走完整條路，不需要相機、GPU 或手機。"""
import csv
import math

import cv2
import numpy as np
import pytest

from pose.keypoints import PersonKeypoints
from pose.topology import NUM_KEYPOINTS, keypoint_index

from posture_loader import load_posture_module


def _person(theta_deg, facing=1.0, shoulder=(300.0, 400.0), length=150.0):
    t = math.radians(theta_deg)
    ear = (shoulder[0] + facing * length * math.sin(t), shoulder[1] - length * math.cos(t))
    nose = (ear[0] + facing * 40.0, ear[1] + 8.0)
    pts = np.full((NUM_KEYPOINTS, 2), np.nan, dtype=np.float32)
    conf = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
    for name, xy in (("right_ear", ear), ("right_shoulder", shoulder), ("nose", nose)):
        pts[keypoint_index(name)] = xy
        conf[keypoint_index(name)] = 0.9
    return PersonKeypoints(points=pts, confidences=conf)


class _SequenceEngine:
    """每呼叫一次回傳序列裡的下一個結果，用完停在最後一個。"""

    def __init__(self, persons):
        self._persons = persons
        self.calls = 0

    def infer(self, bgr_frame):
        person = self._persons[min(self.calls, len(self._persons) - 1)]
        self.calls += 1
        return [] if person is None else [person]


def _images(folder, count):
    folder.mkdir()
    for i in range(count):
        cv2.imwrite(str(folder / f"{i:04d}.png"), np.zeros((48, 64, 3), np.uint8))


def _write_stereo(path, value, rows=40):
    lines = ["# subject=S01", "frame,usable,theta_ca_deg"] + [f"{i},1,{value}" for i in range(rows)]
    path.write_text("\n".join(lines), encoding="utf-8")


def _run(posture, monkeypatch, engine, argv):
    monkeypatch.setattr(posture, "_build_engine", lambda args: engine)
    args = posture.build_parser().parse_args(argv)
    posture._run_profile_ref(args)


def test_profile_ref_end_to_end(tmp_path, monkeypatch, capsys):
    posture = load_posture_module()
    _images(tmp_path / "shots", 20)
    stereo = tmp_path / "head.csv"
    _write_stereo(stereo, 30.0)
    # 前 10 張約 10°，後 10 張約 25°（圖片每張 0.2 秒，所以兩個時間窗各 2 秒）
    engine = _SequenceEngine([_person(10.0)] * 10 + [_person(25.0)] * 10)
    out = tmp_path / "out"
    _run(posture, monkeypatch, engine,
         ["profile-ref", str(tmp_path / "shots"), "--windows", "a=0:2,b=2:4",
          "--stereo-csv", f"b={stereo}", "--out", str(out)])
    text = capsys.readouterr().out
    assert "a " in text and "b " in text and "3D−參考" in text
    assert (out / "profile_report.txt").is_file()
    with open(out / "profile_frames.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 20 and rows[0]["window"] == "a" and rows[-1]["window"] == "b"
    assert float(rows[0]["angle_deg"]) == pytest.approx(10.0, abs=0.01)
    assert float(rows[-1]["angle_deg"]) == pytest.approx(25.0, abs=0.01)
    assert engine.calls == 20


def test_profile_ref_records_missing_person_and_does_not_crash(tmp_path, monkeypatch, capsys):
    posture = load_posture_module()
    _images(tmp_path / "shots", 12)
    engine = _SequenceEngine([_person(10.0)] * 8 + [None] * 4)
    _run(posture, monkeypatch, engine,
         ["profile-ref", str(tmp_path / "shots"), "--out", str(tmp_path / "o")])
    with open(tmp_path / "o" / "profile_frames.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[-1]["angle_deg"] == "" and "沒有偵測到" in rows[-1]["reason"]
    assert "all" in capsys.readouterr().out


def test_profile_ref_facing_left_and_forced_facing(tmp_path, monkeypatch, capsys):
    posture = load_posture_module()
    _images(tmp_path / "shots", 10)
    engine = _SequenceEngine([_person(15.0, facing=-1.0)] * 10)
    _run(posture, monkeypatch, engine, ["profile-ref", str(tmp_path / "shots"), "--out", str(tmp_path / "o")])
    with open(tmp_path / "o" / "profile_frames.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert float(rows[0]["angle_deg"]) == pytest.approx(15.0, abs=0.01)
    # 指定面向與實際相反時，同一組座標變成後仰（負）
    engine = _SequenceEngine([_person(15.0, facing=-1.0)] * 10)
    _run(posture, monkeypatch, engine,
         ["profile-ref", str(tmp_path / "shots"), "--facing", "right", "--out", str(tmp_path / "o2")])
    with open(tmp_path / "o2" / "profile_frames.csv", encoding="utf-8") as fh:
        assert float(next(csv.DictReader(fh))["angle_deg"]) == pytest.approx(-15.0, abs=0.01)


@pytest.mark.parametrize("extra, message", [
    (["--windows", "a=5:1"], "起點要早於"),
    (["--plumb", "1,2,3"], "x1,y1,x2,y2"),
    (["--windows", "a=0:2", "--stereo-csv", "zzz=x.csv"], "時間窗名稱"),
])
def test_profile_ref_rejects_bad_options(tmp_path, monkeypatch, extra, message):
    posture = load_posture_module()
    _images(tmp_path / "shots", 5)
    with pytest.raises(SystemExit) as exc:
        _run(posture, monkeypatch, _SequenceEngine([_person(10.0)]),
             ["profile-ref", str(tmp_path / "shots")] + extra)
    assert message in str(exc.value)


def test_profile_ref_missing_source_and_empty_folder(tmp_path, monkeypatch):
    posture = load_posture_module()
    with pytest.raises(SystemExit) as exc:
        _run(posture, monkeypatch, _SequenceEngine([None]), ["profile-ref", str(tmp_path / "nope.mp4")])
    assert "找不到" in str(exc.value)
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit) as exc:
        _run(posture, monkeypatch, _SequenceEngine([None]), ["profile-ref", str(tmp_path / "empty")])
    assert "沒有 jpg 或 png" in str(exc.value)


def test_profile_ref_reads_a_video_file(tmp_path, monkeypatch, capsys):
    posture = load_posture_module()
    video = tmp_path / "side.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("這台機器的 OpenCV 不能寫 MJPG 影片")
    for _ in range(40):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    engine = _SequenceEngine([_person(20.0)])
    _run(posture, monkeypatch, engine, ["profile-ref", str(video), "--fps", "5"])
    # 10 fps 的影片、取樣 5 fps：40 幀取 20 幀
    assert engine.calls == 20
    assert "+20.00" in capsys.readouterr().out
