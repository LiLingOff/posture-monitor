from pose.topology import COCO18_KEYPOINT_NAMES, NECK_INDEX, NUM_KEYPOINTS, keypoint_index


def test_matches_trt_pose_human_pose_json():
    """鎖定實機核對過的拓樸。

    2026-09-18在Jetson上從trt_pose的human_pose.json讀出來的清單，
    名稱與順序都與這裡一致。改動這個tuple等於改動所有關鍵點的索引，
    三角測量與角度計算全部會跟著錯位，所以直接把實機核對的結果固定寫入作為基準。
    """
    verified_on_jetson = (
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle", "neck",
    )
    assert COCO18_KEYPOINT_NAMES == verified_on_jetson


def test_keypoint_names_unique_and_complete():
    assert NUM_KEYPOINTS == 18
    assert len(set(COCO18_KEYPOINT_NAMES)) == NUM_KEYPOINTS


def test_neck_present():
    assert "neck" in COCO18_KEYPOINT_NAMES
    assert COCO18_KEYPOINT_NAMES[NECK_INDEX] == "neck"


def test_keypoint_index_roundtrip():
    for i, name in enumerate(COCO18_KEYPOINT_NAMES):
        assert keypoint_index(name) == i
