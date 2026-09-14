from pose.topology import COCO18_KEYPOINT_NAMES, NECK_INDEX, NUM_KEYPOINTS, keypoint_index


def test_keypoint_names_unique_and_complete():
    assert NUM_KEYPOINTS == 18
    assert len(set(COCO18_KEYPOINT_NAMES)) == NUM_KEYPOINTS


def test_neck_present():
    assert "neck" in COCO18_KEYPOINT_NAMES
    assert COCO18_KEYPOINT_NAMES[NECK_INDEX] == "neck"


def test_keypoint_index_roundtrip():
    for i, name in enumerate(COCO18_KEYPOINT_NAMES):
        assert keypoint_index(name) == i
