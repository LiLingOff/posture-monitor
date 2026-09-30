from pose.topology import (COCO18_KEYPOINT_NAMES, NECK_INDEX, NUM_KEYPOINTS,
                           UPSTREAM_KEYPOINT_NAMES, keypoint_index)


def test_matches_lightweight_openpose_order():
    """鎖住上游 modules/pose.py 的 kpt_names 順序。

    這份清單抄自 Daniil-Osokin/lightweight-human-pose-estimation.pytorch 的
    modules/pose.py。改動這個tuple等於改動所有關鍵點的索引，
    三角測量與角度計算全部會跟著錯位，所以直接把上游的值固定寫入作為基準。
    """
    upstream_kpt_names = (
        "nose", "neck",
        "r_sho", "r_elb", "r_wri", "l_sho", "l_elb", "l_wri",
        "r_hip", "r_knee", "r_ank", "l_hip", "l_knee", "l_ank",
        "r_eye", "l_eye",
        "r_ear", "l_ear",
    )
    assert UPSTREAM_KEYPOINT_NAMES == upstream_kpt_names


def test_full_names_line_up_with_upstream_short_names():
    """兩份清單必須逐一對應，對不上就代表有一邊改過卻忘了同步。"""
    short_to_full = {
        "nose": "nose", "neck": "neck",
        "r_sho": "right_shoulder", "r_elb": "right_elbow", "r_wri": "right_wrist",
        "l_sho": "left_shoulder", "l_elb": "left_elbow", "l_wri": "left_wrist",
        "r_hip": "right_hip", "r_knee": "right_knee", "r_ank": "right_ankle",
        "l_hip": "left_hip", "l_knee": "left_knee", "l_ank": "left_ankle",
        "r_eye": "right_eye", "l_eye": "left_eye",
        "r_ear": "right_ear", "l_ear": "left_ear",
    }
    expected = tuple(short_to_full[s] for s in UPSTREAM_KEYPOINT_NAMES)
    assert COCO18_KEYPOINT_NAMES == expected


def test_neck_moved_from_index_17_to_index_1():
    """換模型時最容易出錯的一點：neck的位置跟trt_pose時代不同。

    trt_pose是COCO 17點後面接neck（索引17），這裡的neck在索引1。
    沿用舊索引的話，頸部會被當成鼻子、肩膀會被當成頸部，一路錯下去。
    """
    assert NECK_INDEX == 1
    assert COCO18_KEYPOINT_NAMES[1] == "neck"
    assert COCO18_KEYPOINT_NAMES[17] == "left_ear"


def test_angle_keypoints_present():
    """θ_CA與θ_sym實際用到的四個點都要在。"""
    for name in ("left_ear", "right_ear", "left_shoulder", "right_shoulder"):
        assert name in COCO18_KEYPOINT_NAMES


def test_keypoint_names_unique_and_complete():
    assert NUM_KEYPOINTS == 18
    assert len(set(COCO18_KEYPOINT_NAMES)) == NUM_KEYPOINTS
    assert len(UPSTREAM_KEYPOINT_NAMES) == NUM_KEYPOINTS


def test_keypoint_index_roundtrip():
    for i, name in enumerate(COCO18_KEYPOINT_NAMES):
        assert keypoint_index(name) == i


# ---- 畫骨架用的連線表 --------------------------------------------------

def test_the_limb_table_is_written_with_names_not_numbers():
    """索引寫死的話，上面那份名稱清單改順序時連線會默默接到別的地方。"""
    from pose.topology import COCO18_LIMBS, _LIMB_NAMES, keypoint_index

    for a, b in _LIMB_NAMES:
        assert a in COCO18_KEYPOINT_NAMES and b in COCO18_KEYPOINT_NAMES
    assert COCO18_LIMBS == tuple(
        (keypoint_index(a), keypoint_index(b)) for a, b in _LIMB_NAMES
    )


def test_the_limb_table_is_a_skeleton_that_reaches_every_keypoint():
    """漏掉或打錯一條會變成一個連不到的點，數一數就抓得到。"""
    from pose.topology import COCO18_LIMBS

    edges = {frozenset(e) for e in COCO18_LIMBS}
    assert len(edges) == len(COCO18_LIMBS), "有重複的連線"
    assert all(len(e) == 2 for e in edges), "有連到自己的線"
    assert len(COCO18_LIMBS) == len(COCO18_KEYPOINT_NAMES) - 1
    assert {i for e in COCO18_LIMBS for i in e} == set(range(len(COCO18_KEYPOINT_NAMES)))


def test_the_segments_the_angles_use_are_not_in_the_limb_table():
    """耳到肩與雙肩之間是判定的依據，不是肢體。混進灰色骨架就看不出來
    哪一段才是現在在判的東西，所以疊圖時單獨用判定顏色畫。"""
    from pose.topology import COCO18_LIMBS, keypoint_index

    edges = {frozenset(e) for e in COCO18_LIMBS}
    for a, b in (("right_ear", "right_shoulder"), ("left_ear", "left_shoulder"),
                 ("left_shoulder", "right_shoulder")):
        assert frozenset((keypoint_index(a), keypoint_index(b))) not in edges
