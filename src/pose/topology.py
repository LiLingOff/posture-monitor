from __future__ import annotations

# trt_pose預設human_pose.json拓樸：COCO 17點＋第18個"neck"點。
# 假設：trt_pose官方human_pose.json確實內建neck點，順序如下——
# 到Jetson裝好trt_pose後要用 json.load(open("human_pose.json"))["keypoints"] 核對，
# 若順序或是否存在neck不同，只需改這個tuple，不影響其他模組。
COCO18_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "neck",
)

NUM_KEYPOINTS = len(COCO18_KEYPOINT_NAMES)
NECK_INDEX = COCO18_KEYPOINT_NAMES.index("neck")


def keypoint_index(name: str) -> int:
    return COCO18_KEYPOINT_NAMES.index(name)
