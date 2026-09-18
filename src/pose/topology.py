from __future__ import annotations

# trt_pose預設human_pose.json拓樸：COCO 17點＋第18個"neck"點。
# 2026-09-18在Jetson實機上核對過（json.load(open("human_pose.json"))["keypoints"]），
# 名稱與順序完全一致。neck是真實偵測點而非左右肩推算的中點，
# 這正是當初選trt_pose而非COCO 17點模型的理由之一。
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
