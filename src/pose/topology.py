from __future__ import annotations

# Lightweight OpenPose（Daniil-Osokin/lightweight-human-pose-estimation.pytorch）的
# COCO 18點拓樸。順序取自該repo的 modules/pose.py：
#
#     kpt_names = ['nose', 'neck',
#                  'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
#                  'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
#                  'r_eye', 'l_eye',
#                  'r_ear', 'l_ear']
#
# 這跟先前用的trt_pose排序完全不同：trt_pose是COCO 17點後面接neck（neck在索引17），
# 這裡的neck在索引1、而且左右順序是先右後左。改動這個tuple等於改動所有關鍵點索引，
# 三角測量與角度計算會整組錯位且沒有任何徵兆，所以engine載入模型前會拿模型自己的
# kpt_names來核對（見engine.py的_check_topology_matches）。
COCO18_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
)

# 模型原始碼裡的簡寫名稱，順序必須與上面逐一對應。
# engine載入時會把這份清單跟modules.pose.Pose.kpt_names比對，
# 上游改版調動順序時會直接報錯，不會拖到角度算出來才發現。
UPSTREAM_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "neck",
    "r_sho",
    "r_elb",
    "r_wri",
    "l_sho",
    "l_elb",
    "l_wri",
    "r_hip",
    "r_knee",
    "r_ank",
    "l_hip",
    "l_knee",
    "l_ank",
    "r_eye",
    "l_eye",
    "r_ear",
    "l_ear",
)

NUM_KEYPOINTS = len(COCO18_KEYPOINT_NAMES)
NECK_INDEX = COCO18_KEYPOINT_NAMES.index("neck")


def keypoint_index(name: str) -> int:
    return COCO18_KEYPOINT_NAMES.index(name)


# 畫骨架用的連線。**只影響畫面**，與上面那兩份清單不同級：上面改順序會讓所有
# 角度默默錯位，這張表畫錯只是線接錯地方，看一眼就發現。也不必與上游的 PAF
# 配對表一致，那是組裝用的，這是給人看的。
#
# 只有名稱是字面值，索引在匯入時推導，所以 COCO18_KEYPOINT_NAMES 改順序時連線
# 跟著走，名稱打錯會在匯入當下 ValueError，而不是畫出一條接錯的線。
#
# 耳朵到肩膀與左右肩之間**故意不在這裡**。那兩段是量角度用的線段，不是肢體，
# 疊圖時單獨用判定顏色畫出來，混進灰色骨架反而看不出哪一段才是判定的依據。
_LIMB_NAMES: tuple[tuple[str, str], ...] = (
    ("neck", "nose"),
    ("nose", "right_eye"),
    ("right_eye", "right_ear"),
    ("nose", "left_eye"),
    ("left_eye", "left_ear"),
    ("neck", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("neck", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("neck", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("neck", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
)

COCO18_LIMBS: tuple[tuple[int, int], ...] = tuple(
    (keypoint_index(a), keypoint_index(b)) for a, b in _LIMB_NAMES
)
