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
