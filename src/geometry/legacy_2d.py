"""前作的 2D θ_CA 公式，以及它隨方位角偏差的理論值。

存在的唯一理由是說明書表4：同一人、同一姿勢，在不同方位角各量一段，比較 3D 的
θ_CA 與前作 2D 公式算出的 θ_CA。**監測本身不用這裡的任何東西。**

前作（cam_side.py:115）的公式：

    θ_raw = arctan[ (x_ear − x_shoulder) / sin(45°) / (y_ear − y_shoulder) ]

只用單張影像上耳朵與肩膀的像素座標，靠除以 sin(45°) 補償斜視角的壓縮，所以它
只在相機正好架在 45° 時無偏。這裡照原樣實作，不「改良」它。

## 用哪一側、正負號

- **近側**：右耳與右肩，右側不能用才用左耳與左肩。這與 `theta_ca_on_available_side`
  先右後左的順序一致。
- **前傾為正，與 3D 的 θ_CA 同號。** 從受試者右側看過去，人面向影像左方（−x），
  從左側看則面向 +x。所以前方的水平位移是 `forward_sign·(x_ear − x_shoulder)`，
  右側 forward_sign = −1、左側 = +1。
- 影像的 y 向下，耳朵在肩膀上方時 y_ear < y_shoulder，所以向上的高度是
  `y_shoulder − y_ear`（前作原式分母的順序在 y 向下的座標下會讓坐正時變負，
  這裡取成向上為正，得到的是同一個幾何量）。
- 用 `atan2(前方位移, 向上高度)`，不用 `atan(前方位移 / 向上高度)`：兩者在耳朵高過
  肩膀時完全相同，atan2 在高度為 0 或為負時不除以零，也不會翻號。

## 為什麼會隨方位角偏差（推導，也寫在文件 03）

方位角 φ 的定義與 `pipeline.camera_azimuth_deg` 相同：**0° 是正面、90° 是正側面**。
弱透視、相機水平、忽略畸變與頭部左右偏移時，矢狀面內長度 L、前傾 θ 的
肩→耳向量，前方分量 L·sinθ 在影像上被壓縮成 L·sinθ·sinφ，向上分量 L·cosθ 不變。
前作除以 sin45° 之後：

    tan θ_est = (sin φ / sin 45°) · tan θ_true

φ = 45° 無偏；φ < 45° 低估；φ > 45° 高估。相機有俯仰角時，向上分量與前方分量都會
混進深度方向，**即使 φ = 45° 也不再無偏**，那是另一項系統誤差，這裡不建模。

這個模組不碰 cv2、終端機與 argparse。
"""
from __future__ import annotations

import math

_SIN_45 = math.sin(math.radians(45.0))

# 從受試者哪一側看過去，人在影像裡面向哪個方向。見模組說明。
FORWARD_SIGN = {"right": -1.0, "left": 1.0}


def _finite(*values: float | None) -> bool:
    return all(v is not None and math.isfinite(v) for v in values)


def theta_ca_2d(
    ear_xy: tuple[float, float] | None,
    shoulder_xy: tuple[float, float] | None,
    side: str,
) -> float | None:
    """前作的 2D θ_CA（度），前傾為正。任何一個座標缺值就回傳 None，不回傳 0。

    ear_xy、shoulder_xy 是同一側的耳與肩在同一張影像上的像素座標 (x, y)。
    """
    if side not in FORWARD_SIGN:
        raise ValueError(f"side 要是 'right' 或 'left'，收到 {side!r}")
    if ear_xy is None or shoulder_xy is None:
        return None
    (xe, ye), (xs, ys) = ear_xy, shoulder_xy
    if not _finite(xe, ye, xs, ys):
        return None
    forward = FORWARD_SIGN[side] * (xe - xs) / _SIN_45
    up = ys - ye
    return math.degrees(math.atan2(forward, up))


def predicted_2d_bias(theta_true_deg: float, azimuth_deg: float) -> float:
    """弱透視、相機水平時，前作 2D 公式比真值多（正）或少（負）幾度。

    tan θ_est = (sin φ / sin 45°) · tan θ_true；回傳 θ_est − θ_true。
    只在 |θ_true| < 90° 成立。
    """
    ratio = math.sin(math.radians(azimuth_deg)) / _SIN_45
    estimate = math.degrees(math.atan(ratio * math.tan(math.radians(theta_true_deg))))
    return estimate - theta_true_deg


def _cell(row: dict, column: str) -> float | None:
    """CSV 的一格。缺欄、空字串、不是數字都是 None。"""
    text = row.get(column)
    if text is None or str(text).strip() == "":
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def theta_ca_2d_from_row(row: dict) -> tuple[float, str] | None:
    """從逐幀 CSV 的一列（左眼影像座標）算前作的 2D θ_CA，回傳 (角度, 側別)。

    CSV 沒記 θ_CA 用的是哪一側，所以由左眼影像上哪一側的耳與肩四個座標齊全來判斷，
    右側優先，與 3D 的挑法一致。舊檔沒有像素欄、或兩側都不齊就回傳 None。
    """
    for side in ("right", "left"):
        values = [_cell(row, f"{side}_{part}_l{axis}_px")
                  for part in ("ear", "shoulder") for axis in ("x", "y")]
        if None in values:
            continue
        ear_x, ear_y, shoulder_x, shoulder_y = values
        angle = theta_ca_2d((ear_x, ear_y), (shoulder_x, shoulder_y), side)
        if angle is not None:
            return angle, side
    return None


# ---- 前作的 2D 肩線傾斜角（說明書表4，側傾情境）------------------------------
#
# 前作（cam_front.py）用正面相機、雙肩像素連線算傾斜角：
#     arctan[(y_L − y_R) / (x_L − x_R)]，再取絕對值、限制在 0～90°。
# 這裡照同一個式子，但**保留正負號、不 clamp**：
#   - 正負號與 3D 的 θ_sym 相同（右肩較高為正），量角器的實測值也帶號，誤差才能相減；
#   - 端正坐本身就有幾度偏差，先取絕對值再扣端正坐會把負的側傾算成錯的數字。
# 前作「只看大小」的用法，由呼叫端另外取絕對值。
#
# 相機斜角架設（方位角 φ，0° 是正面）時，雙肩連線的水平分量被壓縮成 cosφ 倍而垂直
# 分量不變，所以 tan θ_est = tan θ_true / cosφ（弱透視、相機水平）。與
# `posture_angles.theta_sym` 註解裡說的 1/cos(方位角) 是同一件事。


def theta_sym_2d(
    left_xy: tuple[float, float] | None,
    right_xy: tuple[float, float] | None,
) -> float | None:
    """前作的 2D 肩線傾斜角（度），右肩較高為正。缺值回傳 None，不回傳 0。

    left_xy、right_xy 是受試者左肩、右肩在同一張影像上的像素座標。影像 y 向下，
    右肩較高時 y_右 < y_左，所以 y_左 − y_右 > 0。
    """
    if left_xy is None or right_xy is None:
        return None
    (xl, yl), (xr, yr) = left_xy, right_xy
    if not _finite(xl, yl, xr, yr):
        return None
    return math.degrees(math.atan2(yl - yr, xl - xr))


def theta_sym_2d_from_row(row: dict) -> float | None:
    """從逐幀 CSV 的一列（左眼影像座標）算 2D 肩線傾斜角。缺欄或缺值是 None。"""
    values = [_cell(row, f"{side}_shoulder_l{axis}_px")
              for side in ("left", "right") for axis in ("x", "y")]
    if None in values:
        return None
    return theta_sym_2d((values[0], values[1]), (values[2], values[3]))


def predicted_sym_2d(theta_true_deg: float, azimuth_deg: float) -> float:
    """弱透視、相機水平時，2D 肩線角的預期值：tan θ_est = tan θ_true / cosφ。

    只在 |θ_true| < 90° 且 φ < 90° 成立。
    """
    ratio = 1.0 / math.cos(math.radians(azimuth_deg))
    return math.degrees(math.atan(ratio * math.tan(math.radians(theta_true_deg))))
