"""把關鍵點偵測、三角測量與角度計算串成一次完整量測。

這一層不碰相機也不碰模型，只吃左右兩組 `PersonKeypoints`，所以可以用
假資料完整測試。真實硬體的部分留在 repo 根目錄的 `posture.py`。

除了角度本身，還會一併回報幾項在真實資料上才看得出來的品質指標：
校正後的垂直視差、深度範圍、左右共同偵測到幾個關鍵點。
標定不準或左右配對錯誤時，三角測量不會報錯，只會安靜地給出看似合理的座標，
所以這些指標是唯一的線索。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from calibration.stereo_calibration import StereoCalibrationResult
from pose.keypoints import PersonKeypoints
from pose.topology import COCO18_KEYPOINT_NAMES

from .keypoints3d import PersonKeypoints3D, triangulate_person_keypoints
from .posture_angles import theta_ca, theta_sym
from .triangulation import rectified_vertical_disparity

# 桌前坐姿的合理深度範圍。超出這個範圍多半是配對錯誤或標定尺度不對，
# 而不是受試者真的坐到三公尺外。
_PLAUSIBLE_DEPTH_MM = (200.0, 3000.0)
# 校正後對極線應該是水平的，左右y座標差超過這個值代表標定或配對有問題。
_MAX_VERTICAL_DISPARITY_PX = 3.0


@dataclass
class PostureMeasurement:
    keypoints_3d: PersonKeypoints3D
    vertical_disparity_px: np.ndarray  # shape(18,)，NaN代表該點沒有左右對應
    shared_count: int
    theta_ca_deg: float | None = None
    theta_sym_deg: float | None = None
    angle_errors: list[str] = field(default_factory=list)

    @property
    def depth_range_mm(self) -> tuple[float, float] | None:
        z = self.keypoints_3d.points[:, 2]
        z = z[np.isfinite(z)]
        return (float(z.min()), float(z.max())) if z.size else None

    @property
    def max_abs_vertical_disparity_px(self) -> float | None:
        d = np.abs(self.vertical_disparity_px)
        d = d[np.isfinite(d)]
        return float(d.max()) if d.size else None


def measure_posture(
    calib: StereoCalibrationResult,
    left: PersonKeypoints,
    right: PersonKeypoints,
    min_confidence: float = 0.0,
) -> PostureMeasurement:
    """三角測量並計算坐姿角度。角度算不出來時記錄原因，不中斷整次量測。"""
    keypoints_3d = triangulate_person_keypoints(calib, left, right, min_confidence)
    disparity = rectified_vertical_disparity(calib, left.points, right.points)
    shared = int(np.isfinite(disparity).sum())

    measurement = PostureMeasurement(
        keypoints_3d=keypoints_3d,
        vertical_disparity_px=disparity,
        shared_count=shared,
    )

    # 缺關鍵點或資料退化都會拋例外。一個角度算不出來不該影響另一個，
    # 所以分開接，並把原因留下來給使用者看。
    for name, fn in (("theta_ca", theta_ca), ("theta_sym", theta_sym)):
        try:
            value = fn(keypoints_3d)
        except (ValueError, KeyError) as exc:
            measurement.angle_errors.append(f"{name}: {exc}")
            continue
        if name == "theta_ca":
            measurement.theta_ca_deg = value
        else:
            measurement.theta_sym_deg = value
    return measurement


def format_measurement(
    measurement: PostureMeasurement,
    left: PersonKeypoints,
    right: PersonKeypoints,
    show_all_keypoints: bool = False,
) -> str:
    """把一次量測整理成可讀的診斷輸出。"""
    left_n = int((~np.isnan(left.points).any(axis=1)).sum())
    right_n = int((~np.isnan(right.points).any(axis=1)).sum())

    lines = [
        f"偵測到的關鍵點   左眼 {left_n}/18   右眼 {right_n}/18   左右共同 {measurement.shared_count}",
        "",
    ]

    interesting = ("right_ear", "right_shoulder", "left_shoulder", "left_ear", "neck", "nose")
    names = COCO18_KEYPOINT_NAMES if show_all_keypoints else interesting

    lines.append(f"{'關鍵點':<16} {'左眼像素':>17} {'右眼像素':>17} {'3D座標 (mm)':>28} {'Δy':>7}")
    for name in names:
        i = COCO18_KEYPOINT_NAMES.index(name)
        lp, rp = left.points[i], right.points[i]
        p3 = measurement.keypoints_3d.points[i]
        dy = measurement.vertical_disparity_px[i]
        if np.isnan(lp).any() and np.isnan(rp).any():
            lines.append(f"{name:<16} {'未偵測到':>17}")
            continue
        lt = "        —        " if np.isnan(lp).any() else f"({lp[0]:7.1f},{lp[1]:7.1f})"
        rt = "        —        " if np.isnan(rp).any() else f"({rp[0]:7.1f},{rp[1]:7.1f})"
        p3t = "            —             " if np.isnan(p3).any() else \
            f"[{p3[0]:8.1f},{p3[1]:8.1f},{p3[2]:8.1f}]"
        dyt = "     —" if not np.isfinite(dy) else f"{dy:6.2f}"
        lines.append(f"{name:<16} {lt:>17} {rt:>17} {p3t:>28} {dyt:>7}")

    lines.append("")
    depth = measurement.depth_range_mm
    lines.append(f"深度範圍         {'—' if depth is None else f'{depth[0]:.0f} ~ {depth[1]:.0f} mm'}")
    worst = measurement.max_abs_vertical_disparity_px
    lines.append(f"最大垂直視差     {'—' if worst is None else f'{worst:.2f} px'}")
    lines.append("")

    ca = measurement.theta_ca_deg
    sym = measurement.theta_sym_deg
    lines.append(f"θ_CA  頸椎前傾   {'算不出來' if ca is None else f'{ca:+.2f}°'}")
    lines.append(f"θ_sym 肩膀水平   {'算不出來' if sym is None else f'{sym:+.2f}°'}")
    for err in measurement.angle_errors:
        lines.append(f"  {err}")

    warnings = plausibility_warnings(measurement)
    if warnings:
        lines.append("")
        lines.append("需要注意：")
        lines.extend(f"  {w}" for w in warnings)
    return "\n".join(lines)


def plausibility_warnings(measurement: PostureMeasurement) -> list[str]:
    """回報真實資料上看得出來的異常。這些都不會讓程式出錯，只會讓結果悄悄變錯。"""
    warnings: list[str] = []

    worst = measurement.max_abs_vertical_disparity_px
    if worst is not None and worst > _MAX_VERTICAL_DISPARITY_PX:
        warnings.append(
            f"校正後的垂直視差最大 {worst:.1f} px（應 <{_MAX_VERTICAL_DISPARITY_PX}）。"
            f"對極線校正後左右的 y 應該幾乎相同，差這麼多代表標定不夠準、"
            f"或左右眼配對到不同的人體部位"
        )

    depth = measurement.depth_range_mm
    if depth is not None:
        low, high = _PLAUSIBLE_DEPTH_MM
        if depth[0] < low or depth[1] > high:
            warnings.append(
                f"深度範圍 {depth[0]:.0f}~{depth[1]:.0f} mm 超出桌前坐姿的合理區間"
                f"（{low:.0f}~{high:.0f} mm）。多半是左右配對錯誤，"
                f"或標定時的 --square-size-mm 填錯導致整體尺度不對"
            )

    if measurement.shared_count < 4:
        warnings.append(
            f"左右只有 {measurement.shared_count} 個共同關鍵點，樣本太少，"
            f"算出來的角度參考價值有限"
        )
    return warnings
