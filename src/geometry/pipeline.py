"""把關鍵點偵測、三角測量與角度計算串成一次完整量測。

這一層不碰相機也不碰模型，只吃左右兩組 `PersonKeypoints`，所以可以用
假資料完整測試。真實硬體的部分留在 repo 根目錄的 `posture.py`。

除了角度本身，還會一併回報幾項在真實資料上才看得出來的品質指標：
校正後的垂直視差、深度範圍、左右共同偵測到幾個關鍵點。
標定不準或左右配對錯誤時，三角測量不會報錯，只會安靜地給出看似合理的座標，
所以這些指標是唯一的線索。
"""
from __future__ import annotations

import unicodedata
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
# 關鍵點定位的殘餘誤差。次像素精修把量化壓到0.2px以下，剩下的是模型本身的抖動，
# 1px是保守估計，用來換算角度精度。
_KEYPOINT_NOISE_PX = 1.0
# 耳朵到肩膀的垂直距離，用來把深度誤差換算成角度誤差。
_EAR_SHOULDER_MM = 170.0
# 前作的頸椎前傾判定門檻。
# 單幀誤差在這個硬體上本來就與門檻同量級，所以時間序列平均是必要的而非加分項：
# 平均N幀可以把雜訊降到 1/√N。警告線畫在「單幀誤差超過整個門檻」，
# 因為那代表連平均都救不太回來，該做的是坐近一點。
_THETA_CA_THRESHOLD_DEG = 10.0


@dataclass
class PostureMeasurement:
    keypoints_3d: PersonKeypoints3D
    vertical_disparity_px: np.ndarray  # shape(18,)，NaN代表該點沒有左右對應
    shared_count: int
    theta_ca_deg: float | None = None
    theta_sym_deg: float | None = None
    theta_ca_precision_deg: float | None = None
    reference_depth_mm: float | None = None
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


def reference_depth_mm(keypoints_3d: PersonKeypoints3D) -> float | None:
    """換算角度精度時要用的代表深度。

    優先取耳朵與肩膀，因為要換算的就是這兩點之間的深度差。
    兩者都沒有時退回所有關鍵點深度的中位數。不用最大與最小值的中點，
    是因為只要有一個關鍵點三角測量失敗，那個中點就會被整個拉走——
    手腕配對錯誤跑到三公尺外，換算出來的角度誤差就跟著大一倍。
    """
    depths = [
        float(p[2])
        for side in ("right", "left")
        for part in ("ear", "shoulder")
        if (p := keypoints_3d.get(f"{side}_{part}")) is not None
    ]
    if depths:
        return float(np.mean(depths))
    z = keypoints_3d.points[:, 2]
    z = z[np.isfinite(z)]
    return float(np.median(z)) if z.size else None


def estimate_theta_ca_precision_deg(
    calib: StereoCalibrationResult,
    depth_mm: float,
    keypoint_noise_px: float = _KEYPOINT_NOISE_PX,
) -> float:
    """在這個距離下，單幀 θ_CA 的預期誤差（度）。

    θ_CA 量的是耳朵與肩膀的深度差，而深度來自視差：Z = fx·B/d。
    微分後 σ_Z = Z²·σ_d/(fx·B)——**誤差隨距離平方成長**。
    兩個關鍵點各有一份誤差，所以深度差的誤差是 √2 倍。

    這個數字決定了整個系統在多遠的距離還能用。實機參數（fx 568、基線 60mm）下，
    500mm 時單幀誤差約 3.5°，1900mm 時是 50°——後者遠超過 10° 的判定門檻，
    算出來的角度沒有意義。座位距離是使用者唯一能立刻改變的因素。
    """
    fx = float(calib.P1[0, 0])
    baseline = calib.baseline_mm
    if fx <= 0 or baseline <= 0 or depth_mm <= 0:
        return float("inf")
    sigma_z = depth_mm**2 * keypoint_noise_px / (fx * baseline)
    return float(np.degrees(np.sqrt(2) * sigma_z / _EAR_SHOULDER_MM))


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

    measurement.reference_depth_mm = reference_depth_mm(keypoints_3d)
    if measurement.reference_depth_mm is not None:
        measurement.theta_ca_precision_deg = estimate_theta_ca_precision_deg(
            calib, measurement.reference_depth_mm
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


def _display_width(text: str) -> int:
    """終端機顯示寬度。中日韓字元佔兩欄，但 len() 只算一個。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _cell(text: str, width: int, align: str = "left") -> str:
    """照顯示寬度補空白。直接用 f-string 的 :<16 會讓中文欄位短掉一半。"""
    pad = " " * max(0, width - _display_width(text))
    return text + pad if align == "left" else pad + text


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

    lines.append(
        f"{_cell('關鍵點', 16)} {_cell('左眼像素', 17, 'right')}"
        f" {_cell('右眼像素', 17, 'right')} {_cell('3D座標 (mm)', 28, 'right')}"
        f" {_cell('Δy', 7, 'right')}"
    )
    for name in names:
        i = COCO18_KEYPOINT_NAMES.index(name)
        lp, rp = left.points[i], right.points[i]
        p3 = measurement.keypoints_3d.points[i]
        dy = measurement.vertical_disparity_px[i]
        if np.isnan(lp).any() and np.isnan(rp).any():
            lines.append(f"{_cell(name, 16)} {_cell('未偵測到', 17, 'right')}")
            continue
        lt = "—" if np.isnan(lp).any() else f"({lp[0]:7.1f},{lp[1]:7.1f})"
        rt = "—" if np.isnan(rp).any() else f"({rp[0]:7.1f},{rp[1]:7.1f})"
        p3t = "—" if np.isnan(p3).any() else f"[{p3[0]:8.1f},{p3[1]:8.1f},{p3[2]:8.1f}]"
        dyt = "—" if not np.isfinite(dy) else f"{dy:6.2f}"
        lines.append(
            f"{_cell(name, 16)} {_cell(lt, 17, 'right')} {_cell(rt, 17, 'right')}"
            f" {_cell(p3t, 28, 'right')} {_cell(dyt, 7, 'right')}"
        )

    lines.append("")
    depth = measurement.depth_range_mm
    lines.append(
        _cell("深度範圍", 17)
        + ("—" if depth is None else f"{depth[0]:.0f} ~ {depth[1]:.0f} mm")
    )
    worst = measurement.max_abs_vertical_disparity_px
    lines.append(
        _cell("最大垂直視差", 17) + ("—" if worst is None else f"{worst:.2f} px")
    )
    if measurement.theta_ca_precision_deg is not None:
        lines.append(
            _cell("θ_CA 單幀誤差", 17)
            + f"±{measurement.theta_ca_precision_deg:.1f}°"
            + f"（判定門檻 {_THETA_CA_THRESHOLD_DEG:.0f}°）"
        )
    lines.append("")

    ca = measurement.theta_ca_deg
    sym = measurement.theta_sym_deg
    lines.append(_cell("θ_CA  頸椎前傾", 17) + ("算不出來" if ca is None else f"{ca:+.2f}°"))
    lines.append(_cell("θ_sym 肩膀水平", 17) + ("算不出來" if sym is None else f"{sym:+.2f}°"))
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

    precision = measurement.theta_ca_precision_deg
    if precision is not None and precision > _THETA_CA_THRESHOLD_DEG:
        distance = measurement.reference_depth_mm or 0.0
        warnings.append(
            f"距離 {distance:.0f} mm，θ_CA 單幀誤差約 ±{precision:.1f}°，"
            f"比 {_THETA_CA_THRESHOLD_DEG:.0f}° 的判定門檻還大，這一幀的角度沒有參考價值。"
            f"深度誤差隨距離的平方成長，坐到 600 mm 左右就能降到 ±5° 以內"
        )

    if measurement.shared_count < 4:
        warnings.append(
            f"左右只有 {measurement.shared_count} 個共同關鍵點，樣本太少，"
            f"算出來的角度參考價值有限"
        )
    return warnings
