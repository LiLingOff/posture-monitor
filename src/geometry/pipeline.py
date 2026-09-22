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
from .posture_angles import anatomical_axes, theta_ca, theta_sym
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
# θ_CA 與 θ_sym 實際用到的四個點。品質指標落在這幾點上才會影響角度。
_ANGLE_KEYPOINTS = ("right_ear", "right_shoulder", "left_shoulder", "left_ear")


@dataclass
class PostureMeasurement:
    keypoints_3d: PersonKeypoints3D
    vertical_disparity_px: np.ndarray  # shape(18,)，NaN代表該點沒有左右對應
    shared_count: int
    theta_ca_deg: float | None = None
    theta_sym_deg: float | None = None
    theta_ca_precision_deg: float | None = None
    reference_depth_mm: float | None = None
    camera_azimuth_deg: float | None = None
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
    azimuth_deg: float = 0.0,
    keypoint_noise_px: float = _KEYPOINT_NOISE_PX,
) -> float:
    """這個距離與這個相機方位下，單幀 θ_CA 的預期誤差（度）。

    θ_CA 量的是耳朵相對肩膀往前伸多少，而「往前」在相機座標系裡落在哪個軸，
    取決於雙目模組架在受試者的哪個方位。兩個軸的精度差了一個數量級：

    - 深度軸：σ_Z = Z²·σ_d/(fx·B)，隨距離平方成長。620mm 下約 16mm。
    - 影像平面（左右）軸：σ_X = Z·σ_px/fx，只隨距離線性成長。620mm 下約 1.1mm。

    兩者的比值就是 Z/B，620mm 配 60mm 基線是 10 倍。所以相機架在正面時
    「往前伸」完全落在最差的深度軸上，架在側面則完全落在最好的影像平面軸上。
    方位角 α 的組合誤差是 √((cos α·σ_Z)² + (sin α·σ_X)²)。

    σ_d 是**視差**的誤差，等於 √2·σ_px：視差是左右兩次像素量測的差。
    外層再一個 √2 是因為耳朵與肩膀各有一份誤差。

    蒙地卡羅（3000 次、1px 高斯雜訊、620mm）對照，誤差在 10% 以內：
    方位角 0° 解析式 ±10.8° 對模擬 ±10.0°，90° 是 ±0.7° 對 ±0.9°。
    """
    fx = float(calib.P1[0, 0])
    baseline = calib.baseline_mm
    if fx <= 0 or baseline <= 0 or depth_mm <= 0:
        return float("inf")

    sigma_disparity = np.sqrt(2.0) * keypoint_noise_px
    sigma_depth = depth_mm**2 * sigma_disparity / (fx * baseline)
    sigma_lateral = depth_mm * keypoint_noise_px / fx

    alpha = np.radians(azimuth_deg)
    sigma_forward = np.hypot(np.cos(alpha) * sigma_depth, np.sin(alpha) * sigma_lateral)
    return float(np.degrees(np.sqrt(2) * sigma_forward / _EAR_SHOULDER_MM))


def camera_azimuth_deg(keypoints_3d: PersonKeypoints3D) -> float:
    """雙目模組相對受試者正面的方位角（0~90度）。0是正面、90是正側面。

    由雙肩連線與相機 X 軸的夾角算出。雙肩取不到時回傳 0，
    也就是當成正面——那是 θ_CA 精度最差的情況，估計誤差時取保守值。
    """
    lateral, _ = anatomical_axes(keypoints_3d)
    if keypoints_3d.get("left_shoulder") is None or keypoints_3d.get("right_shoulder") is None:
        return 0.0
    cos_alpha = abs(float(np.dot(lateral, np.array([1.0, 0.0, 0.0]))))
    return float(np.degrees(np.arccos(np.clip(cos_alpha, 0.0, 1.0))))


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
    measurement.camera_azimuth_deg = camera_azimuth_deg(keypoints_3d)
    if measurement.reference_depth_mm is not None:
        measurement.theta_ca_precision_deg = estimate_theta_ca_precision_deg(
            calib, measurement.reference_depth_mm, measurement.camera_azimuth_deg
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
    # 分開印。整體最大常常落在手腕、腳踝這些角度用不到的點上，
    # 只印一個數字會讓人以為是標定有問題。
    worst_all = _worst_disparity(measurement)
    lines.append(
        _cell("最大垂直視差", 17)
        + ("—" if worst_all is None else f"{worst_all[1]:.2f} px（{worst_all[0]}）")
    )
    worst_angle = _worst_disparity(measurement, _ANGLE_KEYPOINTS)
    lines.append(
        _cell("  角度用到的點", 17)
        + ("—" if worst_angle is None else f"{worst_angle[1]:.2f} px（{worst_angle[0]}）")
    )
    if measurement.camera_azimuth_deg is not None:
        lines.append(
            _cell("相機方位角", 17)
            + f"{measurement.camera_azimuth_deg:.0f}°（0 是正面、90 是正側面）"
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


def _worst_disparity(
    measurement: PostureMeasurement, names: tuple[str, ...] | None = None
) -> tuple[str, float] | None:
    """垂直視差最大的那個關鍵點與它的數值。names 限定只看某幾個點。"""
    candidates = names or COCO18_KEYPOINT_NAMES
    worst: tuple[str, float] | None = None
    for name in candidates:
        d = measurement.vertical_disparity_px[COCO18_KEYPOINT_NAMES.index(name)]
        if not np.isfinite(d):
            continue
        if worst is None or abs(d) > worst[1]:
            worst = (name, abs(float(d)))
    return worst


def _implausible_depth_names(measurement: PostureMeasurement) -> list[str]:
    low, high = _PLAUSIBLE_DEPTH_MM
    bad = []
    for i, name in enumerate(COCO18_KEYPOINT_NAMES):
        z = measurement.keypoints_3d.points[i, 2]
        if np.isfinite(z) and not (low <= z <= high):
            bad.append(f"{name} {z:.0f}mm")
    return bad


def plausibility_warnings(measurement: PostureMeasurement) -> list[str]:
    """回報真實資料上看得出來的異常。這些都不會讓程式出錯，只會讓結果悄悄變錯。

    異常要分成兩類來看。落在角度用到的那四個點上，角度本身就不能信；
    落在手腕、腳踝這些點上，多半是自底向上的關聯把左右兩張影像的同一個肢體
    連到了不同位置，角度不受影響。兩者混在一起報，標定明明已經夠準，
    看到的卻還是一行「垂直視差 38px」，會讓人白白再去重拍一次標定板。
    """
    warnings: list[str] = []

    worst_angle = _worst_disparity(measurement, _ANGLE_KEYPOINTS)
    worst_all = _worst_disparity(measurement)
    if worst_angle is not None and worst_angle[1] > _MAX_VERTICAL_DISPARITY_PX:
        warnings.append(
            f"角度用到的 {worst_angle[0]} 垂直視差 {worst_angle[1]:.1f} px"
            f"（應 <{_MAX_VERTICAL_DISPARITY_PX}）。對極線校正後左右的 y 應該幾乎相同，"
            f"差這麼多代表標定不夠準，或這個點左右配對到了不同位置。角度不可信"
        )
    elif worst_all is not None and worst_all[1] > _MAX_VERTICAL_DISPARITY_PX:
        warnings.append(
            f"{worst_all[0]} 垂直視差 {worst_all[1]:.1f} px（應 <{_MAX_VERTICAL_DISPARITY_PX}），"
            f"這個點左右配對錯了。角度用到的點都在門檻內，所以不影響這一次的角度，"
            f"但它會汙染深度範圍這類整體指標"
        )

    bad_depths = _implausible_depth_names(measurement)
    if bad_depths:
        low, high = _PLAUSIBLE_DEPTH_MM
        listed = "、".join(bad_depths[:4])
        more = f" 等 {len(bad_depths)} 個點" if len(bad_depths) > 4 else ""
        negative = any(d.endswith("mm") and float(d.split()[1][:-2]) < 0 for d in bad_depths)
        reason = (
            "深度為負代表算出來的點在相機後方，只可能是左右配對錯誤"
            if negative
            else "整體偏掉通常是標定時的 --square-size-mm 填錯造成尺度不對"
        )
        warnings.append(
            f"{listed}{more} 的深度超出桌前坐姿的合理區間"
            f"（{low:.0f}~{high:.0f} mm）。{reason}"
        )

    precision = measurement.theta_ca_precision_deg
    if precision is not None and precision > _THETA_CA_THRESHOLD_DEG:
        distance = measurement.reference_depth_mm or 0.0
        azimuth = measurement.camera_azimuth_deg or 0.0
        advice = (
            "深度誤差隨距離的平方成長，坐近一點會有幫助"
            if azimuth > 60.0
            else "把雙目模組往側面移比坐近更有效：正面時「往前伸」完全落在深度軸上，"
            "側面則落在影像平面上，兩者精度差了 Z/B 倍"
        )
        warnings.append(
            f"距離 {distance:.0f} mm、方位角 {azimuth:.0f}°，θ_CA 單幀誤差約 ±{precision:.1f}°，"
            f"比 {_THETA_CA_THRESHOLD_DEG:.0f}° 的判定門檻還大，這一幀的角度沒有參考價值。{advice}"
        )

    if measurement.shared_count < 4:
        warnings.append(
            f"左右只有 {measurement.shared_count} 個共同關鍵點，樣本太少，"
            f"算出來的角度參考價值有限"
        )
    return warnings
