"""把關鍵點偵測、三角測量與角度計算串成一次完整量測。

這一層不碰相機也不碰模型，只吃左右兩組 `PersonKeypoints`，所以可以用
假資料完整測試。真實硬體的部分留在 repo 根目錄的 `posture.py`。

除了角度本身，還會一併回報幾項在真實資料上才看得出來的品質指標：
校正後的垂直視差、深度範圍、左右共同偵測到幾個關鍵點。
標定不準或左右配對錯誤時，三角測量不會報錯，給出的座標數量級也正常，
所以這些指標是唯一的線索。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from calibration.stereo_calibration import StereoCalibrationResult
from pose.keypoints import PersonKeypoints
from pose.topology import COCO18_KEYPOINT_NAMES

from .keypoints3d import PersonKeypoints3D, triangulate_person_keypoints
from .posture_angles import (anatomical_axes, theta_ca_on_available_side,
                             theta_sym)
from .terminal import cell as _cell
from .triangulation import rectified_vertical_disparity, triangulate_points

# 桌前坐姿的合理深度範圍。超出這個範圍多半是配對錯誤或標定尺度不對，
# 而不是受試者真的坐到三公尺外。
_PLAUSIBLE_DEPTH_MM = (200.0, 3000.0)
# 校正後對極線應該是水平的，所以左右的 y 差反映兩件不同的事，門檻也要分開。
#
# 標定品質：標定夠準時，正常幀的角度關鍵點落在 1~2.5px。超過 3px 值得回頭看標定，
# 這是拿來評估一整組設定的，不是拿來篩選單幀的。
_MAX_VERTICAL_DISPARITY_PX = 3.0
# 配對錯誤：左右眼對到不同部位時差的是十幾到數十 px。實測的分布有明顯斷層，
# 正常幀最多 2.5px，確定配錯的是 13.5、26.1、36.5px，中間是空的。
# 門檻取在斷層裡。拿 3px 來擋單幀的話，48 幀會擋掉 43 幀，
# 擋掉的絕大多數只是雜訊——每個座標約 1px，相減後 √2 倍，再加上標定的殘餘偏差。
# 這個值是依實測分布訂的，取得更多資料後應該回頭校準。
_MISPAIRED_DISPARITY_PX = 8.0
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
# 桌前坐姿夠得著的距離。σ_Z 隨距離平方成長，所以這個數字決定誤差的量級：
# 700mm 配 50° 方位是 ±6.3°，1800mm 同一個方位是 ±41°。
_COMFORTABLE_DISTANCE_MM = 700.0
# θ_sym 與解剖平面永遠要雙肩，θ_CA 再加上它實際用到的那一側的耳朵。
# 哪一側是動態的，所以用 angle_keypoints() 查而不是寫死一個常數：
# 把兩側的耳朵都算進去的話，遠側那隻被遮住或配對錯誤就會誤報成角度不可信。
_SHOULDER_KEYPOINTS = ("right_shoulder", "left_shoulder")
# 關鍵點落在離畫面邊緣這麼近的位置時，多半不是偵測結果而是被邊界夾住的值：
# 真正的位置在畫面外，模型只能回報它能表示的最接近的點。
# 2026-09-24 實機遇到頭頂出界，鼻子與雙眼的 y 都是 0.0，換算出來的3D座標
# 讓眼睛落在頸部上方 400mm。數值本身沒有異常，靠範圍檢查抓不到。
_FRAME_EDGE_MARGIN_PX = 2.0


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
    theta_ca_side: str | None = None  # θ_CA 實際用了哪一側的耳朵
    edge_keypoints: list[str] = field(default_factory=list)
    angle_errors: list[str] = field(default_factory=list)

    @property
    def depth_range_mm(self) -> tuple[float, float] | None:
        z = self.keypoints_3d.points[:, 2]
        z = z[np.isfinite(z)]
        return (float(z.min()), float(z.max())) if z.size else None

    @property
    def max_abs_vertical_disparity_px(self) -> float | None:
        """所有共同關鍵點裡最差的一個。

        這個值經常比 _MISPAIRED_DISPARITY_PX 大很多，不代表門檻沒生效：
        手腕、腳踝配錯不影響角度，門檻只套在 angle_max_vertical_disparity_px 上。
        """
        d = np.abs(self.vertical_disparity_px)
        d = d[np.isfinite(d)]
        return float(d.max()) if d.size else None

    @property
    def angle_max_vertical_disparity_px(self) -> float | None:
        """算角度實際用到的那幾點裡最差的一個。判定看的是這個。"""
        worst = _worst_disparity(self, angle_keypoints(self.theta_ca_side))
        return None if worst is None else worst[1]


def reference_depth_mm(keypoints_3d: PersonKeypoints3D) -> float | None:
    """換算角度精度時要用的代表深度。

    優先取耳朵與肩膀，因為要換算的就是這兩點之間的深度差。
    兩者都沒有時退回所有關鍵點深度的中位數。不用最大與最小值的中點，
    是因為只要有一個關鍵點三角測量失敗，那個中點就會被整個拉走。
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

    與蒙地卡羅對照過（3000 次、1px 高斯雜訊、620mm、耳肩距 170mm，
    也就是這個函式用的同一組參數），誤差在 20% 以內：

        方位角      0°      30°     45°     60°     75°     90°
        解析式    ±7.60°  ±6.59°  ±5.39°  ±3.83°  ±2.03°  ±0.52°
        模擬      ±7.14°  ±6.98°  ±5.96°  ±4.29°  ±2.16°  ±0.63°
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


def precision_advice(distance_mm: float | None, azimuth_deg: float | None) -> str:
    """誤差太大時，該動距離還是動方位角。

    兩個軸的貢獻是 cos(α)·σ_Z 與 sin(α)·σ_X，而 σ_Z/σ_X = Z/B，在這組硬體上
    是二十到三十倍。所以除非方位角很接近 90°（那個角度遠側肩膀早被擋住），
    深度那一項都是壓倒性的，而它隨距離**平方**成長。
    1796mm 配 50° 方位量到的 ±41°，推到 75° 只降到 ±17°，
    但坐到 700mm 就是 ±6.3°。

    先前這段建議寫死成「把模組往側面移」，在已經 50° 的情況下指錯了方向。
    """
    if distance_mm is None or distance_mm <= 0:
        return "距離量不到，先確認受試者完整在畫面內"
    if distance_mm > _COMFORTABLE_DISTANCE_MM * 1.3:
        factor = (distance_mm / _COMFORTABLE_DISTANCE_MM) ** 2
        return f"坐到 {_COMFORTABLE_DISTANCE_MM:.0f}mm 可降到約 1/{factor:.1f}（誤差隨距離平方成長）"
    if azimuth_deg is None:
        return "方位角量不到，先讓兩邊肩膀都進畫面"
    if azimuth_deg < 60.0:
        return f"距離夠近了。方位角 {azimuth_deg:.0f}°，往側面移到 60~75° 可再降一半以上"
    return "距離與方位角都接近硬體極限，只剩拉長平均視窗"


def camera_azimuth_deg(keypoints_3d: PersonKeypoints3D) -> float | None:
    """雙目模組相對受試者正面的方位角（0~90度）。0是正面、90是正側面。

    由雙肩連線與相機 X 軸的夾角算出，所以雙肩取不到就量不到，回傳 None。
    不回傳 0：那會讓退回值長得跟「量到 0 度，模組正對著受試者」一模一樣，
    而程式會據此建議把模組往側面移。建議一個根本沒量到的東西比不建議更糟。
    """
    if keypoints_3d.get("left_shoulder") is None or keypoints_3d.get("right_shoulder") is None:
        return None
    # 取絕對值是因為左右兩側對精度的影響相同。
    lateral, _ = anatomical_axes(keypoints_3d)
    cos_alpha = abs(float(np.dot(lateral, np.array([1.0, 0.0, 0.0]))))
    return float(np.degrees(np.arccos(np.clip(cos_alpha, 0.0, 1.0))))


def angle_keypoints(side: str | None = None) -> tuple[str, ...]:
    """這一幀的角度實際用到哪些關鍵點。side 是 θ_CA 取的那一側。"""
    return _SHOULDER_KEYPOINTS if side is None else _SHOULDER_KEYPOINTS + (f"{side}_ear",)


def keypoints_at_frame_edge(
    left: PersonKeypoints,
    right: PersonKeypoints,
    image_size: tuple[int, int],
    margin_px: float = _FRAME_EDGE_MARGIN_PX,
) -> list[str]:
    """哪些關鍵點貼在畫面邊緣上，也就是它的真實位置很可能在畫面外。"""
    width, height = image_size
    found: list[str] = []
    for i, name in enumerate(COCO18_KEYPOINT_NAMES):
        for points in (left.points, right.points):
            x, y = points[i]
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            if (x <= margin_px or x >= width - 1 - margin_px
                    or y <= margin_px or y >= height - 1 - margin_px):
                found.append(name)
                break
    return found


@dataclass(frozen=True)
class PersonMatch:
    """左右兩眼各自的偵測結果裡，被判定為同一個人的那一對。"""

    left: PersonKeypoints
    right: PersonKeypoints
    left_count: int
    right_count: int
    median_vertical_disparity_px: float
    distance_mm: float
    rejected_pairs: int
    rejected_farther: int = 0

    @property
    def was_ambiguous(self) -> bool:
        """有超過一種配對可以選，代表至少一眼偵測到不只一個人。"""
        return self.rejected_pairs > 0


def match_person_pair(
    calib: StereoCalibrationResult,
    left_detections: list[PersonKeypoints],
    right_detections: list[PersonKeypoints],
) -> PersonMatch:
    """從左右兩眼的偵測結果裡挑出同一個人。

    不能兩邊各自取第一個。自底向上的模型是把散落的關鍵點組裝成人，
    回傳的順序取決於組裝過程，兩張影像不保證一致。畫面裡只要多一個誤判
    （椅背、反光、另一個人的一部分），左眼的第一個和右眼的第一個就可能
    是不同的對象，而三角測量不會因此報錯，只會給出一個深度離譜的結果。
    2026-09-24 實機連續 20 幀都算出深度 100mm，那需要 341px 的視差，
    而單眼畫面才 1280px 寬。

    兩道篩選，處理的是兩件不同的事。

    第一道是校正後的垂直視差：stereoRectify 的目的就是讓對應點的 y 幾乎相同，
    所以真正成對的那組會對齊，錯配的沒有理由對齊。取中位數而非平均，
    是因為個別關鍵點配錯不該推翻整個人的配對。

    第二道是距離。畫面裡有兩個人時，兩個人各自都能配得很齊，垂直視差分不出
    該追哪一個，背景那位甚至可能對得更好。2026-09-24 實機就是這樣：受試者坐在
    700mm 附近，量出來卻是 1384mm 與 1796mm，那是後面經過的人。
    桌前坐姿監測的對象永遠是最靠近相機的那位，所以在對得齊的組合裡取最近的。
    """
    if not left_detections or not right_detections:
        raise ValueError(
            f"左眼偵測到 {len(left_detections)} 個人、右眼 {len(right_detections)} 個，"
            f"至少要兩邊各一個才能配對"
        )

    candidates: list[tuple[float, float, PersonKeypoints, PersonKeypoints]] = []
    for left in left_detections:
        for right in right_detections:
            disparity = rectified_vertical_disparity(calib, left.points, right.points)
            finite = np.abs(disparity[np.isfinite(disparity)])
            if finite.size == 0:
                continue
            depths = triangulate_points(calib, left.points, right.points)[:, 2]
            depths = depths[np.isfinite(depths)]
            distance = float(np.median(depths)) if depths.size else float("inf")
            candidates.append((float(np.median(finite)), distance, left, right))

    if not candidates:
        raise ValueError("左右兩眼沒有任何共同偵測到的關鍵點，無法配對")

    # 第一道：垂直視差。擋掉左眼的 A 配到右眼的 B，那種組合在對極線上對不齊。
    consistent = [c for c in candidates if c[0] <= _MISPAIRED_DISPARITY_PX]
    # 第二道：距離。兩個人各自都能配得很齊，視差分不出該追哪一個，
    # 而背景那個人甚至可能對得更好。桌前坐姿監測的對象是最靠近相機的那位。
    chosen = min(consistent, key=lambda c: c[1]) if consistent else min(candidates)
    farther = sum(1 for c in consistent if c[1] > chosen[1])

    return PersonMatch(
        left=chosen[2],
        right=chosen[3],
        left_count=len(left_detections),
        right_count=len(right_detections),
        median_vertical_disparity_px=chosen[0],
        distance_mm=chosen[1],
        rejected_pairs=len(candidates) - 1,
        rejected_farther=farther,
    )


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

    measurement.edge_keypoints = keypoints_at_frame_edge(left, right, calib.image_size)
    measurement.reference_depth_mm = reference_depth_mm(keypoints_3d)
    measurement.camera_azimuth_deg = camera_azimuth_deg(keypoints_3d)
    if measurement.reference_depth_mm is not None:
        # 方位角量不到時用 0，那是 θ_CA 精度最差的情況，估計誤差取保守值。
        measurement.theta_ca_precision_deg = estimate_theta_ca_precision_deg(
            calib, measurement.reference_depth_mm, measurement.camera_azimuth_deg or 0.0
        )

    # 缺關鍵點或資料退化都會拋例外。一個角度算不出來不該影響另一個，
    # 所以分開接，並把原因留下來給使用者看。
    try:
        measurement.theta_ca_deg, measurement.theta_ca_side = (
            theta_ca_on_available_side(keypoints_3d)
        )
    except (ValueError, KeyError) as exc:
        measurement.angle_errors.append(f"theta_ca: {exc}")
    try:
        measurement.theta_sym_deg = theta_sym(keypoints_3d)
    except (ValueError, KeyError) as exc:
        measurement.angle_errors.append(f"theta_sym: {exc}")
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
        # 貼邊的點要標出來，否則它看起來跟一個正常的座標沒有兩樣
        mark = "  貼邊" if name in measurement.edge_keypoints else ""
        lines.append(
            f"{_cell(name, 16)} {_cell(lt, 17, 'right')} {_cell(rt, 17, 'right')}"
            f" {_cell(p3t, 28, 'right')} {_cell(dyt, 7, 'right')}{mark}"
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
    worst_angle = _worst_disparity(measurement, angle_keypoints(measurement.theta_ca_side))
    lines.append(
        _cell("  角度用到的點", 17)
        + ("—" if worst_angle is None else f"{worst_angle[1]:.2f} px（{worst_angle[0]}）")
    )
    lines.append(
        _cell("相機方位角", 17)
        + (f"{measurement.camera_azimuth_deg:.0f}°（0 是正面、90 是正側面）"
           if measurement.camera_azimuth_deg is not None
           else "量不到（雙肩沒有同時偵測到）")
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
    side = {"right": "右側", "left": "左側"}.get(measurement.theta_ca_side or "", "")
    lines.append(
        _cell("θ_CA  頸椎前傾", 17)
        + ("算不出來" if ca is None else f"{ca:+.2f}°")
        + (f"   （用{side}耳朵與肩膀）" if side else "")
    )
    lines.append(_cell("θ_sym 肩膀水平", 17) + ("算不出來" if sym is None else f"{sym:+.2f}°"))
    for err in measurement.angle_errors:
        lines.append(f"  {err}")

    warnings = plausibility_warnings(measurement)
    if warnings:
        lines.append("")
        lines.append("需要注意：")
        lines.extend(f"  {w}" for w in warnings)
    return "\n".join(lines)


def unusable_reason(measurement: PostureMeasurement) -> str | None:
    """這一幀能不能拿來用；不能的話回傳一句話說明原因，可以就回傳 None。

    實機上約每十幀就有一兩幀是偵測失誤，而失誤的結果不會是明顯的錯誤，
    是一組看起來正常的數字。2026-09-23 那次錄到的例子：深度 −291mm
    （點在相機後方）、方位角 89°（模組整場沒動過）、距離 74mm。

    最後那個最危險：誤差公式只看距離，距離愈近算出來的誤差愈小，
    所以那一幀印出的是「誤差 ±0.1°」：資料最壞的時候，誤差數字最好看。
    這種幀一定要在進入平均之前擋掉，靠數值大小是擋不住的。
    """
    depth = measurement.reference_depth_mm
    low, high = _PLAUSIBLE_DEPTH_MM
    if depth is None:
        return "沒有可用的耳朵或肩膀深度"
    if depth < 0:
        # 負深度代表視差反號，也就是同一個部位在右眼的 x 比左眼大。
        # 幾何上不可能，一定是左右配對接錯，不是受試者坐在奇怪的位置。
        return f"深度 {depth:.0f}mm 是負的，左右配對接反了"
    if not (low <= depth <= high):
        return f"深度 {depth:.0f}mm 落在桌前坐姿的合理範圍外（{low:.0f}~{high:.0f}mm）"

    at_edge = [n for n in measurement.edge_keypoints if n in angle_keypoints(measurement.theta_ca_side)]
    if at_edge:
        return f"角度用到的 {'、'.join(at_edge)} 貼在畫面邊緣，真實位置在畫面外"

    worst = _worst_disparity(measurement, angle_keypoints(measurement.theta_ca_side))
    if worst is not None and worst[1] > _MISPAIRED_DISPARITY_PX:
        return f"{worst[0]} 的垂直視差 {worst[1]:.1f}px，左右配對錯了"

    if measurement.shared_count < 4:
        return f"左右只有 {measurement.shared_count} 個共同關鍵點"
    return None


def _worst_disparity(
    measurement: PostureMeasurement, names: tuple[str, ...] | None = None
) -> tuple[str, float] | None:
    """垂直視差最大的那個關鍵點與它的數值。names 限定只看某幾個點。"""
    candidates = COCO18_KEYPOINT_NAMES if names is None else names
    worst: tuple[str, float] | None = None
    for name in candidates:
        d = measurement.vertical_disparity_px[COCO18_KEYPOINT_NAMES.index(name)]
        if not np.isfinite(d):
            continue
        if worst is None or abs(d) > worst[1]:
            worst = (name, abs(float(d)))
    return worst


def _implausible_depths(measurement: PostureMeasurement) -> list[tuple[str, float]]:
    """深度落在合理區間外的關鍵點與它的深度值。"""
    low, high = _PLAUSIBLE_DEPTH_MM
    bad: list[tuple[str, float]] = []
    for i, name in enumerate(COCO18_KEYPOINT_NAMES):
        z = float(measurement.keypoints_3d.points[i, 2])
        if np.isfinite(z) and not (low <= z <= high):
            bad.append((name, z))
    return bad


def plausibility_warnings(measurement: PostureMeasurement) -> list[str]:
    """回報真實資料上看得出來的異常。這些都不會讓程式出錯，只會讓結果變錯。

    異常要分成兩類來看。落在角度用到的那四個點上，角度本身就不能信；
    落在手腕、腳踝這些點上，多半是自底向上的關聯把左右兩張影像的同一個肢體
    連到了不同位置，角度不受影響。兩者混在一起報，標定明明已經夠準，
    看到的卻還是一行「垂直視差 38px」，會讓人白白再去重拍一次標定板。
    """
    warnings: list[str] = []

    worst_angle = _worst_disparity(measurement, angle_keypoints(measurement.theta_ca_side))
    worst_all = _worst_disparity(measurement)
    if worst_angle is not None and worst_angle[1] > _MISPAIRED_DISPARITY_PX:
        warnings.append(
            f"角度用到的 {worst_angle[0]} 垂直視差 {worst_angle[1]:.1f}px，"
            f"左右配對錯了，這一幀的角度不可信"
        )
    elif worst_angle is not None and worst_angle[1] > _MAX_VERTICAL_DISPARITY_PX:
        warnings.append(
            f"{worst_angle[0]} 垂直視差 {worst_angle[1]:.1f}px"
            f"（標定目標 <{_MAX_VERTICAL_DISPARITY_PX}）。還在雜訊範圍，這一幀照常使用；"
            f"持續偏高再回頭看標定"
        )
    elif worst_all is not None and worst_all[1] > _MISPAIRED_DISPARITY_PX:
        warnings.append(
            f"{worst_all[0]} 垂直視差 {worst_all[1]:.1f}px，左右配對錯了。"
            f"角度用到的點正常，但深度範圍這類整體指標會被它拉走"
        )

    if measurement.edge_keypoints:
        used = [n for n in measurement.edge_keypoints if n in angle_keypoints(measurement.theta_ca_side)]
        listed = "、".join(measurement.edge_keypoints[:5])
        more = f" 等 {len(measurement.edge_keypoints)} 個點" if len(measurement.edge_keypoints) > 5 else ""
        warnings.append(
            f"{listed}{more} 貼在畫面邊緣，真實位置在畫面外。把相機轉向那個方向。"
            + ("角度用到的點也在裡面，這一幀不可信" if used else "角度沒有用到這些點")
        )

    bad_depths = _implausible_depths(measurement)
    if bad_depths:
        low, high = _PLAUSIBLE_DEPTH_MM
        listed = "、".join(f"{name} {z:.0f}mm" for name, z in bad_depths[:4])
        more = f" 等 {len(bad_depths)} 個點的" if len(bad_depths) > 4 else ""
        reason = (
            "負深度代表點在相機後方，只可能是左右配對錯誤"
            if any(z < 0 for _, z in bad_depths)
            else "整體偏掉通常是標定的 --square-size-mm 填錯"
        )
        tail = "深度超出" if more else " 深度超出"
        warnings.append(f"{listed}{more}{tail} {low:.0f}~{high:.0f}mm 的合理區間。{reason}")

    precision = measurement.theta_ca_precision_deg
    distance = measurement.reference_depth_mm
    azimuth = measurement.camera_azimuth_deg
    # 深度本身就不合理時，換算出來的誤差只是同一個原因的衍生結果。
    # 上面已經講過深度了，再報一次誤差只會把真正的線索淹掉。
    depth_is_sane = distance is not None and _PLAUSIBLE_DEPTH_MM[0] <= distance <= _PLAUSIBLE_DEPTH_MM[1]
    if precision is not None and precision > _THETA_CA_THRESHOLD_DEG and depth_is_sane:
        where = f"距離 {distance:.0f} mm"
        where += "（方位角量不到）" if azimuth is None else f"、方位角 {azimuth:.0f}°"
        warnings.append(
            f"{where}，θ_CA 單幀誤差 ±{precision:.1f}°，超過 "
            f"{_THETA_CA_THRESHOLD_DEG:.0f}° 的判定門檻本身，這一幀沒有參考價值。"
            + precision_advice(distance, azimuth)
        )

    if measurement.shared_count < 4:
        warnings.append(
            f"左右只有 {measurement.shared_count} 個共同關鍵點，樣本太少"
        )
    return warnings
