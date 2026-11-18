"""趟次層級的分析：同趟配對、側傾、表4 的方位角比值、略過率的分軸規則。

`session_analysis` 一次看一份 CSV，`cohort` 按（受試者, 姿勢）分組；這裡看的是
**同一趟**（標頭的 `trip_id`）裡的段落彼此怎麼比。2026-11-18 study 重排之後
才有 trip_id，舊檔沒有這一列，配對時會被明確列成「未配對」，不會默默消失。

## 為什麼要同趟

實機資料顯示原始角度跨趟差很多：同一個人的 θ_CA 跨趟最多差 16°，θ_sym 的零點
每趟不同。所以自然坐要減**同一趟**端正坐的原始角度，跨趟的零點漂移才約得掉。
同一趟有兩段端正坐時（前 5 人趟 1 最後多量一段，檢查漂移）用第一段，與 study
轉身檢查的參考值一致。

## 統計單位

配對差值的統計單位是「受試者 × 桌面條件」，不是幀。每一列的誤差是兩段各自的
批次平均誤差（`uncertainty.standard_error`）合成，不是 std/√N；跨受試者的平均
誤差則是 std/√n（n 是人數），理由見 `cohort.Separation.error_deg`。

## 略過率的分軸規則（2026-11-17 定案）

- θ_sym 上限 20%，θ_CA 15%，θ_KA 沿用 θ_CA 的 15%（它看的是同一批可用幀的
  略過率，不是 θ_KA 自己的缺值率：髖常被桌面擋住，θ_KA 另有稀疏檢查）。
- 超過上限的段落標「附註納入」，照樣進主要結果；另外以 15% 與 20% 兩套門檻
  （超過就排除）各算一次，供敏感度分析。
- 方位角相對參考值漂超過 10° 的段落（轉身）一律作廢，不受放寬影響。

這個模組不碰相機、終端機與 argparse。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .session_analysis import SessionSummary
from .study import TURN_LIMIT_DEG, UPRIGHT, is_natural, is_tilt
from .uncertainty import standard_error

AXES = ("θ_CA", "θ_sym", "θ_KA")
AXIS_LIMITS = {"θ_CA": 0.15, "θ_sym": 0.20, "θ_KA": 0.15}
SENSITIVITY_LIMITS = (0.15, 0.20)

INCLUDED = "納入"
NOTED = "附註納入"
VOID = "作廢"
UNPAIRED = "未配對"

# 表4：3D θ_CA 的平均絕對值低於這個值時，tan θ_3D 接近 0，比值會被雜訊放大到
# 沒有意義，照列但標「不穩」。
MIN_RATIO_THETA_DEG = 5.0
_SIN_45 = math.sin(math.radians(45.0))


# ---- 中繼資料 ----------------------------------------------------------

def _meta_float(summary: SessionSummary, key: str) -> float | None:
    value = summary.meta.get(key)
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def trip_id(summary: SessionSummary) -> str | None:
    return summary.meta.get("trip_id") or None


def _order_key(summary: SessionSummary):
    return (summary.meta.get("captured_at", ""), summary.trial or 0, summary.path.name)


def trips(summaries: list[SessionSummary]) -> dict[tuple[str, str], list[SessionSummary]]:
    """(受試者, trip_id) → 這一趟的段落，依量測時間排序。沒有 trip_id 的不收。"""
    out: dict[tuple[str, str], list[SessionSummary]] = {}
    for summary in summaries:
        tid = trip_id(summary)
        if tid is None:
            continue
        out.setdefault((summary.subject_key, tid), []).append(summary)
    for group in out.values():
        group.sort(key=_order_key)
    return out


def trip_upright(summary: SessionSummary,
                 index: dict[tuple[str, str], list[SessionSummary]]) -> SessionSummary | None:
    """同一趟的第一段端正坐。沒有 trip_id 或那一趟沒有端正坐就是 None。

    相機掉線的那一份（檔尾有 `# camera_lost=1`）跳過：study 會在同一趟裡重量，
    殘缺的那份往往只有幾秒。
    """
    tid = trip_id(summary)
    if tid is None:
        return None
    for other in index.get((summary.subject_key, tid), []):
        if other.condition == UPRIGHT and "camera_lost" not in other.meta:
            return other
    return None


def trip_has_baseline(summary: SessionSummary,
                      index: dict[tuple[str, str], list[SessionSummary]]) -> bool:
    """這一趟有沒有任何一段扣了基準。沒有 trip_id 時只看這一段自己。"""
    tid = trip_id(summary)
    if tid is None:
        return summary.has_baseline
    return any(s.has_baseline for s in index.get((summary.subject_key, tid), []))


# ---- 方位角與轉身 ------------------------------------------------------

def mean_azimuth(summary: SessionSummary) -> float | None:
    column = summary.columns.get("方位角 °")
    return None if column is None else column.mean


def is_azimuth_trip(summary: SessionSummary) -> bool:
    """表4 那一趟的段落：標頭有 azimuth_target_deg。方位角改變是刻意搬相機。"""
    return "azimuth_target_deg" in summary.meta


def turned_deg(summary: SessionSummary,
               index: dict[tuple[str, str], list[SessionSummary]]) -> float | None:
    """這一段相對參考方位角轉了多少。與 study 的規則相同：

    - 有基準的段落：對標頭記的 baseline_azimuth_deg。
    - 沒有基準的段落（自然坐）：對同一趟第一段端正坐的平均方位角。
    - 表4 那一趟、整趟沒有基準、或找不到參考值：None（不判轉身）。
    """
    if is_azimuth_trip(summary) or not trip_has_baseline(summary, index):
        return None
    mean = mean_azimuth(summary)
    if mean is None:
        return None
    reference = _meta_float(summary, "baseline_azimuth_deg")
    if reference is None and not summary.has_baseline:
        upright = trip_upright(summary, index)
        if upright is not None and upright is not summary:
            reference = mean_azimuth(upright)
    return None if reference is None else float(mean - reference)


def is_turned(turn: float | None) -> bool:
    return turn is not None and abs(turn) > TURN_LIMIT_DEG


# ---- 略過率 ------------------------------------------------------------

def axis_rejection_rate(summary: SessionSummary, axis: str) -> float:
    """這一軸沒有值的幀佔全部幀的比例。θ_KA 沿用 θ_CA（見模組說明）。"""
    if summary.frames == 0:
        return 1.0
    if axis == "θ_KA":
        axis = "θ_CA"
    angle = summary.angles.get(f"{axis} 原始")
    count = 0 if angle is None else angle.count
    return 1.0 - count / summary.frames


def axis_status(rates: list[float], axis: str, turned: bool) -> str:
    """一軸的納入狀態。rates 是參與這個量的各段略過率（配對時是兩段）。"""
    if turned:
        return VOID
    return NOTED if max(rates) > AXIS_LIMITS[axis] else INCLUDED


def passes(rates: list[float], status: str, cutoff: float | None) -> bool:
    """敏感度分析用：cutoff 是 None 時就是主要結果（作廢之外全收）。"""
    if status in (VOID, UNPAIRED):
        return False
    return cutoff is None or max(rates) <= cutoff


# ---- 配對差值 ----------------------------------------------------------

def _raw(summary: SessionSummary, axis: str) -> np.ndarray | None:
    angle = summary.angles.get(f"{axis} 原始")
    if angle is None:
        return None
    if axis == "θ_KA" and angle.is_sparse:
        return None
    return angle.values


def _shoulder(summary: SessionSummary) -> np.ndarray | None:
    column = summary.columns.get("肩高 mm")
    return None if column is None else column.values


@dataclass(frozen=True)
class Quantity:
    """配對差值裡的一個量：名稱、取值方式、略過率看哪一軸、CSV 欄名。"""

    label: str
    values: object        # SessionSummary -> np.ndarray | None
    axis: str
    column: str
    unit: str = "°"


NATURAL_QUANTITIES = (
    Quantity("θ_CA", lambda s: _raw(s, "θ_CA"), "θ_CA", "d_theta_ca_deg"),
    Quantity("θ_sym", lambda s: _raw(s, "θ_sym"), "θ_sym", "d_theta_sym_deg"),
    Quantity("θ_KA", lambda s: _raw(s, "θ_KA"), "θ_KA", "d_theta_ka_deg"),
)
# 軀幹前傾：肩高與 θ_KA 並排。兩者在軀幹前傾時會一起動，不是兩份獨立證據。
# 肩高的略過率跟著 θ_sym（同樣只用雙肩）。
FORWARD_QUANTITIES = (
    Quantity("肩高", _shoulder, "θ_sym", "d_shoulder_height_mm", "mm"),
    Quantity("θ_KA", lambda s: _raw(s, "θ_KA"), "θ_KA", "d_theta_ka_deg"),
)


@dataclass(frozen=True)
class Difference:
    value: float | None
    error: float | None
    rates: tuple[float, float]
    status: str
    note: str = ""


@dataclass
class PairedRow:
    """一段（自然坐或軀幹前傾）減同一趟第一段端正坐。"""

    target: SessionSummary
    reference: SessionSummary | None
    differences: dict[str, Difference] = field(default_factory=dict)
    turned_deg: float | None = None
    # 參考的 upright 自己相對基準轉了多少。它轉身的話這一列也作廢（GPT 審查第 2 條）。
    reference_turned_deg: float | None = None
    reason: str = ""          # 未配對時寫原因

    @property
    def subject(self) -> str:
        return self.target.subject_key

    @property
    def paired(self) -> bool:
        return self.reference is not None

    @property
    def desk_level(self) -> str:
        return self.target.meta.get("desk_level", "")


def _difference(target, reference, quantity: Quantity, turned: bool) -> Difference:
    rates = (axis_rejection_rate(target, quantity.axis),
             axis_rejection_rate(reference, quantity.axis))
    status = axis_status(list(rates), quantity.axis, turned)
    a, b = quantity.values(target), quantity.values(reference)
    if a is None or b is None or a.size < 2 or b.size < 2:
        which = "這一段" if a is None or a.size < 2 else "同趟端正坐"
        note = (f"{which}的 θ_KA 有值的幀太少" if quantity.label == "θ_KA"
                else f"{which}沒有足夠的 {quantity.label}")
        return Difference(None, None, rates, status, note)
    error_a, error_b = standard_error(a), standard_error(b)
    error = (None if error_a is None or error_b is None
             else float(math.hypot(error_a, error_b)))
    return Difference(float(a.mean() - b.mean()), error, rates, status)


def paired_differences(summaries: list[SessionSummary], select=is_natural,
                       quantities=NATURAL_QUANTITIES) -> list[PairedRow]:
    """select 選出的每一段減同一趟第一段端正坐的原始值。找不到的照樣回傳、寫原因。"""
    index = trips(summaries)
    rows = []
    for summary in sorted(summaries, key=lambda s: (s.subject_key, _order_key(s))):
        if not select(summary.condition or ""):
            continue
        if trip_id(summary) is None:
            rows.append(PairedRow(summary, None, reason="沒有 trip_id（2026-11-18 之前的舊檔）"))
            continue
        reference = trip_upright(summary, index)
        if reference is None:
            rows.append(PairedRow(summary, None, reason="同一趟沒有 upright"))
            continue
        turn = turned_deg(summary, index)
        reference_turn = turned_deg(reference, index)
        row = PairedRow(summary, reference, turned_deg=turn,
                        reference_turned_deg=reference_turn)
        void = is_turned(turn) or is_turned(reference_turn)
        for quantity in quantities:
            row.differences[quantity.label] = _difference(
                summary, reference, quantity, void)
        rows.append(row)
    return rows


@dataclass(frozen=True)
class GroupMean:
    """跨受試者的平均。誤差分母是人數；只有一個人時誤差是 None。"""

    n: int
    mean: float | None
    error: float | None


def group_mean(values: list[float]) -> GroupMean:
    if not values:
        return GroupMean(0, None, None)
    array = np.asarray(values, dtype=float)
    error = (float(array.std(ddof=1) / np.sqrt(array.size)) if array.size >= 2
             else None)
    return GroupMean(int(array.size), float(array.mean()), error)


def sensitivity(rows: list[PairedRow], label: str, desk_level: str | None = None,
                cutoff: float | None = None) -> GroupMean:
    """一個量在某個桌面條件下的跨受試者平均。cutoff 見 `passes`。

    同一位受試者在同一個桌面條件有好幾列時先取平均，每人只算一次。
    """
    per_subject: dict[str, list[float]] = {}
    for row in rows:
        if not row.paired or (desk_level is not None and row.desk_level != desk_level):
            continue
        diff = row.differences.get(label)
        if diff is None or diff.value is None:
            continue
        if not passes(list(diff.rates), diff.status, cutoff):
            continue
        per_subject.setdefault(row.subject, []).append(diff.value)
    return group_mean([float(np.mean(v)) for v in per_subject.values()])


# ---- 側傾 --------------------------------------------------------------

# θ_sym 右肩較高為正、左肩較高為負。往左側彎時左肩往下、右肩較高，所以預期為
# 正；往右側彎預期為負。這是從定義推出來的，第一批實機資料要核對一次。
TILT_SIGN = {"tilt-left": +1.0, "tilt-right": -1.0}


@dataclass(frozen=True)
class TiltRow:
    summary: SessionSummary
    target_deg: float | None
    measured_deg: float | None
    theta_sym_deg: float | None      # 相對端正坐的 θ_sym（見 source）
    theta_sym_error_deg: float | None
    source: str                      # 扣基準／減同趟端正坐／原始
    rate: float
    status: str
    turned_deg: float | None

    @property
    def side(self) -> str:
        return "左" if self.summary.condition == "tilt-left" else "右"

    @property
    def expected_deg(self) -> float | None:
        if self.measured_deg is None:
            return None
        return TILT_SIGN[self.summary.condition] * self.measured_deg

    @property
    def error_deg(self) -> float | None:
        """θ_sym − 帶號的量角器實測值。"""
        if self.theta_sym_deg is None or self.expected_deg is None:
            return None
        return self.theta_sym_deg - self.expected_deg

    @property
    def magnitude_error_deg(self) -> float | None:
        """|θ_sym| − 實測值。比左右對不對稱時看這個，正負號已經拿掉。"""
        if self.theta_sym_deg is None or self.measured_deg is None:
            return None
        return abs(self.theta_sym_deg) - self.measured_deg

    @property
    def sign_ok(self) -> bool | None:
        if self.theta_sym_deg is None:
            return None
        return bool(np.sign(self.theta_sym_deg) == TILT_SIGN[self.summary.condition])


def tilt_rows(summaries: list[SessionSummary]) -> list[TiltRow]:
    """每一段側傾的 θ_sym 與量角器實測值。

    θ_sym 優先用扣基準的值（側傾的趟都有基準）；沒有基準時用原始值減同一趟端正坐
    的原始值；兩者都沒有才用原始值，並標出來，那個數字含著當趟的零點。
    """
    index = trips(summaries)
    out = []
    for summary in sorted(summaries, key=lambda s: (s.subject_key, _order_key(s))):
        if not is_tilt(summary.condition or ""):
            continue
        corrected = summary.angles.get("θ_sym 扣基準")
        raw = summary.angles.get("θ_sym 原始")
        value = error = None
        source = "原始"
        if summary.has_baseline and corrected is not None:
            value, error, source = corrected.mean, corrected.standard_error, "扣基準"
        elif raw is not None:
            upright = trip_upright(summary, index)
            ref = None if upright is None else upright.angles.get("θ_sym 原始")
            if ref is not None:
                value = raw.mean - ref.mean
                e1, e2 = raw.standard_error, ref.standard_error
                error = None if e1 is None or e2 is None else float(math.hypot(e1, e2))
                source = "減同趟端正坐"
            else:
                value, error = raw.mean, raw.standard_error
        turn = turned_deg(summary, index)
        rate = axis_rejection_rate(summary, "θ_sym")
        out.append(TiltRow(
            summary=summary,
            target_deg=_meta_float(summary, "target_angle_deg"),
            measured_deg=_meta_float(summary, "measured_angle_deg"),
            theta_sym_deg=value, theta_sym_error_deg=error, source=source,
            rate=rate, status=axis_status([rate], "θ_sym", is_turned(turn)),
            turned_deg=turn,
        ))
    return out


def tilt_by_angle(rows: list[TiltRow], cutoff: float | None = None
                  ) -> dict[tuple[str, float | None], GroupMean]:
    """(左右, 目標角度) → 誤差的跨受試者平均。看誤差會不會隨角度變大。"""
    buckets: dict[tuple[str, float | None], dict[str, list[float]]] = {}
    for row in rows:
        if row.error_deg is None or not passes([row.rate], row.status, cutoff):
            continue
        key = (row.side, row.target_deg)
        buckets.setdefault(key, {}).setdefault(row.summary.subject_key, []).append(
            row.error_deg)
    return {key: group_mean([float(np.mean(v)) for v in subjects.values()])
            for key, subjects in sorted(buckets.items(),
                                        key=lambda kv: (kv[0][0], kv[0][1] or 0))}


def tilt_symmetry(rows: list[TiltRow]) -> dict[str, tuple[float | None, float | None]]:
    """受試者 → (左傾的 |θ_sym|−實測 平均, 右傾的)。兩邊差很多就是不對稱。"""
    out: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if row.magnitude_error_deg is None or row.status == VOID:
            continue
        out.setdefault(row.summary.subject_key, {}).setdefault(row.side, []).append(
            row.magnitude_error_deg)
    return {subject: (float(np.mean(sides["左"])) if "左" in sides else None,
                      float(np.mean(sides["右"])) if "右" in sides else None)
            for subject, sides in sorted(out.items())}


# ---- 表4：方位角比值 ---------------------------------------------------

@dataclass(frozen=True)
class RatioRow:
    """一段的 tan θ_2D / tan θ_3D，與理論值 sin φ / sin 45° 比較。

    φ 用這些幀實際算出的平均方位角，不是 azimuth_target_deg（那只是標記）。
    判斷看比值而不是絕對角度，所以三段的姿勢稍有不同也不影響。
    """

    summary: SessionSummary
    target_deg: float | None
    azimuth_deg: float
    frames: int
    theta_3d_deg: float
    theta_2d_deg: float
    ratio: float | None
    ratio_error: float | None

    @property
    def predicted(self) -> float:
        return math.sin(math.radians(self.azimuth_deg)) / _SIN_45

    @property
    def unstable(self) -> bool:
        return abs(self.theta_3d_deg) < MIN_RATIO_THETA_DEG

    @property
    def deviation(self) -> float | None:
        """比值 − 理論值。"""
        return None if self.ratio is None else self.ratio - self.predicted


def ratio_rows(summaries: list[SessionSummary]) -> list[RatioRow]:
    """標頭有 azimuth_target_deg 的段落（表4 那一趟），各算一個比值。

    比值取 mean(tan θ_2D) / mean(tan θ_3D)，不是逐幀比值的平均：θ_3D 接近 0 的幀
    會讓逐幀比值爆掉。誤差用兩者各自的批次平均誤差做一階傳播。
    """
    out = []
    for summary in sorted(summaries, key=lambda s: (s.subject_key, _order_key(s))):
        if not is_azimuth_trip(summary) or not summary.legacy_pairs:
            continue
        pairs = np.asarray(summary.legacy_pairs, dtype=float)
        tan_3d = np.tan(np.radians(pairs[:, 1]))
        tan_2d = np.tan(np.radians(pairs[:, 2]))
        a, b = float(tan_2d.mean()), float(tan_3d.mean())
        ratio = None if b == 0 else a / b
        error = None
        if ratio is not None:
            ea, eb = standard_error(tan_2d), standard_error(tan_3d)
            if ea is not None and eb is not None:
                # σ_r² = (σ_a/b)² + (a·σ_b/b²)²。不寫成相對誤差，分子平均是 0 時才算得出來。
                error = math.hypot(ea / b, a * eb / (b * b))
        out.append(RatioRow(
            summary=summary, target_deg=_meta_float(summary, "azimuth_target_deg"),
            azimuth_deg=float(pairs[:, 0].mean()), frames=int(pairs.shape[0]),
            theta_3d_deg=float(pairs[:, 1].mean()), theta_2d_deg=float(pairs[:, 2].mean()),
            ratio=ratio, ratio_error=error,
        ))
    return out


def ratio_by_group(rows: list[RatioRow], groups: dict[str, str] | None
                   ) -> dict[tuple[str, float | None], tuple[GroupMean, GroupMean]]:
    """(身高層, 目標方位角) → (比值 − 理論值 的跨受試者平均, 實際方位角的平均)。

    不穩的段落不收。沒給分組表時全部歸「未分組」。
    """
    buckets: dict[tuple[str, float | None], dict[str, list[tuple[float, float]]]] = {}
    for row in rows:
        if row.deviation is None or row.unstable:
            continue
        group = (groups or {}).get(row.summary.subject_key, "未分組")
        buckets.setdefault((group, row.target_deg), {}).setdefault(
            row.summary.subject_key, []).append((row.deviation, row.azimuth_deg))
    out = {}
    for key in sorted(buckets, key=lambda k: (k[0], k[1] or 0)):
        subjects = buckets[key]
        deviations = [float(np.mean([d for d, _ in v])) for v in subjects.values()]
        azimuths = [float(np.mean([a for _, a in v])) for v in subjects.values()]
        out[key] = (group_mean(deviations), group_mean(azimuths))
    return out


def read_groups(path) -> dict[str, str]:
    """受試者分組表（CSV，欄位 subject,group）。身高層由操作者另外整理，不進逐幀 CSV。"""
    import csv
    from pathlib import Path

    groups = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"subject", "group"} <= set(reader.fieldnames):
            raise ValueError(f"{path} 要有 subject 與 group 兩欄")
        for row in reader:
            subject = (row.get("subject") or "").strip()
            if subject:
                groups[subject] = (row.get("group") or "").strip() or "未分組"
    return groups
