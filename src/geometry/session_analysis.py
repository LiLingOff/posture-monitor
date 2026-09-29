"""分析 live 留下的逐幀 CSV。

這一層不碰相機，所以在任何機器上都跑得動。量測在 Jetson 上做，分析可以在別處。

要看的東西分成三類：

**資料的品質。** 可用率、被略過的原因分成幾類。每一幀都因為同一件事被擋掉的時候，
那件事就是問題本身。

**角度的統計。** 平均、單幀散佈、平均值的誤差。誤差不能用 `std/√N`，因為相鄰幀
是相關的（見 uncertainty 模組），所以這裡一併把自相關與區段平均印出來：
區段平均的散佈遠大於白雜訊的預期值，就是相關性的直接證據。

**判定的結果。** CSV 有 `posture` 欄時直接統計，沒有的話（舊檔案，或當時加了
`--no-judge`）就用扣除基準後的角度重播一遍。重播包含移動平均的過期處理，
所以結果與當時執行 `live` 會看到的一致。

重播的好處是可以換參數重跑。誤報率對 `--margin` 有多敏感，這樣才問得出來。
"""
from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .baseline import group_rejection_reason
from .judgement import Posture, PostureJudge, windows_are_stale
from .smoothing import RollingAngle
from .uncertainty import lag1_autocorrelation, standard_error

# 沒有記錄姿勢條件的檔案歸在這一組。2026-09-24 與 09-29 的記錄寫在 condition
# 欄存在之前，而它們的分析價值最高，所以要有一個歸屬而不是被濾掉。
UNLABELLED = "未標註"
# 區段的數量。太少看不出漂移的形狀，太多每一段的平均本身就不穩。
_SEGMENTS = 10
# 相關性的警告線：區段平均的散佈超過白雜訊預期值這個倍數，就值得講出來。
_CORRELATION_FACTOR = 1.5


@dataclass(frozen=True)
class Column:
    """CSV 裡一欄的統計。缺欄或整欄是空的時候不建立。"""

    name: str
    values: np.ndarray

    @property
    def mean(self) -> float:
        return float(self.values.mean())

    @property
    def minimum(self) -> float:
        return float(self.values.min())

    @property
    def maximum(self) -> float:
        return float(self.values.max())

    @property
    def spread(self) -> float:
        return float(self.values.std())


@dataclass(frozen=True)
class AngleSummary:
    """一個角度在整段量測裡的統計。"""

    name: str
    values: np.ndarray

    @property
    def count(self) -> int:
        return int(self.values.size)

    @property
    def mean(self) -> float:
        return float(self.values.mean())

    @property
    def single_frame_std(self) -> float:
        return float(self.values.std())

    @property
    def minimum(self) -> float:
        return float(self.values.min())

    @property
    def maximum(self) -> float:
        return float(self.values.max())

    @property
    def standard_error(self) -> float | None:
        """已經把相鄰幀的相關性算進去，不是 std/√N。"""
        return standard_error(self.values)

    @property
    def naive_standard_error(self) -> float:
        """std/√N。只用來對照，說明低估了多少。"""
        return float(self.values.std(ddof=1) / np.sqrt(self.values.size))

    @property
    def autocorrelation(self) -> float | None:
        return lag1_autocorrelation(self.values)

    @property
    def segment_means(self) -> np.ndarray:
        """切成等長的區段各自取平均。漂移的形狀看這個。"""
        size = self.count // _SEGMENTS
        if size < 2:
            return np.array([])
        trimmed = self.values[: size * _SEGMENTS].reshape(_SEGMENTS, size)
        return trimmed.mean(axis=1)

    @property
    def white_noise_segment_spread(self) -> float | None:
        """若相鄰幀真的獨立，區段平均該有的散佈。"""
        size = self.count // _SEGMENTS
        if size < 2:
            return None
        return self.single_frame_std / np.sqrt(size)

    @property
    def is_correlated(self) -> bool:
        expected = self.white_noise_segment_spread
        segments = self.segment_means
        if expected is None or segments.size == 0 or expected == 0:
            return False
        return float(segments.std()) > _CORRELATION_FACTOR * expected


@dataclass(frozen=True)
class JudgementReplay:
    """把判定重播一遍的結果。"""

    window: int
    margin: float
    counts: Counter
    transitions: int
    first_over_frame: int | None
    replayed: bool          # True 是重播的，False 是直接讀 CSV 的 posture 欄

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def share(self, posture: str) -> float:
        return 0.0 if self.total == 0 else self.counts[posture] / self.total


@dataclass
class SessionSummary:
    """一份逐幀記錄的完整摘要。"""

    path: Path
    meta: dict[str, str] = field(default_factory=dict)
    frames: int = 0
    usable: int = 0
    duration_s: float | None = None
    reasons: Counter = field(default_factory=Counter)
    reason_examples: dict[str, str] = field(default_factory=dict)
    angles: dict[str, AngleSummary] = field(default_factory=dict)
    columns: dict[str, Column] = field(default_factory=dict)
    has_baseline: bool = False
    judgement: JudgementReplay | None = None
    azimuth_bins: list = field(default_factory=list)

    @property
    def subject(self) -> str | None:
        return self.meta.get("subject")

    @property
    def condition(self) -> str | None:
        """姿勢條件。早期的檔案沒有這一項，彙整時要看得出來是「沒記」。"""
        return self.meta.get("condition")

    @property
    def trial(self) -> int | None:
        value = self.meta.get("trial")
        return int(value) if value and value.isdigit() else None

    @property
    def subject_key(self) -> str:
        """分組用的受試者名稱。沒記的檔案也要能歸到一組，不能整批消失。"""
        return self.subject or "?"

    @property
    def condition_key(self) -> str:
        """分組用的姿勢名稱，見 UNLABELLED 的說明。"""
        return self.condition or UNLABELLED

    def angle(self, name: str) -> AngleSummary | None:
        """取某個角度，優先用扣除基準之後的值。

        「扣基準優先、沒有才退回原始」是一項政策判斷，不是隨手寫的查表：
        判定看的是扣除後的角度，所以統計也該用同一個。放在這裡是為了讓
        `analyse_session` 裡那幾個標籤字串只出現一次；分散到各個報表去拼的話，
        改一個標籤會讓其他地方安靜地變成 None。
        """
        return self.angles.get(f"{name} 扣基準") or self.angles.get(f"{name} 原始")

    @property
    def rejected(self) -> int:
        return self.frames - self.usable

    @property
    def rejection_rate(self) -> float:
        return 0.0 if self.frames == 0 else self.rejected / self.frames


def _number(text: str) -> float | None:
    """空字串代表這一格沒有值。0 是合法的角度，不能拿它當缺失。"""
    if text is None or text.strip() == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_rows(path: Path) -> tuple[list[dict], dict[str, str]]:
    """讀出資料列與標頭的 `# key=value` 區塊。

    註解列用 `#` 開頭，pandas 讀得掉，csv 模組讀不掉，所以在這裡先濾。

    回傳的中繼資料是原樣的字串對照表，缺的鍵就是缺。早期的檔案只有
    `# subject=`，而那幾份的分析價值最高，不能因為少了欄位就拒讀。
    不認得的鍵也照收，日後多記一項不必同步改這裡。
    """
    text = Path(path).read_text(encoding="utf-8").splitlines()
    meta: dict[str, str] = {}
    body = []
    for line in text:
        if line.startswith("#"):
            key, sep, value = line.lstrip("#").strip().partition("=")
            if sep:
                meta[key.strip()] = value.strip()
            continue
        body.append(line)
    if not body:
        raise ValueError(f"{path} 沒有任何資料列")
    rows = list(csv.DictReader(body))
    if not rows:
        raise ValueError(f"{path} 只有標題列，沒有資料。那一次量測可能沒跑起來")
    return rows, meta


def _collect(rows: list[dict], column: str) -> np.ndarray:
    """把一欄裡有值的格子收成陣列。缺欄時回傳空陣列。"""
    if not rows or column not in rows[0]:
        return np.array([])
    values = [_number(r.get(column, "")) for r in rows]
    return np.array([v for v in values if v is not None], dtype=float)


def _replay_judgement(
    rows: list[dict], window: int, margin: float
) -> JudgementReplay | None:
    """按照 live 的順序重走一遍判定，包含移動平均的過期處理。

    過期處理要一起重播，否則偵測連續失敗那幾段的狀態會與當時看到的不同。
    """
    judge = PostureJudge(margin_factor=margin)
    ca_window, sym_window = RollingAngle(window), RollingAngle(window)
    counts: Counter = Counter()
    transitions = 0
    first_over = None
    misses = 0
    seen_any = False

    for row in rows:
        if row.get("usable") == "1":
            misses = 0
            ca = _number(row.get("theta_ca_corrected_deg", ""))
            sym = _number(row.get("theta_sym_corrected_deg", ""))
            if ca is not None:
                ca_window.add(ca)
                seen_any = True
            if sym is not None:
                sym_window.add(sym)
        else:
            misses += 1
            if windows_are_stale(misses, window):
                ca_window.clear()
                sym_window.clear()

        _, _, verdict = judge.update(ca_window, sym_window)
        counts[verdict.posture.value] += 1
        if verdict.changed:
            transitions += 1
            if verdict.posture is Posture.OVER and first_over is None:
                first_over = int(_number(row.get("frame", "")) or 0)

    if not seen_any:
        return None
    # 第一次進入未知不算狀態變化，那是起始狀態。
    return JudgementReplay(window, margin, counts, max(0, transitions - 1),
                           first_over, replayed=True)


def _read_judgement(rows: list[dict]) -> JudgementReplay | None:
    """直接讀 CSV 的 posture 欄。當時用什麼參數判的就是什麼結果。"""
    if not rows or "posture" not in rows[0]:
        return None
    values = [r.get("posture", "").strip() for r in rows]
    if not any(values):
        return None
    counts = Counter(v for v in values if v)
    transitions = sum(1 for a, b in zip(values, values[1:]) if a != b and b)
    first_over = next(
        (int(_number(r.get("frame", "")) or 0) for r, v in zip(rows, values)
         if v == Posture.OVER.value), None
    )
    return JudgementReplay(0, 0.0, counts, transitions, first_over, replayed=False)


def _used_a_baseline(summary: SessionSummary, meta: dict[str, str]) -> bool:
    """這一段量測有沒有扣除個人基準。

    標頭有記就照標頭。先前只能比對「扣除後的值與原始值是否相同」，那是在
    CSV 還不會自報家門的年代留下的推測，而它在一種真實的情況下會答錯：
    基準剛好是 0.0° 時兩欄完全相同，於是一份有效的量測被標成沒有基準，
    報表跟著印出「這個判定不能當成誤報率」。

    沒有標頭的舊檔案才退回比對。2026-09-24 與 09-29 的記錄都是那個年代的。
    """
    if "baseline_file" in meta or "baseline_theta_ca_deg" in meta:
        return True
    raw = summary.angles.get("θ_CA 原始")
    corrected = summary.angles.get("θ_CA 扣基準")
    return (raw is not None and corrected is not None
            and not np.allclose(raw.values, corrected.values))


def analyse_session(
    path: Path, window: int = 30, margin: float = 1.0, force_replay: bool = False
) -> SessionSummary:
    """讀一份逐幀 CSV，算出所有要看的數字。"""
    rows, meta = read_rows(path)
    summary = SessionSummary(path=Path(path), meta=meta, frames=len(rows))

    for row in rows:
        if row.get("usable") == "1":
            summary.usable += 1
            continue
        reason = (row.get("reject_reason") or "").strip() or "沒有寫明原因"
        key = group_rejection_reason(reason)
        summary.reasons[key] += 1
        summary.reason_examples.setdefault(key, reason)

    elapsed = _collect(rows, "elapsed_s")
    if elapsed.size:
        summary.duration_s = float(elapsed.max())

    usable_rows = [r for r in rows if r.get("usable") == "1"]
    for label, column in (
        ("θ_CA 原始", "theta_ca_deg"),
        ("θ_sym 原始", "theta_sym_deg"),
        ("θ_CA 扣基準", "theta_ca_corrected_deg"),
        ("θ_sym 扣基準", "theta_sym_corrected_deg"),
    ):
        values = _collect(usable_rows, column)
        if values.size >= 2:
            summary.angles[label] = AngleSummary(label, values)

    summary.has_baseline = _used_a_baseline(summary, meta)

    for label, column in (
        ("距離 mm", "distance_mm"),
        ("方位角 °", "azimuth_deg"),
        ("θ_CA 單幀理論誤差 °", "theta_ca_precision_deg"),
        ("共同關鍵點", "shared_keypoints"),
        ("垂直視差 全部 px", "max_vertical_disparity_px"),
        ("垂直視差 角度用 px", "angle_max_vertical_disparity_px"),
    ):
        values = _collect(usable_rows, column)
        values = values[np.isfinite(values)]
        if values.size:
            summary.columns[label] = Column(label, values)

    summary.azimuth_bins = rejection_by_azimuth(rows)
    stored = None if force_replay else _read_judgement(rows)
    summary.judgement = stored or _replay_judgement(rows, window, margin)
    return summary


def correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    """兩欄之間的相關係數。長度不同或其中一欄沒有變化時回傳 None。

    用來檢查角度有沒有殘餘的方位角相依性。解剖平面定得對的話這個值該接近 0。
    """
    if a.size != b.size or a.size < 3:
        return None
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def compare(a: AngleSummary, b: AngleSummary) -> tuple[float, float | None, float | None]:
    """兩段量測的差距、合併誤差與顯著性（幾個標準誤差）。"""
    difference = b.mean - a.mean
    error_a, error_b = a.standard_error, b.standard_error
    if error_a is None or error_b is None:
        return difference, None, None
    combined = float(np.hypot(error_a, error_b))
    sigma = None if combined == 0 else abs(difference) / combined
    return difference, combined, sigma


# 方位角分箱的寬度。太窄每一箱的幀數不夠，太寬看不出劣化從哪裡開始。
_AZIMUTH_BIN_DEG = 10.0
# 少於這麼多幀的箱子不報，比例本身不可信。
_MINIMUM_BIN_FRAMES = 15
# 方位角的跨度小於這個值時整段不報，那不是一次掃描而是一個定點。
_MINIMUM_AZIMUTH_SPAN_DEG = 15.0


@dataclass(frozen=True)
class AzimuthBin:
    """一段方位角範圍內的偵測品質。"""

    low: float
    high: float
    frames: int
    rejected: int
    shoulder_rejected: int
    shared_keypoints: float | None

    @property
    def rejection_rate(self) -> float:
        return 0.0 if self.frames == 0 else self.rejected / self.frames


def rejection_by_azimuth(rows: list[dict]) -> list[AzimuthBin]:
    """略過率隨方位角怎麼變。

    這是「雙目模組能推到多大的方位角」的直接量測。限制不在幾何而在遮擋：
    方位角愈大，遠側的肩膀愈容易被身體擋住，而雙肩正是解剖平面的來源。
    幾何上的精度隨方位角變好，所以上限只能實測，不能用算的。

    被略過的幀也記著方位角，所以這個比例算得出來。分開數「肩膀配對錯誤」，
    因為那正是遮擋的徵狀，而手腕腳踝配錯與方位角無關。

    方位角跨度不夠時回傳空的：定點量測的分箱沒有意義。
    """
    seen = [(_number(r.get("azimuth_deg", "")), r) for r in rows]
    seen = [(a, r) for a, r in seen if a is not None]
    if not seen:
        return []
    values = [a for a, _ in seen]
    if max(values) - min(values) < _MINIMUM_AZIMUTH_SPAN_DEG:
        return []

    buckets: dict[int, list[dict]] = {}
    for azimuth, row in seen:
        buckets.setdefault(int(azimuth // _AZIMUTH_BIN_DEG), []).append(row)

    out: list[AzimuthBin] = []
    for index in sorted(buckets):
        group = buckets[index]
        if len(group) < _MINIMUM_BIN_FRAMES:
            continue
        rejected = [r for r in group if r.get("usable") != "1"]
        shoulder = [r for r in rejected if "shoulder" in (r.get("reject_reason") or "")]
        shared = [_number(r.get("shared_keypoints", "")) for r in group]
        shared = [v for v in shared if v is not None]
        out.append(AzimuthBin(
            low=index * _AZIMUTH_BIN_DEG,
            high=(index + 1) * _AZIMUTH_BIN_DEG,
            frames=len(group),
            rejected=len(rejected),
            shoulder_rejected=len(shoulder),
            shared_keypoints=float(np.mean(shared)) if shared else None,
        ))
    return out
