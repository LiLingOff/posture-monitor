"""評分者信度與有效警報分析（說明書表3）。

**統計單位是段落**（一份逐幀 CSV），不是幀也不是照片。同一位受試者的
好幾段彼此不獨立（同一個人、同一份基準、同一趟），所以卡方與 Fisher 的
p 值只能當描述，報告會註明；受試者層級另列每人貢獻幾段。

**操作定義（與文件 10 一致）：**

- 系統警報：該段判定為超標。兩種判定方式分別算：
  - 相對基準線：CSV 的 `posture` 欄（量測當下的判定，扣個人基準加遲滯）。
    判為「超標」的幀佔「超標＋正常」幀的比例 ≥ alarm_share（預設 0.5）。
    整段沒有判定（舊檔或 --no-judge）時這一段在這個方式下不分類。
  - 絕對標準：可用幀的原始角度（未扣基準）與固定門檻比，設定檔的
    absolute_thresholds_deg（預設 θ_CA ±10°，本專案的設計值）。任一設定的
    角度絕對值超過門檻，這一幀就算超標；超標幀佔有值幀的比例 ≥ alarm_share。
- 評分判不良：該段的主觀評分（評分者平均）比同趟端正坐低 ≥ rating_drop_threshold
  （預設 2 分）。「低」依量表方向：higher_is_better 為 false 時分數變高才是變差。
  段落分數 = 各照片的評分者平均再對照片平均。同趟端正坐有好幾段時取平均。
  沒有 trip_id 的舊資料以受試者為一趟。
- Kendall 判不良：一位評分者對一張照片，不良項目數 ≥ min_failed_items 就是不良；
  一段之中不良的（評分者 × 照片）比例 ≥ agreement.min_bad_share（預設 0.5）。
- 真的不良 = 評分判不良 且 Kendall 判不良。
- 有效警報 = 系統警報 且 真的不良。
- 誤報 = 系統警報 但 不是真的不良（評分或 Kendall 任一條不成立）。
- 漏報 = 系統未警報 但 真的不良。
- 有效警報率 = 有效 / 警報；誤報率 = 誤報 / 警報；漏報率 = 漏報 / 真的不良。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .rating_config import RatingConfig
from .rating_import import KeyRow, RatingSet
from .rating_stats import (BOOTSTRAP_REPS, Interval, TwoByTwo, bootstrap, icc2,
                           kappa, kappa_method, mcnemar_exact)

METHODS = ("relative", "absolute")
METHOD_NAMES = {"relative": "相對基準線", "absolute": "絕對標準"}


# ---- 評分者信度 ------------------------------------------------------------

@dataclass(frozen=True)
class Agreement:
    item: str
    method: str
    estimate: float | None
    interval: Interval
    targets: int            # 每位評分者都評過的照片數
    subjects: int


@dataclass
class Reliability:
    raters: list[str]
    kappas: list[Agreement] = field(default_factory=list)
    iccs_single: list[Agreement] = field(default_factory=list)
    iccs_average: list[Agreement] = field(default_factory=list)


def reliability(ratings: RatingSet, key: dict[str, KeyRow], config: RatingConfig,
                reps: int = BOOTSTRAP_REPS) -> Reliability:
    """只用每位評分者都評過的照片。有人缺的照片不補、不猜。"""
    raters = sorted(ratings.raters)
    result = Reliability(raters=raters)
    if len(raters) < 2:
        return result
    by_target: dict[str, dict[str, object]] = {}
    for rating in ratings.ratings:
        by_target.setdefault(rating.blind_id, {})[rating.rater] = rating
    complete = sorted(b for b, per in by_target.items() if len(per) == len(raters))
    clusters = [key[b].subject for b in complete]
    subjects = len(set(clusters))
    method = kappa_method(len(raters))

    def kendall_rows(name):
        if name is None:
            return [[config.kendall_photo_bad(list(by_target[b][r].kendall.values()))
                     for r in raters] for b in complete]
        return [[by_target[b][r].kendall[name] for r in raters] for b in complete]

    for name in [i.name for i in config.kendall_items] + [None]:
        rows = kendall_rows(name)
        result.kappas.append(Agreement(
            item=name or "Kendall 總判定", method=method,
            estimate=kappa(rows), interval=bootstrap(rows, clusters, kappa, reps),
            targets=len(rows), subjects=subjects))

    for item in config.score_items:
        rows = [[by_target[b][r].scores[item.name] for r in raters] for b in complete]
        for attr, bucket, label in (("single", result.iccs_single, "ICC(2,1)"),
                                    ("average", result.iccs_average, "ICC(2,k)")):
            def stat(sample, attr=attr):
                return getattr(icc2(sample), attr) if len(sample) >= 2 else None
            bucket.append(Agreement(
                item=item.name, method=label,
                estimate=stat(rows), interval=bootstrap(rows, clusters, stat, reps),
                targets=len(rows), subjects=subjects))
    return result


# ---- 系統判定 --------------------------------------------------------------

@dataclass(frozen=True)
class SystemCall:
    alarm: bool | None
    share: float | None
    frames: int


def _read_rows(path: Path) -> list[dict]:
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines()
             if not line.startswith("#")]
    return list(csv.DictReader(lines))


def _float(text) -> float | None:
    try:
        return float(text) if text not in (None, "") else None
    except ValueError:
        return None


def relative_call(rows: list[dict], alarm_share: float) -> SystemCall:
    judged = [r.get("posture", "") for r in rows if r.get("posture") in ("超標", "正常")]
    if not judged:
        return SystemCall(None, None, 0)
    share = judged.count("超標") / len(judged)
    return SystemCall(share >= alarm_share, share, len(judged))


def absolute_call(rows: list[dict], thresholds: dict, alarm_share: float) -> SystemCall:
    over = total = 0
    for row in rows:
        if row.get("usable", "1") != "1":
            continue
        values = [(abs(v), thresholds[c]) for c in thresholds
                  if (v := _float(row.get(c))) is not None]
        if not values:
            continue
        total += 1
        over += any(v > limit for v, limit in values)
    if total == 0:
        return SystemCall(None, None, 0)
    share = over / total
    return SystemCall(share >= alarm_share, share, total)


# ---- 段落 ------------------------------------------------------------------

@dataclass
class SegmentResult:
    segment: str
    subject: str
    trip_id: str
    condition: str
    trial: str
    photos: int
    score: float | None
    kendall_share: float | None
    reference_score: float | None = None
    calls: dict = field(default_factory=dict)       # 方法 → SystemCall
    note: str = ""

    def drop(self, config: RatingConfig) -> float | None:
        if self.score is None or self.reference_score is None:
            return None
        diff = self.reference_score - self.score
        return diff if config.primary_item.higher_is_better else -diff

    def rating_bad(self, config: RatingConfig) -> bool | None:
        drop = self.drop(config)
        return None if drop is None else drop >= config.rating_drop_threshold - 1e-9

    def kendall_bad(self, config: RatingConfig) -> bool | None:
        if self.kendall_share is None:
            return None
        return self.kendall_share >= config.kendall_bad_share - 1e-9

    def truly_bad(self, config: RatingConfig) -> bool | None:
        rating, kendall = self.rating_bad(config), self.kendall_bad(config)
        if rating is None or kendall is None:
            return None
        return rating and kendall

    def outcome(self, method: str, config: RatingConfig) -> str | None:
        """有效、誤報、漏報、正確未警報；資料不足時 None。"""
        call = self.calls.get(method)
        bad = self.truly_bad(config)
        if call is None or call.alarm is None or bad is None:
            return None
        if call.alarm:
            return "有效" if bad else "誤報"
        return "漏報" if bad else "正確未警報"


def segment_results(ratings: RatingSet, key: dict[str, KeyRow], config: RatingConfig,
                    sessions_root: Path | None = None) -> list[SegmentResult]:
    per_photo: dict[str, list] = {}
    for rating in ratings.ratings:
        per_photo.setdefault(rating.blind_id, []).append(rating)

    segments: dict[str, list[KeyRow]] = {}
    for row in key.values():
        segments.setdefault(row.segment, []).append(row)

    results = []
    for segment, rows in sorted(segments.items()):
        first = rows[0]
        photo_scores, kendall_flags = [], []
        for row in rows:
            got = per_photo.get(row.blind_id, [])
            if not got:
                continue
            photo_scores.append(float(np.mean([r.scores[config.primary] for r in got])))
            kendall_flags.extend(config.kendall_photo_bad(list(r.kendall.values()))
                                 for r in got)
        result = SegmentResult(
            segment=segment, subject=first.subject, trip_id=first.trip_id,
            condition=first.condition, trial=first.trial, photos=len(photo_scores),
            score=float(np.mean(photo_scores)) if photo_scores else None,
            kendall_share=(sum(kendall_flags) / len(kendall_flags)) if kendall_flags else None,
        )
        csv_path = (Path(sessions_root) / segment if sessions_root is not None
                    else Path(first.segment_csv))
        if csv_path.is_file():
            frames = _read_rows(csv_path)
            result.calls["relative"] = relative_call(frames, config.alarm_share)
            result.calls["absolute"] = absolute_call(
                frames, config.absolute_thresholds_deg, config.alarm_share)
        else:
            result.note = f"找不到逐幀 CSV {csv_path}"
        if result.score is None:
            result.note = (result.note + "；" if result.note else "") + "沒有任何評分"
        results.append(result)

    # 同趟端正坐的評分（評分者平均），同一趟有好幾段端正坐時取平均。
    reference: dict[tuple[str, str], list[float]] = {}
    for result in results:
        if result.condition == config.reference_condition and result.score is not None:
            reference.setdefault((result.subject, result.trip_id), []).append(result.score)
    for result in results:
        scores = reference.get((result.subject, result.trip_id))
        if scores:
            result.reference_score = float(np.mean(scores))
        elif result.score is not None:
            result.note = (result.note + "；" if result.note else "") + \
                f"同趟沒有 {config.reference_condition} 的評分"
    return results


# ---- 每種判定方式的彙總 -----------------------------------------------------

@dataclass(frozen=True)
class MethodSummary:
    method: str
    classified: int
    alarms: int
    effective: int
    false_alarms: int
    misses: int
    truly_bad: int
    unclassified: int
    subjects: int

    @property
    def effective_rate(self) -> float | None:
        return self.effective / self.alarms if self.alarms else None

    @property
    def false_alarm_rate(self) -> float | None:
        return self.false_alarms / self.alarms if self.alarms else None

    @property
    def miss_rate(self) -> float | None:
        return self.misses / self.truly_bad if self.truly_bad else None


def summarise(results: list[SegmentResult], config: RatingConfig) -> dict[str, MethodSummary]:
    out = {}
    for method in METHODS:
        outcomes = [(r, r.outcome(method, config)) for r in results]
        done = [(r, o) for r, o in outcomes if o is not None]
        count = {k: sum(1 for _, o in done if o == k)
                 for k in ("有效", "誤報", "漏報", "正確未警報")}
        out[method] = MethodSummary(
            method=method, classified=len(done),
            alarms=count["有效"] + count["誤報"], effective=count["有效"],
            false_alarms=count["誤報"], misses=count["漏報"],
            truly_bad=count["有效"] + count["漏報"],
            unclassified=len(outcomes) - len(done),
            subjects=len({r.subject for r, _ in done}),
        )
    return out


@dataclass(frozen=True)
class MethodComparison:
    table: TwoByTwo                 # 列：相對、絕對；欄：有效、誤報
    paired_segments: int            # 兩種方式都分類得了的段落
    only_relative_correct: int      # McNemar 的 b
    only_absolute_correct: int      # McNemar 的 c
    mcnemar_p: float | None


def compare_methods(results: list[SegmentResult], config: RatingConfig,
                    summaries: dict[str, MethodSummary]) -> MethodComparison:
    rel, ab = summaries["relative"], summaries["absolute"]
    table = TwoByTwo(rel.effective, rel.false_alarms, ab.effective, ab.false_alarms)
    b = c = paired = 0
    for result in results:
        r, a = result.outcome("relative", config), result.outcome("absolute", config)
        if r is None or a is None:
            continue
        paired += 1
        r_ok, a_ok = r in ("有效", "正確未警報"), a in ("有效", "正確未警報")
        b += r_ok and not a_ok
        c += a_ok and not r_ok
    return MethodComparison(table, paired, b, c, mcnemar_exact(b, c))
