"""把評分者信度與有效警報分析排成文字與 CSV。回傳字串，由呼叫端決定印在哪。"""
from __future__ import annotations

import csv
import io

from .rating_alarm import (METHOD_NAMES, METHODS, MethodComparison, MethodSummary,
                           Reliability, SegmentResult)
from .rating_config import RatingConfig
from .rating_import import RatingSet
from .rating_stats import FEW_SUBJECTS, WIDE_INTERVAL, is_wide

NO_RATINGS = "沒有評分檔"


def _f(value, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _pct(value) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def _agreement_lines(title: str, agreements) -> list[str]:
    lines = [title]
    for a in agreements:
        ci = a.interval
        flag = ""
        if is_wide(ci, a.subjects):
            flag = "  ※區間很寬，點估計僅供參考"
        interval = ("區間算不出來" if ci.low is None
                    else f"95% CI {ci.low:.2f}～{ci.high:.2f}（以{ci.unit}重抽）")
        lines.append(f"  {a.item}：{a.method} = {_f(a.estimate)}，{interval}，"
                     f"{a.targets} 張照片、{a.subjects} 位受試者{flag}")
    return lines


def reliability_text(rel: Reliability) -> str:
    lines = ["== 評分者信度 =="]
    if len(rel.raters) < 2:
        lines.append(f"只有 {len(rel.raters)} 位評分者，算不了評分者間信度（至少要兩位）。")
        return "\n".join(lines)
    lines.append(f"評分者 {len(rel.raters)} 位：{', '.join(rel.raters)}。"
                 f"只用每位評分者都評過的照片。")
    lines += _agreement_lines("Kendall 檢核（二分）：", rel.kappas)
    lines += _agreement_lines("主觀評分，單一評分者（雙向隨機、絕對一致）：", rel.iccs_single)
    lines += _agreement_lines("主觀評分，評分者平均（分析用的是這個）：", rel.iccs_average)
    lines.append(f"區間是以受試者為單位的群集 bootstrap 百分位區間。受試者少於 "
                 f"{FEW_SUBJECTS} 位或區間寬度超過 {WIDE_INTERVAL} 時加註。")
    return "\n".join(lines)


def coverage_text(ratings: RatingSet, total: int) -> str:
    lines = ["== 評分檔 ==", f"讀了 {len(ratings.files)} 個檔、{len(ratings.ratings)} 筆評分，"
             f"盲評包共 {total} 張。"]
    for rater in ratings.raters:
        missing = ratings.missing.get(rater, [])
        if missing:
            shown = ", ".join(missing[:8]) + (" …" if len(missing) > 8 else "")
            lines.append(f"  缺漏：{rater} 還有 {len(missing)} 張沒評（{shown}）")
        else:
            lines.append(f"  {rater} 全部評完")
    return "\n".join(lines)


def _summary_row(s: MethodSummary) -> str:
    return (f"  {METHOD_NAMES[s.method]}：分類 {s.classified} 段（{s.subjects} 位受試者），"
            f"警報 {s.alarms}、有效 {s.effective}、誤報 {s.false_alarms}、"
            f"漏報 {s.misses}；有效警報率 {_pct(s.effective_rate)}、"
            f"誤報率 {_pct(s.false_alarm_rate)}、漏報率 {_pct(s.miss_rate)}"
            + (f"；{s.unclassified} 段資料不足未分類" if s.unclassified else ""))


def alarm_text(results: list[SegmentResult], summaries: dict[str, MethodSummary],
               comparison: MethodComparison, config: RatingConfig) -> str:
    thresholds = "、".join(f"{c} ±{v:g}°" for c, v in config.absolute_thresholds_deg.items())
    lines = [
        "== 有效警報（說明書表3）==",
        "統計單位：段落。同一位受試者的段落彼此不獨立，下面的檢定只能當描述。",
        f"系統警報 = 一段之中判超標的幀 ≥ {config.alarm_share:.0%}；"
        f"絕對標準用原始角度 {thresholds}。",
        f"真的不良 = 評分比同趟 {config.reference_condition} 低 ≥ "
        f"{config.rating_drop_threshold:g} 分，且 Kendall 判不良的比例 ≥ "
        f"{config.kendall_bad_share:.0%}。",
    ]
    lines += [_summary_row(summaries[m]) for m in METHODS]
    t = comparison.table
    lines += [
        "",
        "判定方式 × 警報結果（2×2）：",
        "              有效    誤報",
        f"  相對基準線  {t.a:>4}    {t.b:>4}",
        f"  絕對標準    {t.c:>4}    {t.d:>4}",
    ]
    chi, p, smallest = t.chi_square()
    fisher = t.fisher_p()
    if chi is None:
        lines.append("  卡方：有一列或一欄是 0，算不了。")
    else:
        warn = "（最小期望次數 < 5，以 Fisher 為準）" if smallest < 5 else ""
        lines.append(f"  Pearson 卡方 = {chi:.2f}，p = {p:.3f}{warn}")
    if fisher is not None:
        lines.append(f"  Fisher 精確檢定 p = {fisher:.3f}")
    lines.append("  注意：兩列是同一批段落用兩種方式判，不是獨立的兩組，卡方的前提不成立；"
                 "配對比較看下面的 McNemar。")
    lines.append(f"  配對（兩種方式都分類得了的 {comparison.paired_segments} 段）："
                 f"只有相對基準線判對 {comparison.only_relative_correct} 段、"
                 f"只有絕對標準判對 {comparison.only_absolute_correct} 段，"
                 f"McNemar 精確 p = {_f(comparison.mcnemar_p, 3)}")

    per_subject: dict[str, int] = {}
    for r in results:
        per_subject[r.subject] = per_subject.get(r.subject, 0) + 1
    lines.append("每位受試者的段落數：" + "、".join(
        f"{s} {n}" for s, n in sorted(per_subject.items())))
    notes = [f"  {r.segment}：{r.note}" for r in results if r.note]
    if notes:
        lines.append("未能分類的原因：")
        lines += notes
    return "\n".join(lines)


SEGMENT_COLUMNS = ("segment", "subject", "trip_id", "condition", "trial", "photos",
                   "score_mean", "reference_score", "rating_drop", "rating_bad",
                   "kendall_bad_share", "kendall_bad", "truly_bad",
                   "relative_over_share", "relative_alarm", "relative_outcome",
                   "absolute_over_share", "absolute_alarm", "absolute_outcome", "note")


def _cell(value, digits: int = 3):
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return round(value, digits)
    return value


def segments_csv(results: list[SegmentResult], config: RatingConfig) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(SEGMENT_COLUMNS)
    for r in results:
        rel, ab = r.calls.get("relative"), r.calls.get("absolute")
        writer.writerow([_cell(v) for v in (
            r.segment, r.subject, r.trip_id, r.condition, r.trial, r.photos,
            r.score, r.reference_score, r.drop(config), r.rating_bad(config),
            r.kendall_share, r.kendall_bad(config), r.truly_bad(config),
            None if rel is None else rel.share, None if rel is None else rel.alarm,
            r.outcome("relative", config),
            None if ab is None else ab.share, None if ab is None else ab.alarm,
            r.outcome("absolute", config), r.note,
        )])
    return buffer.getvalue()
