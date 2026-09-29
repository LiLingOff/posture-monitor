"""把 Cohort 排成可以直接貼進報告的表。

輸出 Markdown 與 CSV 兩種。Markdown 是給報告用的，CSV 是要再算的時候用的。
兩者的數字一致，排版分開寫，因為 Markdown 要對齊、CSV 不要。

排版與統計分開，理由與 `session_report` 相同：排版的斷言只能確認字串裡有某段
文字，那擋不住算錯。
"""
from __future__ import annotations

import csv
from io import StringIO

from .cohort import Cohort, Separation, UNLABELLED, session_error_deg

_SESSION_COLUMNS = (
    "subject", "condition", "trial", "file",
    "frames", "usable", "rejection_rate",
    "theta_ca_deg", "theta_ca_error_deg", "theta_ca_single_frame_std_deg",
    "theta_sym_deg", "distance_mm", "azimuth_deg", "over_share",
)


def _fmt(value, digits: int = 2, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.{digits}f}{suffix}"


def _percent(value) -> str:
    """比例寫成百分比。None 是「沒有判定資料」，與 0% 是兩回事。"""
    return "—" if value is None else f"{value * 100:.0f}%"


def _angle(summary, key: str):
    return summary.angles.get(key)


def session_rows(cohort: Cohort) -> list[dict]:
    """逐 session 的原始數字。Markdown 與 CSV 共用這一份。"""
    rows = []
    for summary in cohort.sessions:
        corrected = _angle(summary, "θ_CA 扣基準") or _angle(summary, "θ_CA 原始")
        sym = _angle(summary, "θ_sym 扣基準") or _angle(summary, "θ_sym 原始")
        judgement = summary.judgement
        over = (judgement.share("超標") if judgement is not None else None)
        rows.append({
            "subject": summary.subject or "?",
            "condition": summary.condition or UNLABELLED,
            "trial": summary.trial,
            "file": summary.path.name,
            "frames": summary.frames,
            "usable": summary.usable,
            "rejection_rate": summary.rejection_rate,
            "theta_ca_deg": None if corrected is None else corrected.mean,
            "theta_ca_error_deg": session_error_deg(summary),
            "theta_ca_single_frame_std_deg":
                None if corrected is None else corrected.single_frame_std,
            "theta_sym_deg": None if sym is None else sym.mean,
            "distance_mm": _column(summary, "距離 mm"),
            "azimuth_deg": _column(summary, "方位角 °"),
            "over_share": over,
        })
    return rows


def _column(summary, name: str):
    column = summary.columns.get(name)
    return None if column is None else column.mean


def sessions_markdown(cohort: Cohort) -> str:
    rows = session_rows(cohort)
    if not rows:
        return "沒有讀到任何資料。"
    out = [
        "| 受試者 | 姿勢 | 次 | θ_CA | 單幀散佈 | θ_sym | 可用 | 略過 | 超標 | 距離 | 方位角 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        # 先算好每一格再組字串。塞進 f-string 裡的條件運算式在這種寬度下
        # 讀不出來，而表格有十一欄，錯位一格在成品上看不出來。
        flag = " ⚠" if r["rejection_rate"] > 0.15 else ""
        trial = "—" if r["trial"] is None else str(r["trial"])
        angle = _fmt(r["theta_ca_deg"], 2, "°")
        error = _fmt(r["theta_ca_error_deg"], 2, "°")
        over = _percent(r["over_share"])
        out.append(
            f"| {r['subject']} | {r['condition']} | {trial} "
            f"| {angle} ± {error} "
            f"| ±{_fmt(r['theta_ca_single_frame_std_deg'], 1, '°')} "
            f"| {_fmt(r['theta_sym_deg'], 2, '°')} "
            f"| {r['usable']}/{r['frames']} "
            f"| {r['rejection_rate'] * 100:.0f}%{flag} "
            f"| {over} "
            f"| {_fmt(r['distance_mm'], 0, 'mm')} "
            f"| {_fmt(r['azimuth_deg'], 0, '°')} |"
        )
    # 圖例只在真的有標記時才印。沒有東西被標卻印一行解釋，讀的人會回頭找
    # 那個符號在哪裡。
    if any(r["rejection_rate"] > 0.15 for r in rows):
        out.append("")
        out.append("⚠ 是略過率超過 15% 的 session。留下來的幀代表不了整段姿勢，"
                   "這些不列入下面的統計。")
    return "\n".join(out)


def subjects_markdown(cohort: Cohort) -> str:
    """逐受試者、逐姿勢的代表值。"""
    conditions = cohort.conditions
    if not conditions:
        return ""
    header = "| 受試者 | " + " | ".join(conditions) + " |"
    divider = "|---" * (len(conditions) + 1) + "|"
    out = [header, divider]
    for subject in cohort.subjects:
        cells = []
        for condition in conditions:
            result = cohort.result(subject, condition)
            cells.append(_fmt(result.theta_ca_deg, 2, "°"))
        out.append(f"| {subject} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def _why_no_separation(separation: Separation, cohort: Cohort) -> str:
    """算不出差距時說清楚缺的是什麼。

    只說「沒有可用資料」的話，使用者分不出是量錯了、名字打錯了、還是那批
    檔案根本沒記姿勢條件，而這三件事要做的補救完全不同。
    """
    wanted = {separation.baseline_condition, separation.other}
    present = set(cohort.conditions)
    unlabelled = sum(1 for s in cohort.sessions if s.condition is None)

    lines = [f"算不出「{separation.baseline_condition}」與「{separation.other}」的差距。"]
    missing = sorted(wanted - present)
    if missing:
        lines.append(f"資料裡沒有這些姿勢：{'、'.join(missing)}。"
                     f"目前有的是：{'、'.join(cohort.conditions) or '（無）'}。"
                     f"名稱要與 `study --conditions` 用的一致。")
    if unlabelled:
        lines.append(
            f"有 {unlabelled} 段沒有記錄姿勢條件，那些是在 CSV 加上 `condition` "
            f"欄之前量的。它們照樣列在上面的逐段表裡，但不能參與比較，"
            f"因為事後補標會變成猜，而猜錯的地方不會有任何痕跡。"
        )
    if not missing and not unlabelled:
        lines.append("兩種姿勢都有資料，但沒有受試者同時具備兩者的可用段落。"
                     "略過率超過 15% 的段落不列入統計，檢查上面標了 ⚠ 的那幾列。")
    return "\n\n".join(lines)


def separation_markdown(separation: Separation, cohort: Cohort) -> str:
    """兩種姿勢分不分得開。這是報告的主要結果。"""
    if not separation.per_subject:
        return _why_no_separation(separation, cohort)

    out = [
        f"### {separation.baseline_condition} → {separation.other}",
        "",
        "| 受試者 | 差距 |",
        "|---|---|",
    ]
    for subject, value in separation.per_subject.items():
        out.append(f"| {subject} | {value:+.2f}° |")

    mean, error, sigma = separation.mean_deg, separation.error_deg, separation.sigma
    out.append("")
    if error is None:
        out.append(f"**平均差距 {mean:+.2f}°**，但只有 {separation.subjects} 位受試者，"
                   f"算不出跨受試者的誤差。一個樣本沒有散佈可言。")
    else:
        shown = f"**平均差距 {mean:+.2f}° ± {error:.2f}°**"
        if sigma is not None:
            shown += f"（{sigma:.1f} 個標準誤差）"
        out.append(shown + f"，n = {separation.subjects}。")
        out.append("")
        out.append("這裡的誤差是**跨受試者**的：分母是人數，不是幀數。"
                   "θ_CA 的相鄰幀自相關是 0.73，拿幀數當分母會把信賴水準"
                   "講得比實際高。")

    spreads = [r["theta_ca_single_frame_std_deg"] for r in session_rows(cohort)
               if r["theta_ca_single_frame_std_deg"] is not None]
    if mean is not None and spreads:
        out.append("")
        out.append(f"差距 / 最大單幀散佈 = {abs(mean) / max(spreads):.1f} 倍。"
                   f"只報標準誤差會高估可靠度，因為那個分母是平均值的誤差，"
                   f"而實際比較面對的是姿勢本身的變異。")
    return "\n".join(out)


def to_markdown(cohort: Cohort, separation: Separation | None) -> str:
    parts = [
        "# 量測結果彙整",
        "",
        f"共 {len(cohort.sessions)} 段量測，{len(cohort.subjects)} 位受試者。",
        "",
        "## 逐段",
        "",
        sessions_markdown(cohort),
    ]
    subjects = subjects_markdown(cohort)
    if subjects:
        parts += ["", "## 逐受試者的 θ_CA", "", subjects]
    if separation is not None:
        parts += ["", "## 姿勢之間的差距", "", separation_markdown(separation, cohort)]
    return "\n".join(parts) + "\n"


def to_csv(cohort: Cohort) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=_SESSION_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in session_rows(cohort):
        writer.writerow({
            key: ("" if row[key] is None else row[key]) for key in _SESSION_COLUMNS
        })
    return buffer.getvalue()
