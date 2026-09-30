"""把 Cohort 排成可以讀的報告。

排版跟 `session_report` 一樣是對齊的純文字：同一個工具的兩個子指令不該一個
給 Markdown、一個給文字，讀的人要換一種看法。要進報告的表格從 CSV 匯入，
那本來就是 CSV 的用途。

欄寬用 geometry.terminal 的顯示寬度，中日韓字元佔兩欄，用 len() 會歪掉。

不用符號標記。圖例要對照、複製貼上會掉字、而且一個符號能表達的事情，
一個詞也能：略過率偏高的那一列直接在數字後面寫「偏高」。

排版與統計分開，理由與 `session_report` 相同：排版的斷言只能確認字串裡有某段
文字，那擋不住算錯。
"""
from __future__ import annotations

import csv
from io import StringIO

from .cohort import REJECTION_LIMIT, Cohort, Separation, session_error_deg
from .judgement import Posture
from .terminal import cell, display_width

_SESSION_COLUMNS = (
    "subject", "condition", "trial", "file",
    "frames", "usable", "rejection_rate",
    "theta_ca_deg", "theta_ca_error_deg", "theta_ca_single_frame_std_deg",
    "theta_sym_deg", "distance_mm", "azimuth_deg", "over_share",
)

# 逐段表每一欄的寬度。手寫而不是照內容算，因為量測之間的數字量級差不多，
# 固定寬度在不同批資料之間才對得起來。
_COLUMNS = (
    ("受試者", 12), ("姿勢", 12), ("次", 4), ("θ_CA", 20), ("單幀散佈", 12),
    ("θ_sym", 10), ("可用", 12), ("略過", 12), ("超標", 8),
    ("距離", 10), ("方位角", 8),
)


def _fmt(value, digits: int = 2, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.{digits}f}{suffix}"


def _percent(value) -> str:
    """比例寫成百分比。None 是「沒有判定資料」，與 0% 是兩回事。"""
    return "—" if value is None else f"{value * 100:.0f}%"


def _heading(text: str, width: int = 58) -> str:
    """區段標題，與 session_report 同一個樣式。

    橫線的長度照顯示寬度算。用 `len(text) * 2` 估的話，中英混排的標題
    （例如「逐受試者的 θ_CA」）會算出比實際寬的值，線就短一截。
    """
    return f"── {text} " + "─" * max(0, width - 4 - display_width(text))


def _table(header: tuple, rows: list[list[str]]) -> list[str]:
    """對齊的表格。header 是 (標題, 寬度) 的序列。"""
    out = ["".join(cell(name, width) for name, width in header)]
    for row in rows:
        out.append("".join(cell(text, width)
                           for text, (_, width) in zip(row, header)))
    return out


def session_rows(cohort: Cohort) -> list[dict]:
    """逐 session 的原始數字。文字報告與 CSV 共用這一份。"""
    rows = []
    for summary in cohort.sessions:
        corrected = summary.angle("θ_CA")
        sym = summary.angle("θ_sym")
        judgement = summary.judgement
        over = None if judgement is None else judgement.share(Posture.OVER.value)
        rows.append({
            "subject": summary.subject_key,
            "condition": summary.condition_key,
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


def sessions_table(rows: list[dict]) -> str:
    """逐段的表。"""
    if not rows:
        return "沒有讀到任何資料。"
    body = []
    for r in rows:
        # 先算好每一格再組列。塞進一行裡的條件運算式在十一欄的寬度下讀不出來，
        # 而錯位一格在成品上看不出來。
        over_limit = r["rejection_rate"] > REJECTION_LIMIT
        body.append([
            r["subject"],
            r["condition"],
            "—" if r["trial"] is None else str(r["trial"]),
            f"{_fmt(r['theta_ca_deg'], 2, '°')} ± {_fmt(r['theta_ca_error_deg'], 2, '°')}",
            f"±{_fmt(r['theta_ca_single_frame_std_deg'], 1, '°')}",
            _fmt(r["theta_sym_deg"], 2, "°"),
            f"{r['usable']}/{r['frames']}",
            _percent(r["rejection_rate"]) + (" 偏高" if over_limit else ""),
            _percent(r["over_share"]),
            _fmt(r["distance_mm"], 0, "mm"),
            _fmt(r["azimuth_deg"], 0, "°"),
        ])
    out = _table(_COLUMNS, body)
    # 說明只在真的有那種列時才印。沒有東西被標卻印一行解釋，讀的人會回頭
    # 找它在哪裡。
    if any(r["rejection_rate"] > REJECTION_LIMIT for r in rows):
        out.append("")
        out.append(f"  標「偏高」的是略過率超過 {REJECTION_LIMIT * 100:.0f}% 的段落。"
                   f"留下來的幀代表不了整段姿勢，這些不列入下面的統計。")
    return "\n".join(out)


def subjects_table(cohort: Cohort) -> str:
    """逐受試者、逐姿勢的代表值。"""
    conditions = cohort.conditions
    if not conditions:
        return ""
    header = (("受試者", 12),) + tuple((c, 14) for c in conditions)
    body = [
        [subject] + [_fmt(cohort.result(subject, c).theta_ca_deg, 2, "°")
                     for c in conditions]
        for subject in cohort.subjects
    ]
    return "\n".join(_table(header, body))


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
        lines.append(f"  資料裡沒有這些姿勢：{'、'.join(missing)}。"
                     f"目前有的是：{'、'.join(cohort.conditions) or '（無）'}。"
                     f"名稱要與 study --conditions 用的一致。")
    if unlabelled:
        lines.append(
            f"  有 {unlabelled} 段沒有記錄姿勢條件，那些是在 CSV 加上 condition "
            f"欄之前量的。它們照樣列在上面的逐段表裡，但不能參與比較，"
            f"因為事後補標會變成猜，而猜錯的地方不會有任何痕跡。"
        )
    if not missing and not unlabelled:
        lines.append(f"  兩種姿勢都有資料，但沒有受試者同時具備兩者的可用段落。"
                     f"略過率超過 {REJECTION_LIMIT * 100:.0f}% 的段落不列入統計，"
                     f"檢查上面標了「偏高」的那幾列。")
    return "\n".join(lines)


def separation_text(separation: Separation, cohort: Cohort) -> str:
    """兩種姿勢分不分得開。這是報告的主要結果。"""
    if not separation.per_subject:
        return _why_no_separation(separation, cohort)

    out = [f"{separation.baseline_condition} 相對 {separation.other}", ""]
    out += _table((("受試者", 12), ("差距", 12)),
                  [[subject, f"{value:+.2f}°"]
                   for subject, value in separation.per_subject.items()])

    # 只需要一欄的最大值，不必重建整份 rows（那會把每個 session 的
    # standard_error 重算一次）。
    spreads = [angle.single_frame_std for summary in cohort.sessions
               if (angle := summary.angle("θ_CA")) is not None]
    comparison = separation.comparison(max(spreads) if spreads else None)

    out.append("")
    if comparison.error is None:
        out.append(f"平均差距 {comparison.difference:+.2f}°，"
                   f"但只有 {separation.subjects} 位受試者，算不出跨受試者的誤差。"
                   f"一個樣本沒有散佈可言。")
    else:
        # describe() 與單一 session 的報表共用，兩邊的這一句永遠一致。
        out.append(f"平均差距 {comparison.describe()}，n = {separation.subjects}。")
        out.append("")
        out.append("這裡的誤差是跨受試者的：分母是人數，不是幀數。"
                   "θ_CA 的相鄰幀自相關是 0.73，拿幀數當分母會把信賴水準"
                   "講得比實際高。")

    if comparison.spread_ratio is not None:
        out.append("")
        out.append(f"差距 / 最大單幀散佈 = {comparison.spread_ratio:.1f} 倍。"
                   f"只報標準誤差會高估可靠度，因為那個分母是平均值的誤差，"
                   f"而實際比較面對的是姿勢本身的變異。")
    return "\n".join(out)


def to_text(cohort: Cohort, separation: Separation | None,
            rows: list[dict] | None = None) -> str:
    rows = session_rows(cohort) if rows is None else rows
    parts = [
        "量測結果彙整",
        "=" * 58,
        f"共 {len(cohort.sessions)} 段量測，{len(cohort.subjects)} 位受試者。",
        "",
        _heading("逐段"),
        sessions_table(rows),
    ]
    subjects = subjects_table(cohort)
    if subjects:
        parts += ["", _heading("逐受試者的 θ_CA"), subjects]
    if separation is not None:
        parts += ["", _heading("姿勢之間的差距"),
                  separation_text(separation, cohort)]
    return "\n".join(parts) + "\n"


def to_csv(rows: list[dict]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=_SESSION_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({
            key: ("" if row[key] is None else row[key]) for key in _SESSION_COLUMNS
        })
    return buffer.getvalue()
