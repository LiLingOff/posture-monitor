"""把 `trips` 的結果排成文字與 CSV。`analyse` 與 `cohort` 共用。

排版與統計分開，理由與 `session_report` 相同。回傳字串，由呼叫端決定印在哪、
寫到哪個檔案。
"""
from __future__ import annotations

import csv
from io import StringIO

from .terminal import cell, display_width, with_error
from .trips import (AXIS_LIMITS, FORWARD_QUANTITIES, NATURAL_QUANTITIES,
                    SENSITIVITY_LIMITS, UNPAIRED, VOID, PairedRow, is_natural,
                    paired_differences, ratio_by_group, ratio_rows, sensitivity,
                    tilt_by_angle, tilt_rows, tilt_symmetry)


def _table(header, rows) -> list[str]:
    widths = [w for _, w in header]
    out = ["".join(cell(name, w) for name, w in header).rstrip()]
    out.append("-" * display_width(out[0]))
    for row in rows:
        out.append("".join(cell(text, w) for text, w in zip(row, widths)).rstrip())
    return out


def _fmt(value, digits=2, suffix="°", signed=True) -> str:
    if value is None:
        return "—"
    return f"{value:+.{digits}f}{suffix}" if signed else f"{value:.{digits}f}{suffix}"


def _pct(value) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def _meta(row_summary, key) -> str:
    return row_summary.meta.get(key, "")


def _limits_line() -> str:
    return ("略過率上限 θ_CA {:.0f}%、θ_sym {:.0f}%、θ_KA {:.0f}%（沿用 θ_CA）；超過的標"
            "「附註納入」照樣進主要結果，轉身（方位角漂超過 10°）一律作廢。").format(
        AXIS_LIMITS["θ_CA"] * 100, AXIS_LIMITS["θ_sym"] * 100, AXIS_LIMITS["θ_KA"] * 100)


# ---- 配對差值 ----------------------------------------------------------

def paired_text(rows: list[PairedRow], quantities, title_target: str,
                independence_note: str = "") -> str:
    if not rows:
        return ""
    header = [("受試者", 12), ("桌面", 9), ("墊高", 6), ("桌減肘", 8), ("段", 14)]
    header += [(q.label, 22) for q in quantities]
    body = []
    for row in rows:
        s = row.target
        head = [row.subject, row.desk_level or "—", _meta(s, "lift_cm") or "—",
                _meta(s, "desk_offset_cm") or "—", f"{s.condition}#{s.trial or '—'}"]
        if not row.paired:
            body.append(head + [f"{UNPAIRED}：{row.reason}"])
            continue
        cells = []
        for q in quantities:
            d = row.differences[q.label]
            text = (d.note if d.value is None
                    else with_error(d.value, d.error, q.unit, 1 if q.unit == "mm" else 2))
            if d.status != "納入":
                text += f" {d.status}"
            cells.append(text)
        body.append(head + cells)
    out = [f"{title_target}減同一趟第一段 upright 的原始值（統計單位：受試者 × 桌面條件）", ""]
    out += _table(header, body)
    unpaired = [r for r in rows if not r.paired]
    if unpaired:
        out += ["", f"  {len(unpaired)} 段找不到同趟的端正坐，列在上表、不進下面的平均。"]
    turned = [r for r in rows if r.paired and any(d.status == VOID
                                                   for d in r.differences.values())]
    if turned:
        out += [f"  {len(turned)} 段相對同趟參考方位角轉了 10° 以上，作廢。"]
    out += ["", "  每列的誤差是兩段各自的批次平均誤差合成，不是 std/√N。", "  " + _limits_line()]
    if independence_note:
        out += ["  " + independence_note]
    return "\n".join(out)


def sensitivity_text(rows: list[PairedRow], quantities) -> str:
    """每個桌面條件、每個量：主要結果與 15%、20% 兩套門檻的跨受試者平均。"""
    paired = [r for r in rows if r.paired]
    if not paired:
        return ""
    levels = sorted({r.desk_level for r in paired})
    header = [("桌面", 9), ("量", 8), ("主要（附註納入）", 24)]
    header += [(f"≤{c * 100:.0f}%", 24) for c in SENSITIVITY_LIMITS]
    body = []
    for level in levels:
        for q in quantities:
            cells = [level or "—", q.label]
            for cutoff in (None, *SENSITIVITY_LIMITS):
                g = sensitivity(paired, q.label, level, cutoff)
                digits = 1 if q.unit == "mm" else 2
                cells.append("—" if g.mean is None else
                             f"{with_error(g.mean, g.error, q.unit, digits)}（n={g.n}）")
            body.append(cells)
    out = ["跨受試者平均（分母是人數），以及略過率門檻的敏感度分析", ""]
    out += _table(header, body)
    out += ["", "  ≤15%、≤20% 兩欄把配對兩段任一段超過該門檻的列排除；只有一個人時沒有誤差。"]
    return "\n".join(out)


def paired_csv(rows: list[PairedRow], quantities) -> str:
    fields = ["subject", "desk_level", "lift_cm", "desk_offset_cm", "trip_id",
              "condition", "trial", "reference_trial", "file", "reference_file"]
    for q in quantities:
        base = q.column
        fields += [base, base + "_error",
                   f"{base}_rejection_rate", f"{base}_inclusion", f"{base}_note"]
    fields += ["turned_deg", "status", "reason"]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        s, r = row.target, row.reference
        out = {
            "subject": row.subject, "desk_level": row.desk_level,
            "lift_cm": _meta(s, "lift_cm"), "desk_offset_cm": _meta(s, "desk_offset_cm"),
            "trip_id": _meta(s, "trip_id"), "condition": s.condition or "",
            "trial": s.trial if s.trial is not None else "",
            "reference_trial": "" if r is None or r.trial is None else r.trial,
            "file": s.path.name, "reference_file": "" if r is None else r.path.name,
            "turned_deg": "" if row.turned_deg is None else round(row.turned_deg, 2),
            "status": "paired" if row.paired else "unpaired", "reason": row.reason,
        }
        for q in quantities:
            d = row.differences.get(q.label)
            base = q.column
            out[base] = "" if d is None or d.value is None else round(d.value, 4)
            out[base + "_error"] = "" if d is None or d.error is None else round(d.error, 4)
            out[f"{base}_rejection_rate"] = "" if d is None else round(max(d.rates), 4)
            out[f"{base}_inclusion"] = UNPAIRED if d is None else d.status
            out[f"{base}_note"] = "" if d is None else d.note
        writer.writerow(out)
    return buffer.getvalue()


# ---- 側傾 --------------------------------------------------------------

def tilt_text(rows) -> str:
    if not rows:
        return ""
    header = [("受試者", 12), ("段", 14), ("目標", 7), ("量角器", 8), ("θ_sym", 20),
              ("來源", 14), ("誤差", 10), ("符號", 6), ("略過", 10)]
    body = []
    for row in rows:
        s = row.summary
        sign = {True: "對", False: "反", None: "—"}[row.sign_ok]
        body.append([
            s.subject_key, f"{s.condition}#{s.trial or '—'}",
            _fmt(row.target_deg, 0, "°", signed=False),
            _fmt(row.measured_deg, 1, "°", signed=False),
            with_error(row.theta_sym_deg, row.theta_sym_error_deg),
            row.source, _fmt(row.error_deg), sign,
            _pct(row.rate) + ("" if row.status == "納入" else f" {row.status}"),
        ])
    out = ["θ_sym 與量角器實測值（誤差 = θ_sym − 帶號實測值）", ""]
    out += _table(header, body)
    out += ["", "  符號慣例：θ_sym 右肩較高為正。往左側彎時右肩較高，預期為正；往右側彎預期為負。",
            "  量角器量的是上身傾斜角，θ_sym 量的是肩線傾斜角，兩者本來就有小差異。"]
    missing = sum(1 for r in rows if r.measured_deg is None)
    if missing:
        out.append(f"  {missing} 段沒有量角器實測值（measured_angle_deg），算不出誤差。")

    by_angle = tilt_by_angle(rows)
    if by_angle:
        strict = {c: tilt_by_angle(rows, c) for c in SENSITIVITY_LIMITS}

        def cell_text(g):
            return "—" if g is None or g.mean is None else (
                f"{with_error(g.mean, g.error)}（n={g.n}）")

        out += ["", "依目標角度（誤差會不會隨角度變大；分母是人數），附略過率敏感度分析", ""]
        out += _table([("側", 4), ("目標", 8), ("主要（附註納入）", 24)]
                      + [(f"≤{c * 100:.0f}%", 24) for c in SENSITIVITY_LIMITS],
                      [[side, _fmt(target, 0, "°", signed=False), cell_text(g)]
                       + [cell_text(strict[c].get((side, target))) for c in SENSITIVITY_LIMITS]
                       for (side, target), g in by_angle.items()])
    symmetry = tilt_symmetry(rows)
    if symmetry:
        out += ["", "左右是否對稱（|θ_sym| − 實測值，兩邊接近才算對稱）", ""]
        out += _table([("受試者", 12), ("左傾", 10), ("右傾", 10), ("左 − 右", 10)],
                      [[subject, _fmt(left), _fmt(right),
                        _fmt(None if left is None or right is None else left - right)]
                       for subject, (left, right) in symmetry.items()])
    return "\n".join(out)


def tilt_csv(rows) -> str:
    fields = ["subject", "trip_id", "desk_level", "condition", "trial", "file",
              "target_angle_deg", "measured_angle_deg", "expected_theta_sym_deg",
              "theta_sym_deg", "theta_sym_error_deg", "theta_sym_source",
              "error_deg", "magnitude_error_deg", "sign_ok",
              "theta_sym_rejection_rate", "inclusion", "turned_deg"]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()

    def r(value, digits=4):
        return "" if value is None else round(value, digits)

    for row in rows:
        s = row.summary
        writer.writerow({
            "subject": s.subject_key, "trip_id": _meta(s, "trip_id"),
            "desk_level": _meta(s, "desk_level"), "condition": s.condition,
            "trial": "" if s.trial is None else s.trial, "file": s.path.name,
            "target_angle_deg": r(row.target_deg), "measured_angle_deg": r(row.measured_deg),
            "expected_theta_sym_deg": r(row.expected_deg),
            "theta_sym_deg": r(row.theta_sym_deg),
            "theta_sym_error_deg": r(row.theta_sym_error_deg),
            "theta_sym_source": row.source, "error_deg": r(row.error_deg),
            "magnitude_error_deg": r(row.magnitude_error_deg),
            "sign_ok": "" if row.sign_ok is None else int(row.sign_ok),
            "theta_sym_rejection_rate": r(row.rate), "inclusion": row.status,
            "turned_deg": r(row.turned_deg, 2),
        })
    return buffer.getvalue()


# ---- 表4 ---------------------------------------------------------------

def ratio_text(rows, groups=None) -> str:
    if not rows:
        return ""
    header = [("受試者", 12), ("段", 16), ("目標", 6), ("實際方位角", 12), ("幀數", 6),
              ("3D θ_CA", 10), ("2D θ_CA", 10), ("tan比值", 18), ("sinφ/sin45°", 12),
              ("差", 8)]
    body = []
    for row in rows:
        s = row.summary
        ratio = "—" if row.ratio is None else with_error(row.ratio, row.ratio_error, "", 3,
                                                         signed=False)
        if row.unstable:
            ratio += " 不穩"
        body.append([
            s.subject_key, f"{s.condition}#{s.trial or '—'}",
            _fmt(row.target_deg, 0, "°", signed=False),
            _fmt(row.azimuth_deg, 1, "°", signed=False), str(row.frames),
            _fmt(row.theta_3d_deg), _fmt(row.theta_2d_deg), ratio,
            f"{row.predicted:.3f}", _fmt(row.deviation, 3, ""),
        ])
    out = ["表4：tan θ_2D / tan θ_3D 對理論值 sin φ / sin 45°（φ 是實際算出的方位角）", ""]
    out += _table(header, body)
    out += ["", "  看的是比值，不是絕對角度，三段姿勢稍有不同也不影響。",
            "  3D θ_CA 平均的絕對值小於 5° 的段落比值會被雜訊放大，標「不穩」、不進分組平均。",
            "  相機有俯仰角時即使 φ = 45° 也不再無偏，這一項沒有建模。"]
    grouped = ratio_by_group(rows, groups)
    if grouped:
        out += ["", "依身高層與目標方位角（比值 − 理論值；分母是人數）", ""]
        out += _table([("身高層", 10), ("目標", 6), ("比值 − 理論值", 22),
                       ("實際方位角", 12), ("n", 4)],
                      [[group, _fmt(target, 0, "°", signed=False),
                        with_error(dev.mean, dev.error, "", 3),
                        _fmt(az.mean, 1, "°", signed=False), str(dev.n)]
                       for (group, target), (dev, az) in grouped.items()])
        if groups is None:
            out.append("  沒有給 --subject-groups，全部歸「未分組」。")
    return "\n".join(out)


def ratio_csv(rows, groups=None) -> str:
    fields = ["subject", "group", "trip_id", "condition", "trial", "file",
              "azimuth_target_deg", "azimuth_deg", "frames", "theta_3d_deg",
              "theta_2d_deg", "tan_ratio", "tan_ratio_error", "predicted_ratio",
              "deviation", "unstable"]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()

    def r(value, digits=4):
        return "" if value is None else round(value, digits)

    for row in rows:
        s = row.summary
        writer.writerow({
            "subject": s.subject_key, "group": (groups or {}).get(s.subject_key, ""),
            "trip_id": _meta(s, "trip_id"), "condition": s.condition,
            "trial": "" if s.trial is None else s.trial, "file": s.path.name,
            "azimuth_target_deg": r(row.target_deg), "azimuth_deg": r(row.azimuth_deg, 2),
            "frames": row.frames, "theta_3d_deg": r(row.theta_3d_deg),
            "theta_2d_deg": r(row.theta_2d_deg), "tan_ratio": r(row.ratio),
            "tan_ratio_error": r(row.ratio_error), "predicted_ratio": r(row.predicted),
            "deviation": r(row.deviation), "unstable": int(row.unstable),
        })
    return buffer.getvalue()


# ---- 組合 --------------------------------------------------------------

_FORWARD_NOTE = ("軀幹整個往前倒時肩高與 θ_KA 會一起動，這兩欄不是兩份獨立證據。"
                 "肩高往上為正，負值代表肩膀下沉。舊標定的 1.1mm 與 58.0mm 已作廢，不要引用。")


def _is_forward(condition: str) -> bool:
    return condition == "forward"


def trip_sections(summaries, groups=None) -> list[tuple[str, str]]:
    """(標題, 內文) 的清單，沒有資料的部分不列。"""
    out = []
    natural = paired_differences(summaries, is_natural, NATURAL_QUANTITIES)
    if natural:
        out.append(("自然坐配對差值", paired_text(natural, NATURAL_QUANTITIES, "自然坐")))
        sens = sensitivity_text(natural, NATURAL_QUANTITIES)
        if sens:
            out.append(("自然坐：跨受試者平均與敏感度分析", sens))
    forward = paired_differences(summaries, _is_forward, FORWARD_QUANTITIES)
    if any(r.paired for r in forward):
        out.append(("軀幹前傾：肩高與 θ_KA", paired_text(
            forward, FORWARD_QUANTITIES, "軀幹前傾", _FORWARD_NOTE)))
        sens = sensitivity_text(forward, FORWARD_QUANTITIES)
        if sens:
            out.append(("軀幹前傾：跨受試者平均與敏感度分析", sens))
    tilts = tilt_rows(summaries)
    if tilts:
        out.append(("側傾：θ_sym 對量角器", tilt_text(tilts)))
    ratios = ratio_rows(summaries)
    if ratios:
        out.append(("表4：方位角比值", ratio_text(ratios, groups)))
    return out


def trip_csvs(summaries, groups=None) -> dict[str, str]:
    """檔名 → CSV 內容。沒有資料的不產生。"""
    out = {}
    natural = paired_differences(summaries, is_natural, NATURAL_QUANTITIES)
    if natural:
        out["paired_natural.csv"] = paired_csv(natural, NATURAL_QUANTITIES)
    forward = paired_differences(summaries, _is_forward, FORWARD_QUANTITIES)
    if any(r.paired for r in forward):
        out["forward_shoulder.csv"] = paired_csv(forward, FORWARD_QUANTITIES)
    tilts = tilt_rows(summaries)
    if tilts:
        out["tilt.csv"] = tilt_csv(tilts)
    ratios = ratio_rows(summaries)
    if ratios:
        out["azimuth_ratio.csv"] = ratio_csv(ratios, groups)
    return out
