"""把 SessionSummary 排成可以讀的報告。

分析與排版分開，因為數字要能單獨測試。排版的斷言只能確認字串裡有某段文字，
那擋不住算錯。

欄寬用 geometry.terminal 的顯示寬度，中日韓字元佔兩欄，用 len() 會歪掉。
"""
from __future__ import annotations

from .session_analysis import SessionSummary, compare, correlation
from .terminal import cell, display_width

_LABEL = 22


def _row(label: str, text: str) -> str:
    return cell(label, _LABEL) + text


def _quality(summary: SessionSummary) -> list[str]:
    lines = [
        _row("可用幀", f"{summary.usable} / {summary.frames}"
             f"（略過 {summary.rejection_rate * 100:.1f}%）")
    ]
    if summary.duration_s:
        fps = summary.frames / summary.duration_s if summary.duration_s > 0 else 0
        lines.append(_row("長度", f"{summary.duration_s:.0f} 秒，約 {fps:.1f} fps"))

    if not summary.reasons:
        return lines
    lines.append("")
    lines.append("略過的原因：")
    for key, count in summary.reasons.most_common():
        lines.append(f"  {count:4d}  {summary.reason_examples[key]}")
    # 全部因為同一件事被擋掉時，那件事就是問題本身，值得單獨講一次。
    top_key, top_count = summary.reasons.most_common(1)[0]
    if summary.rejected and top_count / summary.rejected > 0.8 and summary.rejected > 5:
        lines.append(f"  略過的幀幾乎都是同一個原因（{top_count}/{summary.rejected}），"
                     f"那就是要查的東西")
    return lines


def _angles(summary: SessionSummary) -> list[str]:
    if not summary.angles:
        return ["沒有可用的角度資料"]

    lines = [
        cell("角度", 16) + cell("平均", 20) + cell("單幀標準差", 14) + "範圍",
    ]
    for name, angle in summary.angles.items():
        error = angle.standard_error
        shown = f"{angle.mean:+.2f}°" + ("" if error is None else f" ± {error:.2f}°")
        lines.append(
            cell(name, 16) + cell(shown, 20)
            + cell(f"±{angle.single_frame_std:.2f}°", 14)
            + f"{angle.minimum:+.1f}° ~ {angle.maximum:+.1f}°"
        )

    if not summary.has_baseline:
        lines.append("")
        lines.append("這一段沒有扣除個人基準，所以「扣基準」與「原始」相同。"
                     "判定門檻套在原始角度上會因人而異")
    return lines


def _correlation_evidence(summary: SessionSummary) -> list[str]:
    """相鄰幀相關的直接證據：區段平均的散佈比白雜訊該有的大多少。"""
    angle = summary.angles.get("θ_CA 原始")
    if angle is None:
        return []
    segments = angle.segment_means
    if segments.size == 0:
        return []

    lines = [_row("lag-1 自相關", "—" if angle.autocorrelation is None
                  else f"{angle.autocorrelation:+.2f}")]
    lines.append(_row("區段平均", " ".join(f"{v:+.1f}" for v in segments)))
    expected = angle.white_noise_segment_spread
    observed = float(segments.std())
    if expected is not None:
        lines.append(_row("區段平均的散佈",
                          f"±{observed:.2f}°，白雜訊該有 ±{expected:.2f}°"))
    lines.append(_row("平均值的誤差",
                      f"{'—' if angle.standard_error is None else f'±{angle.standard_error:.2f}°'}"
                      f"（std/√N 會給 ±{angle.naive_standard_error:.2f}°）"))
    if angle.is_correlated:
        lines.append("  區段之間的差距遠大於雜訊能解釋的範圍，"
                     "受試者在這段時間裡真的在動。std/√N 不適用")
    return lines


def _conditions(summary: SessionSummary) -> list[str]:
    if not summary.columns:
        return []
    lines = [cell("量測條件", 22) + cell("平均", 12) + "範圍"]
    for name, column in summary.columns.items():
        lines.append(
            cell(name, 22) + cell(f"{column.mean:8.1f}", 12)
            + f"{column.minimum:.1f} ~ {column.maximum:.1f}"
        )

    angle = summary.angles.get("θ_CA 原始")
    azimuth = summary.columns.get("方位角 °")
    if angle is not None and azimuth is not None:
        r = correlation(angle.values, azimuth.values)
        if r is not None:
            verdict = ("方位角解釋不了角度的變化，解剖平面是有效的"
                       if abs(r) < 0.3 else
                       "相關性偏高，角度可能還殘留方位角相依性，值得回頭查")
            lines.append("")
            lines.append(_row("corr(θ_CA, 方位角)", f"{r:+.2f}　{verdict}"))
    return lines


def _azimuth(summary: SessionSummary) -> list[str]:
    """略過率隨方位角怎麼變。這是方位角上限的直接量測。

    幾何上的精度隨方位角變好，所以上限只能來自遮擋：遠側肩膀被身體擋住之後，
    解剖平面與 θ_sym 都算不出來。看的是肩膀配對錯誤從哪一箱開始變多。
    """
    bins = summary.azimuth_bins
    if len(bins) < 2:
        return []
    lines = [cell("方位角", 12) + cell("幀數", 8) + cell("略過率", 10)
             + cell("其中肩膀配錯", 15) + "共同關鍵點"]
    for b in bins:
        shared = "—" if b.shared_keypoints is None else f"{b.shared_keypoints:.1f}"
        lines.append(
            cell(f"{b.low:.0f}~{b.high:.0f}°", 12)
            + cell(f"{b.frames}", 8)
            + cell(f"{b.rejection_rate * 100:.0f}%", 10)
            + cell(f"{b.shoulder_rejected}", 15)
            + shared
        )
    dropped = summary.frames - sum(b.frames for b in bins)
    if dropped:
        # 量不到方位角的幀不在任何一箱裡，而那些幀往好幾種原因偏：雙肩沒有
        # 同時偵測到就算不出方位角，而那也正是最容易被略過的情況。不講的話
        # 表上的略過率會比整段低得莫名其妙。
        lines.append("")
        lines.append(f"  另有 {dropped} 幀量不到方位角（雙肩沒有同時偵測到），"
                     f"不在上表任何一箱裡。整段的略過率是 "
                     f"{summary.rejection_rate * 100:.0f}%，"
                     f"與上表的數字對不上就是因為這些幀。")

    worst = max(bins, key=lambda b: b.rejection_rate)
    best = min(bins, key=lambda b: b.rejection_rate)
    if worst.rejection_rate <= 2 * max(best.rejection_rate, 0.02):
        return lines

    times = worst.rejection_rate / max(best.rejection_rate, 0.01)
    lines.append("")
    if worst.low > best.low:
        # 方位角愈大愈糟，這才是遮擋的樣子。
        lines.append(f"  {worst.low:.0f}° 以上的略過率是 {best.low:.0f}° 那一段的 "
                     f"{times:.0f} 倍，遠側肩膀開始被擋住。這是方位角上限的證據")
    else:
        # 方位角愈小反而愈糟，那就不是遮擋。先前這裡不分方向，一律寫成
        # 「遠側肩膀開始被擋住」，於是在這種資料上講出與數字相反的結論。
        lines.append(f"  略過最多的是 {worst.low:.0f}~{worst.high:.0f}° 這一箱"
                     f"（{best.low:.0f}° 那一段的 {times:.0f} 倍），而它不是"
                     f"方位角最大的一箱。遮擋解釋不了這個順序，要往別的原因查")
    return lines


def _judgement(summary: SessionSummary) -> list[str]:
    replay = summary.judgement
    if replay is None:
        return ["沒有判定資料。當時沒給 --baseline，或整段都沒有可用的角度"]

    source = (f"重播（視窗 {replay.window} 幀，遲滯 {replay.margin:.1f} 倍標準誤差）"
              if replay.replayed else "讀自 CSV 的 posture 欄")
    lines = [_row("來源", source)]
    for state in ("超標", "正常", "未知"):
        if replay.counts.get(state):
            lines.append(_row(state, f"{replay.counts[state]:4d} 幀"
                               f"（{replay.share(state) * 100:.0f}%）"))
    lines.append(_row("狀態變化", f"{replay.transitions} 次"))
    if replay.first_over_frame:
        lines.append(_row("首次超標", f"第 {replay.first_over_frame} 幀"))
    else:
        lines.append(_row("首次超標", "整段從未超標"))

    if not summary.has_baseline:
        lines.append("  沒有個人基準，這個判定不能當成誤報率")
    return lines


def format_session(summary: SessionSummary) -> str:
    """一份記錄的完整報告。"""
    title = f"{summary.path.name}"
    if summary.subject:
        title += f"（受試者 {summary.subject}）"

    blocks = [
        ("資料品質", _quality(summary)),
        ("角度", _angles(summary)),
        ("相鄰幀的相關性", _correlation_evidence(summary)),
        ("量測條件", _conditions(summary)),
        ("方位角與遮擋", _azimuth(summary)),
        ("判定", _judgement(summary)),
    ]
    out = [title, "=" * 60]
    for heading, lines in blocks:
        if not lines:
            continue
        out.append("")
        # 橫線照顯示寬度算，不是字元數：中英混排的標題會算錯。
        out.append(f"── {heading} " + "─" * max(0, 46 - display_width(heading)))
        out.extend(lines)
    return "\n".join(out)


def format_comparison(summaries: list[SessionSummary]) -> str:
    """兩段以上的記錄放在一起比。

    要比的是扣除基準後的角度，因為那才是判定實際看的數字。
    """
    key = "θ_CA 扣基準"
    usable = [s for s in summaries if key in s.angles]
    if len(usable) < 2:
        return ""

    out = ["", "── 兩段對照 " + "─" * 38]
    for first, second in zip(usable, usable[1:]):
        comparison = compare(first.angles[key], second.angles[key])
        out.append(f"{first.path.name} → {second.path.name}")
        out.append(_row("  θ_CA 差距", comparison.describe()))
        # 訊號要和姿勢本身的變異比才知道分不分得開，只報差距會高估可靠度。
        if comparison.spread_ratio is not None:
            out.append(_row("  差距 / 單幀散佈",
                            f"{comparison.spread_ratio:.1f} 倍"))
    return "\n".join(out)


def format_report(summaries: list[SessionSummary]) -> str:
    parts = [format_session(s) for s in summaries]
    comparison = format_comparison(summaries)
    if comparison:
        parts.append(comparison)
    return "\n\n".join(parts)


def segment_table(summary: SessionSummary, segments: int = 6) -> str:
    """把整段切成幾塊各自印統計，用來看漂移。

    三十分鐘的量測要看的是「每五分鐘往哪走」，整段一個平均看不出來。
    """
    angle = summary.angle("θ_CA")
    if angle is None or angle.count < segments * 2:
        return ""
    size = angle.count // segments
    duration = summary.duration_s or 0.0
    lines = ["", "── 分段 " + "─" * 42,
             cell("區段", 14) + cell("平均", 18) + "單幀標準差"]
    for i in range(segments):
        chunk = angle.values[i * size:(i + 1) * size]
        start = duration * i / segments
        end = duration * (i + 1) / segments
        lines.append(
            cell(f"{start:5.0f}~{end:5.0f}s", 14)
            + cell(f"{chunk.mean():+.2f}°", 18)
            + f"±{chunk.std():.2f}°"
        )
    drift = float(angle.values[-size:].mean() - angle.values[:size].mean())
    lines.append("")
    lines.append(_row("頭尾差距", f"{drift:+.2f}°"))
    lines.append("  單調往一個方向走的話，那是姿勢隨時間劣化，不是量測雜訊")
    return "\n".join(lines)
