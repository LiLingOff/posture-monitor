"""量測當下與段落結束時印給人看的文字。

這一層只把數字排成字串，不碰相機、不碰 argparse，也不自己 print。理由是
這些句子本身出過不少錯，而它們先前全部混在 `posture.py` 的迴圈裡，沒有一條
測試看得到：訊息寫死一個會過期的相機編號（2026-10-06）、註解寫 15° 而程式判
10°、同一次錄製印出兩個不同的略過率、探測相機時同一件事講三次。這些都不是
演算法的錯，是文字的錯，而文字是使用者唯一看得到的東西。

回傳字串而不是直接印，呼叫端才決定要印在哪裡,逐幀那一行要原地更新，
狀態變化與摘要要獨立成行。
"""
from __future__ import annotations

import numpy as np

from .baseline import REJECTION_LIMIT
from .study import TURN_LIMIT_DEG
from .uncertainty import standard_error


def elapsed_text(seconds: float) -> str:
    """經過時間排成 m:ss。"""
    whole = int(seconds)
    return f"{whole // 60:2d}:{whole % 60:02d}"


def angle_text(window, instant: float | None) -> str:
    """平均值擺前面，單幀值放在括號裡，要看的是平均。"""
    mean = window.mean
    error = window.standard_error
    if mean is None:
        return "   —  "
    shown = f"{mean:+5.1f}" + (f"±{error:.1f}" if error is not None else "     ")
    return f"{shown}(單幀{instant:+5.1f})" if instant is not None else f"{shown}(單幀  — )"


def conditions_text(measurement) -> str:
    """距離與方位角。方位角要雙肩才量得到，所以它常常是 None。

    一邊肩膀沒偵測到時，深度與 θ_CA 仍然算得出來，那一幀不會被擋掉，
    但方位角是 None。直接格式化會在那一幀當掉。
    """
    distance = measurement.reference_depth_mm
    azimuth = measurement.camera_azimuth_deg
    return (
        ("  ——mm" if distance is None else f"  {distance:4.0f}mm")
        + (" ——°" if azimuth is None else f" {azimuth:2.0f}°")
    )


def live_line(
    ca_window, sym_window, measurement, corrected, rejected: int, reason,
    verdict=None, remaining_s: float | None = None,
) -> str:
    """逐幀原地更新的那一行。"""
    # 固定秒數的段落要顯示剩下多久。受試者維持姿勢時最想知道的就是這個，
    # 不知道還要多久就容易提早鬆掉。
    left = "" if remaining_s is None else f"  剩 {max(0.0, remaining_s):3.0f}s"
    if reason is not None:
        # 調整架設位置時正是略過最多的時候，這幾個數字不能跟著消失
        return (f"略過：{reason}{conditions_text(measurement)}"
                f"  已略過 {rejected} 幀{left}")
    # 括號裡放的是扣掉基準之後的單幀值。放原始角度的話它跟前面的平均差了一個
    # 基準的量，看起來像兩個不相干的數字。
    return (
        f"θ_CA {angle_text(ca_window, corrected[0])}"
        f"  θ_sym {angle_text(sym_window, corrected[1])}"
        f"{conditions_text(measurement)}"
        f"  {ca_window.count:2d}/{ca_window.window}幀"
        + (f"  略過{rejected}" if rejected else "")
        + ("" if verdict is None else f"  {verdict.posture.value}")
        + left
    )


def monitor_line(state, result, rejected: int) -> str:
    """monitor 的那一行。看不到視窗的時候，這裡仍然要說得出現在在做什麼。"""
    head = result.mode.value
    if result.phase_remaining_s is not None:
        head += f" 剩 {max(0.0, result.phase_remaining_s):3.0f}s"
    if result.reason is not None:
        return f"{head}  略過：{result.reason}（已略過 {rejected} 幀）"
    return (f"{head}  θ_CA {angle_text(state.ca_window, result.corrected[0])}"
            f"  θ_sym {angle_text(state.sym_window, result.corrected[1])}"
            + (f"  {result.verdict.posture.value}" if result.verdict else "")
            + (f"  略過{rejected}" if rejected else ""))


def transition_lines(transitions: list[str]) -> list[str]:
    """把狀態變化重印一次。

    逐幀的那一行會被下一幀蓋掉，狀態變化夾在裡面很容易錯過，而這幾行
    正是要記錄下來的東西。
    """
    if not transitions:
        return ["整段沒有狀態變化"]
    return [f"狀態變化 {len(transitions)} 次："] + [f"  {line}" for line in transitions]


def summary_lines(recording, baseline=None) -> list[str]:
    """結束時整段的統計，這才是可以記錄下來的數字。

    印的是**整段**，不是移動視窗。視窗只有 30 幀（約 6 秒），拿它當結尾的摘要
    等於把一百秒的量測講成最後六秒的樣子，而標題寫的是整段。移動視窗另外印
    一行，因為判定看的是它，兩個數字差很多本身就是資訊：那代表姿勢在變。

    吃整個 Recording 而不是拆開的四個欄位，因為略過率的分母只能有一個說法。
    先前這裡用 `rejected + 有角度的幀數`，而 study 的總結用 `Recording.frames`，
    兩者在「通過檢查但算不出 θ_CA」的幀上不同，於是同一次錄製印出兩個百分比。
    """
    lines = ["整段（相對個人基準的偏移量）：" if baseline
             else "整段（原始角度，未扣除個人基準）："]
    session = recording.session
    for name, values in (("θ_CA ", session.ca), ("θ_sym", session.sym)):
        if len(values) < 2:
            lines.append(f"{name}  沒有足夠的量測")
            continue
        array = np.asarray(values)
        error = standard_error(array)
        shown = "" if error is None else f" ± {error:.1f}°"
        lines.append(f"{name}  {array.mean():+.2f}°{shown}"
                     f"（{len(values)} 幀，單幀標準差 ±{array.std():.1f}°）")
    for name, window in (("θ_CA ", recording.ca_window),
                         ("θ_sym", recording.sym_window)):
        if window.mean is None:
            continue
        error = ("" if window.standard_error is None
                 else f" ± {window.standard_error:.1f}°")
        lines.append(f"  結束前 {window.count} 幀  {name} {window.mean:+.2f}°{error}")
    if recording.rejected:
        lines.append(f"略過 {recording.rejected} / {recording.frames} 幀"
                     f"（{recording.rejection_rate * 100:.0f}%）偵測失誤")
    return lines


def _segment_flags(segment) -> str:
    """這一段右邊那個「← …」。沒有問題就是空字串。"""
    flags = []
    if segment.camera_lost:
        flags.append("相機掉線，後面重量了" if segment.retaken else "相機掉線")
    elif segment.too_many_rejected:
        flags.append("略過率偏高")
    if segment.turned_too_far:
        flags.append(f"轉身 {segment.turned_deg:+.0f}°")
    return ("  ← " + "、".join(flags)) if flags else ""


def study_summary_lines(segments, subject: str, log_paths) -> list[str]:
    """一趟 study 全部量完之後的總結。

    每一段自己的摘要已經印過了，這裡看的是段與段之間：哪一段的資料不能用、
    以及接下來該跑什麼。
    """
    lines = ["", "=" * 60, f"受試者 {subject}，共 {len(segments)} 段"]
    for segment in segments:
        shown = ("—" if segment.theta_ca_deg is None
                 else f"{segment.theta_ca_deg:+.2f}°")
        lines.append(f"  {segment.condition:12s} #{segment.trial}  "
                     f"θ_CA {shown:>9s}  "
                     f"可用 {segment.usable}/{segment.frames}"
                     f"（略過 {segment.rejection_rate * 100:.0f}%）"
                     f"{_segment_flags(segment)}")

    bad = [seg for seg in segments if seg.needs_redo]
    if bad:
        lines.append("")
        # 轉身與略過率分開講，因為該做的事不一樣：略過率高是偵測的問題，
        # 轉身是受試者的問題，重量之前要先提醒他盯著固定的點。
        if any(seg.too_many_rejected for seg in bad):
            lines.append(f"略過率超過 {REJECTION_LIMIT * 100:.0f}% 的段落，"
                         "資料的代表性有限。")
        if any(seg.turned_too_far for seg in bad):
            lines.append(f"方位角比取基準時漂了 {TURN_LIMIT_DEG:.0f}° 以上的段落，"
                         "受試者轉身了，扣基準之後的角度不能用。"
                         "重量之前提醒他全程盯著牆上那個點。")
        lines.append("建議重量："
                     + "、".join(f"{seg.condition} #{seg.trial}" for seg in bad)
                     + "。重量就整趟重跑，只補一段的話基準不是同一份")

    usable = [str(path) for segment, path in zip(segments, log_paths)
              if path is not None and segment.usable_data]
    lines += ["", "接下來：先翻一遍存下來的畫面，確認每一張都是預期的姿勢，再跑",
              "  python posture.py analyse " + " ".join(usable)]
    return lines
