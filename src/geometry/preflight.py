"""上機前檢查（`posture.py preflight`）的判斷部分：拿到的數字過不過、沒過要做什麼。

自測手冊「開始前」那幾項原本要組員一項一項手動看（`once --display` 數共同
關鍵點、記垂直視差、量距離、確認實機程式是最新的）。組員一次只來一小時，
這些事改成 AI 在組員來之前先用 ssh 跑一次，當天組員再跑一次，看到
「可以開始」就開始。

門檻都沿用既有的：共同關鍵點 12 以上與距離 550～700mm 是自測手冊的、
垂直視差目標 3px 是文件 09 的、8px 是 `pipeline` 判定配對錯誤的門檻、
可用幀比例對應略過率上限 15%。
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from .baseline import REJECTION_LIMIT

MIN_SHARED_KEYPOINTS = 12
DISTANCE_RANGE_MM = (550.0, 700.0)
VERTICAL_DISPARITY_TARGET_PX = 3.0
VERTICAL_DISPARITY_LIMIT_PX = 8.0


@dataclass(frozen=True)
class FrameProbe:
    """預檢時一幀的結果。偵測不到人或配不起來時 error 有值，其餘為 None。"""

    error: str | None = None
    shared_count: int | None = None
    angle_disparity_px: float | None = None
    depth_mm: float | None = None
    unusable: str | None = None      # pipeline.unusable_reason 的結果


@dataclass(frozen=True)
class Item:
    ok: bool
    label: str
    detail: str
    advice: str = ""
    # 只是提醒、不擋開始（例如連不上 GitHub 比對不了版本）。
    warning_only: bool = False


def git_items(local_head: str | None, remote_head: str | None,
              dirty: bool | None) -> list[Item]:
    """實機程式是不是最新、有沒有被手改過。"""
    items = []
    if local_head is None:
        items.append(Item(False, "程式版本", "讀不到目前的 commit",
                          "確認在 posture-monitor 資料夾裡執行", warning_only=True))
    elif remote_head is None:
        items.append(Item(False, "程式版本", f"目前 {local_head[:7]}，連不上 GitHub 比對",
                          "網路不通時可以照常量，事後再對版本", warning_only=True))
    elif local_head != remote_head:
        items.append(Item(False, "程式版本",
                          f"實機 {local_head[:7]}，GitHub 上是 {remote_head[:7]}",
                          "先跑 git pull --ff-only 再開始"))
    else:
        items.append(Item(True, "程式版本", f"與 GitHub 相同（{local_head[:7]}）"))
    if dirty:
        items.append(Item(False, "程式有沒有被改過", "有未 commit 的修改",
                          "不要在實機上直接改程式；把 git status 的輸出傳給使用者",
                          warning_only=True))
    elif dirty is not None:
        items.append(Item(True, "程式有沒有被改過", "乾淨"))
    return items


def probe_items(probes: list[FrameProbe]) -> list[Item]:
    """幾幀試拍的結果，整理成「人看得到嗎、配得起來嗎、距離對嗎」。"""
    total = len(probes)
    matched = [p for p in probes if p.error is None]
    if not matched:
        reason = probes[0].error if probes else "沒有拍到任何一幀"
        return [Item(False, "偵測到受試者", f"{total} 幀都沒有配到人：{reason}",
                     "請受試者坐到相機前、整個上半身在畫面裡，光線要夠")]

    items = []
    shared = median(p.shared_count for p in matched)
    items.append(Item(
        shared >= MIN_SHARED_KEYPOINTS, "共同關鍵點",
        f"中位數 {shared:g} 個（要 {MIN_SHARED_KEYPOINTS} 以上）",
        "上半身要整個入鏡，不要被桌子或螢幕擋住；長髮綁起來"))

    usable = [p for p in matched if p.unusable is None]
    fraction = len(usable) / total
    need = 1.0 - REJECTION_LIMIT
    reasons = sorted({p.unusable for p in matched if p.unusable} |
                     {p.error for p in probes if p.error})
    items.append(Item(
        fraction >= need, "可用幀",
        f"{len(usable)}/{total}（要 {need * 100:.0f}% 以上）"
        + (f"；略過原因：{reasons[0]}" if reasons and fraction < need else ""),
        "看略過原因：深度不合理多半是坐太遠或太近，配對錯多半是耳朵或肩膀被擋住"))

    disparities = [p.angle_disparity_px for p in usable if p.angle_disparity_px is not None]
    if disparities:
        worst = median(disparities)
        ok = worst <= VERTICAL_DISPARITY_LIMIT_PX
        detail = f"角度用到的點中位數 {worst:.1f}px（目標 {VERTICAL_DISPARITY_TARGET_PX:g}px 內）"
        advice = ""
        if worst > VERTICAL_DISPARITY_TARGET_PX:
            advice = ("超過目標但還能量；如果最近搬過相機，量完把這個數字告訴使用者，"
                      "可能要重標定")
        if not ok:
            advice = "相機可能被碰過，先不要量，把這個數字傳給使用者"
        items.append(Item(ok, "垂直視差", detail, advice,
                          warning_only=ok and worst > VERTICAL_DISPARITY_TARGET_PX))

    depths = [p.depth_mm for p in usable if p.depth_mm is not None]
    if depths:
        depth = median(depths)
        low, high = DISTANCE_RANGE_MM
        if depth < low:
            advice = "受試者離相機太近，椅子往後或相機往後移"
        elif depth > high:
            advice = "受試者離相機太遠，椅子往前或相機往前移"
        else:
            advice = ""
        items.append(Item(low <= depth <= high, "受試者距離",
                          f"{depth:.0f}mm（要 {low:.0f}～{high:.0f}mm）", advice))
    return items


def verdict_lines(items: list[Item]) -> list[str]:
    """印給操作者看的清單與結論。"""
    lines = ["", "上機前檢查"]
    for item in items:
        mark = "通過" if item.ok and not item.warning_only else (
            "注意" if item.warning_only else "沒過")
        lines.append(f"  [{mark}] {item.label}：{item.detail}")
        if item.advice and (not item.ok or item.warning_only):
            lines.append(f"         → {item.advice}")
    blocking = [i for i in items if not i.ok and not i.warning_only]
    lines.append("")
    if blocking:
        lines.append("結論：還不能開始。照上面「沒過」那幾項調整，再跑一次 preflight。")
    else:
        lines.append("結論：可以開始。")
    return lines


def all_clear(items: list[Item]) -> bool:
    return not any(not i.ok and not i.warning_only for i in items)
