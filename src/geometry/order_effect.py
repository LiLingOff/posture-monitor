"""順序效應：A、B 兩組量同樣的姿勢，只是順序不同，比較兩組的結果。

說明書 3.4：「隨機分兩組採不同姿態呈現順序，以檢驗順序效應」。run-plan 把受試者
平衡分派到 A、B 兩組（`run_plan.assign_order`），B 組把可調換的趟序與趟內姿勢反過來
（趟 1 與固定的端正坐、自然坐不動），每一段的標頭記 `order_group`。

## 比什麼

每一段（自然坐、端正坐除外）減**同一趟第一段端正坐**的原始值，與 `trips.paired_differences`
同一個做法，所以跨趟的零點漂移約得掉。同一位受試者同一個姿勢有好幾份時（重量過）先取
平均，每人只算一次，誤差的分母是人數，不是幀數（理由與 `cohort.py` 相同）。

- 自然坐兩組都排在整個流程最前面，沒有被順序影響，不在這裡比。
- 端正坐是每一趟的對照，不比。
- 側傾依目標角度分開：左傾 10° 與左傾 20° 是兩個不同的姿勢。
- 組別以標頭的 `order_group` 為準；舊檔沒有時才查外面給的分組表。兩邊都沒有的受試者
  不列入，並在報告裡數出來。

## 能不能下結論

任一組不到 2 人時沒有誤差，只列數字、註明不能比。誤差用兩組各自的平均誤差合成
（`hypot`），這是兩個獨立組平均的差；它沒有做顯著性檢定，只是讓人看「差距是誤差的幾倍」。

這個模組不碰相機、終端機與 argparse。
"""
from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from dataclasses import dataclass
from io import StringIO

from .study import UPRIGHT, is_natural
from .terminal import cell, display_width, with_error
from .trips import (NATURAL_QUANTITIES, GroupMean, group_mean, paired_differences,
                    passes)

GROUPS = ("A", "B")
LABELS = ("θ_CA", "θ_sym")
_QUANTITIES = tuple(q for q in NATURAL_QUANTITIES if q.label in LABELS)

# 報告裡姿勢的排列順序；沒列到的接在後面，照名稱排。
_ORDER = ("head-forward", "forward", "left-shoulder-up", "right-shoulder-up",
          "tilt-left", "tilt-right")


def _is_compared(condition: str) -> bool:
    return bool(condition) and condition != UPRIGHT and not is_natural(condition)


def posture_key(summary) -> str:
    """姿勢的名稱：側傾要帶目標角度（tilt-left 10°），其他就是姿勢本身。"""
    condition = summary.condition or "未標註"
    target = summary.meta.get("target_angle_deg", "").strip()
    if target:
        try:
            return f"{condition} {float(target):g}°"
        except ValueError:
            pass
    return condition


def _sort_key(key: str):
    name, _, rest = key.partition(" ")
    rank = _ORDER.index(name) if name in _ORDER else len(_ORDER)
    try:
        angle = float(rest.rstrip("°")) if rest else 0.0
    except ValueError:
        angle = 0.0
    return (rank, name, angle)


def group_of(summary, fallback: Mapping[str, str] | None = None) -> str | None:
    """這一段的受試者在哪一組。標頭優先，再查外面給的表。"""
    marked = summary.meta.get("order_group", "").strip()
    if marked:
        return marked
    if fallback:
        return fallback.get(summary.subject_key) or None
    return None


@dataclass(frozen=True)
class OrderCell:
    """一個姿勢、一個量：A、B 兩組的平均與差。"""

    posture: str
    label: str
    a: GroupMean
    b: GroupMean

    @property
    def comparable(self) -> bool:
        """兩組都至少 2 人才有誤差，才談得上比。"""
        return self.a.error is not None and self.b.error is not None

    @property
    def difference(self) -> float | None:
        if self.a.mean is None or self.b.mean is None:
            return None
        return self.a.mean - self.b.mean

    @property
    def error(self) -> float | None:
        if not self.comparable:
            return None
        return float(math.hypot(self.a.error, self.b.error))

    @property
    def multiple(self) -> float | None:
        """差距是誤差的幾倍。"""
        if self.difference is None or not self.error:
            return None
        return self.difference / self.error


@dataclass(frozen=True)
class OrderEffect:
    cells: tuple[OrderCell, ...]
    people: dict[str, int]          # 組別 → 人數（有任何一段被納入的受試者）
    ungrouped: tuple[str, ...]      # 沒有組別、因此沒列入的受試者


def order_effect(summaries, fallback: Mapping[str, str] | None = None,
                 cutoff: float | None = None) -> OrderEffect | None:
    """沒有任何一段有組別時回傳 None（報告就不列這一節）。

    cutoff 是略過率敏感度分析用的門檻，意思與 `trips.passes` 相同；None 是主要結果。
    """
    rows = paired_differences(summaries, _is_compared, _QUANTITIES)
    # (姿勢, 組別, 量) → {受試者: [差值]}
    bucket: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    people: dict[str, set[str]] = {g: set() for g in GROUPS}
    ungrouped: set[str] = set()
    for row in rows:
        group = group_of(row.target, fallback)
        if group is None:
            ungrouped.add(row.subject)
            continue
        if group not in GROUPS or not row.paired:
            continue
        key = posture_key(row.target)
        for label in LABELS:
            diff = row.differences.get(label)
            if diff is None or diff.value is None:
                continue
            if not passes(list(diff.rates), diff.status, cutoff):
                continue
            bucket.setdefault((key, group, label), {}).setdefault(
                row.subject, []).append(diff.value)
            people[group].add(row.subject)
    if not any(people.values()):
        return None
    postures = sorted({k for k, _, _ in bucket}, key=_sort_key)
    cells = []
    for key in postures:
        for label in LABELS:
            means = {}
            for group in GROUPS:
                per_subject = bucket.get((key, group, label), {})
                means[group] = group_mean(
                    [sum(v) / len(v) for v in per_subject.values()])
            cells.append(OrderCell(key, label, means["A"], means["B"]))
    return OrderEffect(tuple(cells), {g: len(s) for g, s in people.items()},
                       tuple(sorted(ungrouped - set().union(*people.values()))))


def _table(header, rows) -> list[str]:
    widths = [w for _, w in header]
    out = ["".join(cell(name, w) for name, w in header).rstrip()]
    out.append("-" * display_width(out[0]))
    for row in rows:
        out.append("".join(cell(text, w) for text, w in zip(row, widths)).rstrip())
    return out


def _mean_text(group: GroupMean) -> str:
    if group.mean is None:
        return "—"
    return f"{with_error(group.mean, group.error, '°', 1)}（n={group.n}）"


def order_text(effect: OrderEffect) -> str:
    out = ["A 組是計畫檔本身的順序。B 組趟 1 仍排第一，之後的趟序與各趟內的姿勢反過來；",
           "自然坐、每趟第一段端正坐、趟 1 收尾的端正坐位置不變。",
           "每個姿勢減同一趟第一段端正坐，每位受試者先取平均，誤差的分母是人數。",
           "自然坐兩組都排在最前面，沒有被順序影響，不在這裡比。", ""]
    people = "、".join(f"{g} 組 {effect.people[g]} 人" for g in GROUPS)
    out.append(f"人數：{people}")
    if effect.ungrouped:
        out.append(f"沒有組別所以沒列入：{'、'.join(effect.ungrouped)}")
    out.append("")
    body = []
    for c in effect.cells:
        diff = ("—" if c.difference is None
                else with_error(c.difference, c.error, "°", 1))
        body.append([c.posture, c.label, _mean_text(c.a), _mean_text(c.b), diff,
                     "—" if c.multiple is None else f"{c.multiple:+.1f}"])
    out += _table([("姿勢", 20), ("量", 8), ("A 組", 24), ("B 組", 24),
                   ("A − B", 18), ("差是幾倍誤差", 12)], body)
    out += ["",
            "  任何一組不到 2 人時沒有誤差，只列數字，不能比。",
            "  「差是幾倍誤差」只是讓人看差距相對於誤差有多大，沒有做顯著性檢定。",
            "  兩組人數少的時候，差很容易是個人差異，不是順序造成的。"]
    return "\n".join(out)


def order_csv(effect: OrderEffect) -> str:
    fields = ["posture", "quantity", "a_n", "a_mean", "a_error", "b_n", "b_mean",
              "b_error", "a_minus_b", "a_minus_b_error", "multiple_of_error"]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()

    def r(value, digits=4):
        return "" if value is None else round(value, digits)

    for c in effect.cells:
        writer.writerow({
            "posture": c.posture, "quantity": c.label,
            "a_n": c.a.n, "a_mean": r(c.a.mean), "a_error": r(c.a.error),
            "b_n": c.b.n, "b_mean": r(c.b.mean), "b_error": r(c.b.error),
            "a_minus_b": r(c.difference), "a_minus_b_error": r(c.error),
            "multiple_of_error": r(c.multiple, 2),
        })
    return buffer.getvalue()
