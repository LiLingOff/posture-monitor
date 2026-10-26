"""用標定板驗證三角測量的尺度：算出來的 mm 是不是真的 mm。

到 2026-10-20 為止沒有任何一項檢查能回答這件事。合成測試證明數學對，標定
驗算證明參數自洽,但那是兩回事：內參、外參與基線整組等比例縮放的話，重投影
誤差一樣漂亮，而算出來的每一個距離都錯同一個倍率，沒有任何一項既有檢查會叫。

做法是把標定板放在相機前，三角測量它的角點，相鄰角點的距離應該剛好是方格
邊長。這條路徑不經過姿態模型，所以幾何的誤差與關鍵點的誤差分得開,量到的
偏差只可能來自標定。

**這裡的散佈不是獨立樣本的散佈。** 同一張畫面上的幾十段距離共用同一組標定
參數，而尺度錯了的話它們會一起錯。所以要看的是平均值偏離多少（那是系統性
的部分），散佈只說明單段距離量得多準。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .terminal import cell, with_error


def adjacent_pairs(ids, squares_x: int, squares_y: int):
    """板上相鄰的角點編號配對，回傳 [(a, b, 是不是橫向), ...]。

    ChArUco 的內部角點是 (squares_x-1) × (squares_y-1) 個，由左上往右、
    一列一列編號。所以編號差 1 是橫向鄰居（同一列），差一整列是縱向鄰居。

    橫向與縱向分開回報：兩者都該等於方格邊長，只有一邊不對的話問題出在校正
    （rectify）而不是尺度。
    """
    cols, rows = squares_x - 1, squares_y - 1
    have = set(int(i) for i in ids)
    pairs = []
    for corner in sorted(have):
        col, row = corner % cols, corner // cols
        if col < cols - 1 and corner + 1 in have:
            pairs.append((corner, corner + 1, True))
        if row < rows - 1 and corner + cols in have:
            pairs.append((corner, corner + cols, False))
    return pairs


# 少於這麼多段距離就不足以下結論。板子完整入鏡時一幀就有幾十段。
_ENOUGH_PAIRS = 8


@dataclass(frozen=True)
class ScaleCheck:
    """一次尺度驗證的結果。距離單位都是 mm。"""

    expected_mm: float
    horizontal: np.ndarray
    vertical: np.ndarray
    distances_mm: np.ndarray
    depth_mm: float | None
    frames: int
    corner_pairs: int

    @property
    def measured_mm(self) -> float | None:
        return float(self.distances_mm.mean()) if len(self.distances_mm) else None

    @property
    def spread_mm(self) -> float | None:
        return float(self.distances_mm.std()) if len(self.distances_mm) > 1 else None

    @property
    def error_mm(self) -> float | None:
        """平均值的誤差。這幾段距離共用同一組標定，不是獨立樣本,見模組說明。"""
        if len(self.distances_mm) < 2:
            return None
        return float(self.distances_mm.std() / np.sqrt(len(self.distances_mm)))

    @property
    def relative_error(self) -> float | None:
        """量到的比該有的大多少。0.01 就是尺度多報 1%。"""
        measured = self.measured_mm
        if measured is None or self.expected_mm == 0:
            return None
        return measured / self.expected_mm - 1.0

    @property
    def axes_differ_mm(self) -> float | None:
        """橫向與縱向的平均差多少。只有一邊不對的話問題不在尺度。"""
        if len(self.horizontal) < 1 or len(self.vertical) < 1:
            return None
        return float(abs(self.horizontal.mean() - self.vertical.mean()))


def check_scale(points_by_id, squares_x: int, squares_y: int,
                square_size_mm: float, frames: int = 1) -> ScaleCheck:
    """每一幀一份 {角點編號: 3D 座標}，量出相鄰角點的距離。

    收多幀是因為單幀的角點偵測本身有雜訊，而我們要看的是系統性的偏差。多幀
    的距離全部倒進同一袋,它們量的是同一個東西。
    """
    horizontal: list[float] = []
    vertical: list[float] = []
    for points in points_by_id:
        for a, b, is_horizontal in adjacent_pairs(points, squares_x, squares_y):
            gap = float(np.linalg.norm(points[a] - points[b]))
            if not np.isfinite(gap):
                continue
            (horizontal if is_horizontal else vertical).append(gap)

    depths = [float(p[2]) for points in points_by_id for p in points.values()
              if np.isfinite(p[2])]
    return ScaleCheck(
        expected_mm=square_size_mm,
        horizontal=np.asarray(horizontal),
        vertical=np.asarray(vertical),
        distances_mm=np.asarray(horizontal + vertical),
        depth_mm=float(np.mean(depths)) if depths else None,
        frames=frames,
        corner_pairs=len(horizontal) + len(vertical),
    )


def format_scale_check(check: ScaleCheck, tolerance: float = 0.02) -> list[str]:
    """排成一段可以貼進文件的報告。"""
    lines = ["── 三角測量的尺度 ──────────────────────────────────"]
    if check.corner_pairs == 0:
        lines.append("  一對相鄰角點都沒量到。確認板子完整入鏡、兩眼都看得到它。")
        return lines

    lines.append(cell("量到的方格邊長", 18)
                 + with_error(check.measured_mm, check.error_mm, "mm", 2, signed=False))
    lines.append(cell("板子實際的邊長", 18) + f"{check.expected_mm:.2f}mm")
    relative = check.relative_error
    lines.append(cell("差距", 18) + f"{relative * 100:+.2f}%")
    # 只量到一段距離時沒有散佈可言。印「—」而不是 0,0 是「完全沒有誤差」，
    # 而這裡是「只有一個樣本」。先前這裡直接格式化 None，整個報告當掉，
    # 而那正是板子只有一角入鏡的時候,最需要看到訊息的時候。
    spread = "—" if check.spread_mm is None else f"±{check.spread_mm:.2f}mm"
    lines.append(cell("單段距離的散佈", 18) + spread)
    lines.append(cell("取樣", 18)
                 + f"{check.corner_pairs} 段（橫 {len(check.horizontal)}、"
                   f"縱 {len(check.vertical)}），{check.frames} 幀"
                 + ("" if check.depth_mm is None else f"，板子在 {check.depth_mm:.0f}mm"))

    axes = check.axes_differ_mm
    if axes is not None:
        lines.append(cell("橫向與縱向差", 18) + f"{axes:.2f}mm")

    lines.append("")
    if check.corner_pairs < _ENOUGH_PAIRS:
        lines.append(f"  只量到 {check.corner_pairs} 段距離。板子完整入鏡時"
                     f"一幀就有幾十段,這個數字這麼小，代表大半個板子只有"
                     f"一眼看得到，下面的結論不要當真。")
    if abs(relative) <= tolerance:
        lines.append(f"  尺度正確：偏差 {abs(relative) * 100:.2f}% 在 "
                     f"{tolerance * 100:.0f}% 以內。算出來的 mm 是真的 mm，"
                     f"而這條路徑沒有經過姿態模型，所以幾何的部分到此為止可信。")
    else:
        lines.append(f"  尺度偏了 {relative * 100:+.2f}%。所有用到距離的結果都要乘上 "
                     f"{1 / (1 + relative):.4f} 才對,肩部垂直位移直接受影響，"
                     f"θ_CA 與 θ_sym 是角度，等比例縮放不影響它們。")
        lines.append("  最可能的來源是標定時填的方格邊長與板子實際的邊長不符。"
                     "拿尺量一次板子，對照標定時下的 --square-size-mm。")

    if axes is not None and axes > check.expected_mm * tolerance:
        lines.append(f"  橫向與縱向差了 {axes:.2f}mm。尺度錯會讓兩邊一起錯，"
                     f"只有一邊不對的話問題在校正（rectify）而不是尺度。")
    return lines
