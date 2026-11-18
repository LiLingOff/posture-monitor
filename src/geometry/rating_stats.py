"""評分者信度與表3 用的統計，只用標準函式庫與 numpy（不依賴 scipy）。

**選擇與理由（寫進文件 05、10）：**

- Kendall 二分項目：兩位評分者用 Cohen's kappa；三位以上用 **Fleiss' kappa**
  （不是平均兩兩 kappa）。Fleiss 把所有評分者一起看，數字不會因為兩兩配對
  的組合數而偏向人多的那幾對；代價是只能用每位評分者都評過的照片。
- 主觀評分：**ICC(2,1) 與 ICC(2,k)**（Shrout & Fleiss 1979 的記法；McGraw &
  Wong 1996 的 ICC(A,1)、ICC(A,k)），雙向隨機效果、絕對一致。評分者被當成
  可以換人的隨機樣本，而且「誰給分偏高」要算進不一致（有效警報看的是分數差
  ≥2，評分者之間的系統偏移會直接影響這個判斷）。分析用的是評分者平均，
  所以真正對應的是 ICC(2,k)；ICC(2,1) 一併列出，代表單一評分者的可信度。
- 信賴區間：**以受試者為單位的群集 bootstrap**（百分位法，2000 次，種子固定）。
  同一位受試者的照片彼此相關，以照片為單位重抽會讓區間窄得不合理；受試者
  只有一位時退回以照片重抽，並在報告註明。人少時區間很寬，那是實情，報告
  一律同時列點估計與區間，並在受試者少於 10 位或區間寬度超過 0.4 時加註。
"""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261118
# 少於這麼多位受試者，或區間寬度超過這個值，報告加註「區間很寬」。
FEW_SUBJECTS = 10
WIDE_INTERVAL = 0.4


def cohen_kappa(a: list, b: list) -> float | None:
    """兩位評分者對同一批對象的類別判斷。

    完全一致是 1；與隨機一致無異是 0。兩人都只用同一個類別時（預期一致率
    為 1）kappa 沒有定義，回傳 None，不要回傳 1 或 0 假裝有答案。
    """
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1.0 - expected)


def fleiss_kappa(table: list[list]) -> float | None:
    """table[i] 是第 i 個對象被每位評分者判的類別，每列人數要相同（≥2）。"""
    if not table:
        return None
    k = len(table[0])
    if k < 2 or any(len(row) != k for row in table):
        return None
    n_items = len(table)
    totals: Counter = Counter()
    agreement = 0.0
    for row in table:
        counts = Counter(row)
        totals.update(counts)
        agreement += (sum(c * c for c in counts.values()) - k) / (k * (k - 1))
    p_bar = agreement / n_items
    p_e = sum((c / (n_items * k)) ** 2 for c in totals.values())
    if math.isclose(p_e, 1.0):
        return None
    return (p_bar - p_e) / (1.0 - p_e)


def kappa(table: list[list]) -> float | None:
    """兩位評分者用 Cohen，三位以上用 Fleiss。"""
    if not table:
        return None
    if len(table[0]) == 2:
        return cohen_kappa([r[0] for r in table], [r[1] for r in table])
    return fleiss_kappa(table)


def kappa_method(raters: int) -> str:
    return "Cohen's kappa" if raters == 2 else "Fleiss' kappa"


@dataclass(frozen=True)
class ICC:
    single: float | None    # ICC(2,1)
    average: float | None   # ICC(2,k)


def icc2(matrix) -> ICC:
    """ICC(2,1) 與 ICC(2,k)。matrix 是 n 個對象 × k 位評分者，不能有缺值。

    MSR、MSC、MSE 分別是對象間、評分者間、殘差的均方：
        ICC(2,1) = (MSR − MSE) / (MSR + (k−1)MSE + k(MSC − MSE)/n)
        ICC(2,k) = (MSR − MSE) / (MSR + (MSC − MSE)/n)
    分母為 0（例如所有分數都一樣）時沒有定義，回傳 None。
    """
    y = np.asarray(matrix, dtype=float)
    if y.ndim != 2 or y.shape[0] < 2 or y.shape[1] < 2:
        return ICC(None, None)
    n, k = y.shape
    grand = y.mean()
    ss_rows = k * ((y.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((y.mean(axis=0) - grand) ** 2).sum()
    ss_total = ((y - grand) ** 2).sum()
    # 數學上不會是負的；完全加成的資料相減會剩 -1e-14 這種浮點誤差。
    ss_error = max(ss_total - ss_rows - ss_cols, 0.0)
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_error / ((n - 1) * (k - 1))
    den1 = msr + (k - 1) * mse + k * (msc - mse) / n
    den2 = msr + (msc - mse) / n
    single = None if math.isclose(den1, 0.0, abs_tol=1e-12) else float((msr - mse) / den1)
    average = None if math.isclose(den2, 0.0, abs_tol=1e-12) else float((msr - mse) / den2)
    return ICC(single, average)


@dataclass(frozen=True)
class Interval:
    low: float | None
    high: float | None
    unit: str               # 「受試者」或「照片」：bootstrap 重抽的單位
    reps: int               # 算得出數值的重抽次數

    @property
    def width(self) -> float | None:
        if self.low is None or self.high is None:
            return None
        return self.high - self.low


def bootstrap(items: list, clusters: list[str], statistic: Callable[[list], float | None],
              reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED) -> Interval:
    """以群集重抽的百分位 95% 區間。clusters[i] 是 items[i] 所屬的受試者。"""
    groups: dict[str, list] = {}
    for item, cluster in zip(items, clusters):
        groups.setdefault(cluster, []).append(item)
    rng = random.Random(seed)
    if len(groups) >= 2:
        names = sorted(groups)
        unit = "受試者"

        def draw():
            out = []
            for _ in names:
                out.extend(groups[rng.choice(names)])
            return out
    else:
        unit = "照片"

        def draw():
            return [rng.choice(items) for _ in items]
    values = []
    for _ in range(reps):
        value = statistic(draw())
        if value is not None and math.isfinite(value):
            values.append(value)
    if len(values) < reps * 0.5:
        # 一半以上的重抽算不出來（例如都只剩一個類別），區間沒有意義。
        return Interval(None, None, unit, len(values))
    low, high = np.percentile(values, [2.5, 97.5])
    return Interval(float(low), float(high), unit, len(values))


def is_wide(interval: Interval, subjects: int) -> bool:
    width = interval.width
    return subjects < FEW_SUBJECTS or width is None or width > WIDE_INTERVAL


# ---- 表3：2×2 檢定 ---------------------------------------------------------

@dataclass(frozen=True)
class TwoByTwo:
    """[[a, b], [c, d]]，列是判定方式，欄是有效／誤報。"""

    a: int
    b: int
    c: int
    d: int

    @property
    def n(self) -> int:
        return self.a + self.b + self.c + self.d

    def chi_square(self) -> tuple[float | None, float | None, float | None]:
        """Pearson 卡方（自由度 1，不做 Yates 校正）、p 值、最小期望次數。"""
        rows = (self.a + self.b, self.c + self.d)
        cols = (self.a + self.c, self.b + self.d)
        if self.n == 0 or 0 in rows or 0 in cols:
            return None, None, None
        observed = ((self.a, self.b), (self.c, self.d))
        chi = 0.0
        smallest = math.inf
        for i in range(2):
            for j in range(2):
                expected = rows[i] * cols[j] / self.n
                smallest = min(smallest, expected)
                chi += (observed[i][j] - expected) ** 2 / expected
        return chi, math.erfc(math.sqrt(chi / 2)), smallest

    def fisher_p(self) -> float | None:
        """Fisher 精確檢定的雙尾 p 值（機率不大於觀察值的表加總）。"""
        row1, col1, n = self.a + self.b, self.a + self.c, self.n
        if n == 0:
            return None
        low, high = max(0, row1 + col1 - n), min(row1, col1)

        def prob(x):
            return (math.comb(col1, x) * math.comb(n - col1, row1 - x)
                    / math.comb(n, row1))
        observed = prob(self.a)
        total = sum(p for p in (prob(x) for x in range(low, high + 1))
                    if p <= observed * (1 + 1e-9))
        return min(1.0, total)


def mcnemar_exact(b: int, c: int) -> float | None:
    """McNemar 精確檢定（配對、雙尾）。b、c 是兩種判定方式意見不同的兩格。"""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)
