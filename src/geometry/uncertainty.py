"""平均值的誤差。

`std/√N` 的前提是每一幀彼此獨立。θ_CA 不滿足這個前提。

2026-09-24 實機（`data/sessions/0924-upright.csv`，408 幀 / 81 秒，受試者保持坐正）：
相鄰幀的自相關是 0.73，前傾那份是 0.68。把 408 幀切成十段各取平均，段與段之間
從 +1.1° 到 +17.7° 都有，散佈 ±4.7°；若真是獨立的白雜訊，這個散佈該是 ±1.2°。
相關的來源是受試者本人，人微調坐姿的時間尺度是數秒，而取樣是每秒六幀。

低估的幅度用同一份資料驗證過：取所有 60 幀的滑動視窗算平均，這些平均的實際散佈
是 ±3.5°，而 `std/√N` 給的是 ±1.4°，差 2.6 倍。

這裡用批次平均法（batch means）。把序列切成長度 b 的區塊，各自取平均，再看這些
區塊平均的散佈。只要 b 大於相關的時間尺度，區塊之間就接近獨立，散佈就是真的。
b 取 √N 是這個方法的常見選法，在上面那份資料的 60 幀視窗上給出 ±3.1°，
與實測的 ±3.5° 相符。
"""
from __future__ import annotations

import numpy as np

# 低於這個長度就切不出足夠的區塊，批次平均法本身的變異會比它要估的東西還大。
_MINIMUM_SAMPLES = 8


def lag1_autocorrelation(values) -> float | None:
    """相鄰兩幀的相關係數。接近 0 代表獨立，接近 1 代表整段在慢慢漂。"""
    x = np.asarray(values, dtype=float)
    if x.size < 3:
        return None
    centred = x - x.mean()
    denominator = float(np.dot(centred, centred))
    if denominator == 0.0:
        return None
    return float(np.dot(centred[:-1], centred[1:]) / denominator)


def block_size(count: int) -> int:
    """批次平均法的區塊長度。"""
    return max(2, int(round(np.sqrt(count))))


def standard_error(values) -> float | None:
    """平均值本身的誤差，已經把相鄰幀的相關性算進去。

    樣本太少時回到 `std/√N`。那個值偏小，但少於八個樣本本來就切不出區塊，
    給一個偏小的估計仍然比不給好。
    """
    x = np.asarray(values, dtype=float)
    if x.size < 2:
        return None
    if x.size < _MINIMUM_SAMPLES:
        return float(x.std(ddof=1) / np.sqrt(x.size))

    b = block_size(x.size)
    blocks = x.size // b
    means = x[: blocks * b].reshape(blocks, b).mean(axis=1)
    return float(means.std(ddof=1) / np.sqrt(blocks))
