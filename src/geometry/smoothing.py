"""時間序列平均。

在這個硬體上這不是加分項而是必要的。2026-09-23 實機量測：受試者保持不動，
距離 735mm、方位角約 8°，θ_CA 的單幀值掃過 50 度（標準差 ±11.5°），
而判定門檻是 10°。同一份資料的 θ_sym 標準差只有 ±2.1°，本來就堪用——
差別在 θ_CA 靠深度、θ_sym 不靠。

平均 N 幀把雜訊降到 1/√N，所以 30 幀（約 5 秒）可以把 θ_CA 壓到 ±2.1°。
代價是反應延遲，姿勢改變後要等一個視窗才會完全反映出來。

這一層只做平均與離群值排除，判定門檻與個人基準（θ_offset）屬於執行期監測邏輯，
不在這裡處理。
"""
from __future__ import annotations

from collections import deque

import numpy as np


class RollingAngle:
    """最近 N 個有效值的平均。

    無效的幀（偵測失誤）要在呼叫端先擋掉，不要丟進來用數值大小判斷——
    壞掉的三角測量給出的角度可以落在任何範圍，包含看起來完全正常的範圍。
    """

    def __init__(self, window: int = 30):
        if window < 1:
            raise ValueError(f"window必須>=1，收到{window}")
        self._window = window
        self._values: deque[float] = deque(maxlen=window)

    def add(self, value: float) -> None:
        self._values.append(float(value))

    def clear(self) -> None:
        self._values.clear()

    @property
    def window(self) -> int:
        return self._window

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def is_full(self) -> bool:
        """視窗填滿之前算出來的平均，降噪效果還沒到宣稱的 1/√N。"""
        return len(self._values) == self._window

    @property
    def mean(self) -> float | None:
        return float(np.mean(self._values)) if self._values else None

    @property
    def std(self) -> float | None:
        """目前視窗內的標準差。

        這是**實測**的雜訊量，可以拿來跟 estimate_theta_ca_precision_deg 的
        理論值對照。兩者差很多的話，代表誤差模型漏掉了什麼，
        或是受試者在視窗期間真的動了。
        """
        return float(np.std(self._values)) if len(self._values) > 1 else None

    @property
    def standard_error(self) -> float | None:
        """平均值本身的誤差，也就是 std/√N。判定該看的是這個數字。"""
        if len(self._values) < 2:
            return None
        return float(np.std(self._values) / np.sqrt(len(self._values)))
