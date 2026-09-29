"""把一整批逐幀記錄彙整成報告要的數字。

`session_analysis` 處理一份記錄，這裡處理一整個資料夾。

## 統計單位是受試者，不是幀

五個人各量 500 幀不是 2500 個樣本，是 5 個。跨受試者的誤差要用「各受試者
平均值的散佈」算，不是把所有幀倒在一起。

這件事在這個專案不是理論顧慮。θ_CA 的相鄰幀自相關是 0.73，拿幀數當分母
已經犯過一次錯：原本用 `std/√N` 算平均值的誤差，實測低估 2.6 倍，兩種姿勢的
差距因此被報成 40 個標準誤差而實際是 10 個（見 uncertainty 模組）。同一個
錯誤在跨受試者的層級會更嚴重，因為人與人之間的差異遠大於幀與幀之間。

## 缺中繼資料的檔案

2026-09-24 與 09-29 的記錄寫在 `condition` 欄存在之前，而它們的分析價值最高。
缺條件的檔案歸到 `未標註` 這一組，照樣出現在逐 session 表裡，但不參與需要
條件的比較。不讓它們悄悄消失：少了一份資料，表格上看不出任何異狀。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from .baseline import REJECTION_LIMIT
from .judgement import Posture
from .session_analysis import UNLABELLED, SessionSummary, analyse_session
from .uncertainty import standard_error

# UNLABELLED 與 REJECTION_LIMIT 都從別處 re-export，讓 cohort 的使用者不必
# 知道它們原本住在哪一個模組。


@dataclass(frozen=True)
class ConditionResult:
    """一位受試者在一種姿勢下的結果，可能來自好幾次重複。"""

    subject: str
    condition: str
    sessions: list[SessionSummary]

    @cached_property
    def usable(self) -> list[tuple[SessionSummary, float]]:
        """可以進統計的 session，以及各自的 θ_CA 平均。

        平均跟著一起回傳，因為篩選本來就得算它來判斷算不算得出來，而
        `theta_ca_deg` 接著又要用同一個值。分開寫的話每個 session 的平均
        會算兩次。
        """
        out = []
        for summary in self.sessions:
            if summary.rejection_rate > REJECTION_LIMIT:
                continue
            angle = summary.angle("θ_CA")
            if angle is not None:
                out.append((summary, angle.mean))
        return out

    @property
    def usable_sessions(self) -> list[SessionSummary]:
        return [summary for summary, _ in self.usable]

    @property
    def theta_ca_deg(self) -> float | None:
        """這個人這種姿勢的代表值。

        多次重複時取各次平均值的平均，不是把所有幀倒在一起：一次量 300 幀、
        一次量 60 幀的話，後者的權重不該只有五分之一，它是獨立的一次嘗試。
        """
        values = [mean for _, mean in self.usable]
        return float(np.mean(values)) if values else None

    @property
    def over_share(self) -> float | None:
        """判定為超標的幀數比例。坐正時這就是誤報率，前傾時是靈敏度。"""
        total = over = 0
        for summary in self.usable_sessions:
            if summary.judgement is None:
                continue
            total += summary.judgement.total
            over += summary.judgement.counts.get(Posture.OVER.value, 0)
        return over / total if total else None


@dataclass
class Cohort:
    """一整批資料，按受試者與姿勢分組。"""

    sessions: list[SessionSummary] = field(default_factory=list)

    @cached_property
    def _groups(self) -> dict[tuple[str, str], ConditionResult]:
        """(受試者, 姿勢) → 結果。建一次就好。

        先前每次 `result()` 都掃一遍整個 session 清單，而報表會問
        「每個人 × 每種姿勢」再加上差距的兩次，五個人兩種姿勢就是二十幾遍。
        分組的鍵一律走 SessionSummary 的 subject_key/condition_key，
        免得這裡的歸屬規則與 subjects/conditions 的列舉規則各走各的。
        """
        buckets: dict[tuple[str, str], list[SessionSummary]] = {}
        for summary in self.sessions:
            buckets.setdefault(
                (summary.subject_key, summary.condition_key), []
            ).append(summary)
        return {
            key: ConditionResult(subject=key[0], condition=key[1], sessions=group)
            for key, group in buckets.items()
        }

    @cached_property
    def subjects(self) -> list[str]:
        return sorted({subject for subject, _ in self._groups})

    @cached_property
    def conditions(self) -> list[str]:
        """出現過的姿勢名稱，未標註的排在最後。"""
        seen = {condition for _, condition in self._groups}
        named = sorted(c for c in seen if c != UNLABELLED)
        return named + ([UNLABELLED] if UNLABELLED in seen else [])

    def result(self, subject: str, condition: str) -> ConditionResult:
        return self._groups.get(
            (subject, condition),
            ConditionResult(subject=subject, condition=condition, sessions=[]),
        )

    def differences(self, baseline_condition: str, other: str) -> dict[str, float]:
        """每個人在兩種姿勢之間差多少。回傳 {受試者: 差距}。

        逐人算完再跨人平均，而不是把兩組的所有幀各自平均再相減。後者會讓
        量得多的人主導結果，而每個人都只算一次嘗試。
        """
        out = {}
        for subject in self.subjects:
            a = self.result(subject, baseline_condition).theta_ca_deg
            b = self.result(subject, other).theta_ca_deg
            if a is not None and b is not None:
                out[subject] = b - a
        return out


@dataclass(frozen=True)
class Separation:
    """兩種姿勢分不分得開，跨受試者的版本。"""

    baseline_condition: str
    other: str
    per_subject: dict[str, float]

    @property
    def subjects(self) -> int:
        return len(self.per_subject)

    @property
    def mean_deg(self) -> float | None:
        values = list(self.per_subject.values())
        return float(np.mean(values)) if values else None

    @cached_property
    def error_deg(self) -> float | None:
        """跨受試者的標準誤差。分母是人數，不是幀數。

        只有一個人時回傳 None。一個樣本算不出散佈，而回傳 0 會被讀成
        「這個差距在人與人之間完全一致」，那是這份資料最不支持的一種說法。

        **這裡刻意不用 `uncertainty.standard_error`。** 那個函式在樣本數夠多時
        改走批次平均法，前提是相鄰的樣本彼此相關（θ_CA 的逐幀資料正是如此）。
        受試者之間沒有這種序列相關，一個人的資料放在清單裡的第幾個位置不帶任何
        訊息，套批次平均法只會把誤差算小。這一行就是 `std/√n`，而在這個層級
        它是對的。
        """
        values = list(self.per_subject.values())
        if len(values) < 2:
            return None
        return float(np.std(values, ddof=1) / np.sqrt(len(values)))

    @property
    def sigma(self) -> float | None:
        mean, error = self.mean_deg, self.error_deg
        if mean is None or error is None or error == 0:
            return None
        return abs(mean) / error


def collect(paths: list[Path]) -> Cohort:
    """讀進一批 CSV。資料夾會遞迴展開。"""
    files: list[Path] = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.csv")))
        elif path.suffix.lower() == ".csv":
            files.append(path)
    sessions = []
    for file in files:
        try:
            sessions.append(analyse_session(file))
        except ValueError:
            # 空檔或只有標題列的檔案跳過。量測沒跑起來就會留下那種檔案，
            # 讓整批彙整因此中斷不合理。
            continue
    return Cohort(sessions=sessions)


def session_error_deg(summary: SessionSummary) -> float | None:
    """單一 session 的誤差，已經把相鄰幀的相關性算進去。"""
    angle = summary.angle("θ_CA")
    return None if angle is None else standard_error(angle.values)
