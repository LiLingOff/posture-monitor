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

2026-09-24 與 09-29 的記錄寫在 `condition` 欄存在之前，而那幾份正是最值得
分析的資料。缺條件的檔案歸到 `未標註` 這一組，照樣出現在逐 session 表裡，
但不參與需要條件的比較。不讓它們消失，因為消失的資料沒有人會發現。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .session_analysis import SessionSummary, analyse_session
from .uncertainty import standard_error

# 沒有記錄姿勢條件的檔案歸在這一組。
UNLABELLED = "未標註"
# 略過率超過這個比例的 session 不列入彙整的統計，但仍然會出現在逐 session 表裡。
_REJECTION_LIMIT = 0.15


@dataclass(frozen=True)
class ConditionResult:
    """一位受試者在一種姿勢下的結果，可能來自好幾次重複。"""

    subject: str
    condition: str
    sessions: list[SessionSummary]

    @property
    def usable_sessions(self) -> list[SessionSummary]:
        """略過率過高的 session 不進統計。留下來的幀代表不了整段姿勢。"""
        return [s for s in self.sessions
                if s.rejection_rate <= _REJECTION_LIMIT and self._mean(s) is not None]

    @staticmethod
    def _mean(summary: SessionSummary) -> float | None:
        angle = summary.angles.get("θ_CA 扣基準") or summary.angles.get("θ_CA 原始")
        return None if angle is None else angle.mean

    @property
    def theta_ca_deg(self) -> float | None:
        """這個人這種姿勢的代表值。

        多次重複時取各次平均值的平均，不是把所有幀倒在一起：一次量 300 幀、
        一次量 60 幀的話，後者的權重不該只有五分之一，它是獨立的一次嘗試。
        """
        values = [self._mean(s) for s in self.usable_sessions]
        return float(np.mean(values)) if values else None

    @property
    def over_share(self) -> float | None:
        """判定為超標的幀數比例。坐正時這就是誤報率，前傾時是靈敏度。"""
        total = over = 0
        for summary in self.usable_sessions:
            if summary.judgement is None:
                continue
            total += summary.judgement.total
            over += summary.judgement.counts.get("超標", 0)
        return over / total if total else None

    @property
    def rejection_rate(self) -> float | None:
        frames = sum(s.frames for s in self.sessions)
        rejected = sum(s.rejected for s in self.sessions)
        return rejected / frames if frames else None


@dataclass
class Cohort:
    """一整批資料，按受試者與姿勢分組。"""

    sessions: list[SessionSummary] = field(default_factory=list)

    @property
    def subjects(self) -> list[str]:
        seen = {s.subject or "?" for s in self.sessions}
        return sorted(seen)

    @property
    def conditions(self) -> list[str]:
        """出現過的姿勢名稱，未標註的排在最後。"""
        seen = {s.condition or UNLABELLED for s in self.sessions}
        named = sorted(c for c in seen if c != UNLABELLED)
        return named + ([UNLABELLED] if UNLABELLED in seen else [])

    def result(self, subject: str, condition: str) -> ConditionResult:
        return ConditionResult(
            subject=subject, condition=condition,
            sessions=[s for s in self.sessions
                      if (s.subject or "?") == subject
                      and (s.condition or UNLABELLED) == condition],
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

    @property
    def error_deg(self) -> float | None:
        """跨受試者的標準誤差。分母是人數，不是幀數。

        只有一個人時回傳 None：一個樣本算不出散佈，而回傳 0 會被讀成
        「完全沒有變異」，那是最糟的失效方式。
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
    angle = summary.angles.get("θ_CA 扣基準") or summary.angles.get("θ_CA 原始")
    return None if angle is None else standard_error(angle.values)
