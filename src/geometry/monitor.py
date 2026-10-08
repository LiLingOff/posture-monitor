"""`monitor` 模式的規則：這一幀該做什麼。

一個視窗跑到底，按 `c` 當場歸零、按 `q` 離開。這裡是它的狀態機，沒有相機、
沒有 cv2、沒有 argparse，所以測得起來不必湊出整條管線。

**按 `c` 不是抓一幀。** 前作的做法是把當下那一幀的角度設成偏移值。我們的單幀
散佈實測是 ±5.3° 到 ±7.8°，而判定門檻是 10°，一幀的零點比沒有零點更糟。所以
`c` 觸發的是倒數加一段連續取樣，跟 `baseline` 子指令收的是同一種東西。

順帶一提，這樣做讓零間隔變成結構保證。2026-09-29 誤報率 63% 那次的起因是基準
與量測之間隔了九分鐘，而在同一個視窗裡按 `c`，兩者之間不可能有間隔。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .baseline import REJECTION_LIMIT, BaselineCollector, PostureBaseline
from .judgement import Judgement, PostureJudge, windows_are_stale
from .pipeline import unusable_reason
from .smoothing import RollingAngle
from .terminal import with_error


class Mode(Enum):
    """現在在做什麼。"""

    WAITING = "待命"
    COUNTDOWN = "倒數"
    COLLECTING = "取基準"
    MONITORING = "監測"


@dataclass(frozen=True)
class FrameResult:
    """一幀走完之後的結果。"""

    mode: Mode
    corrected: tuple[float | None, float | None]
    reason: str | None
    verdict: Judgement | None = None
    phase_remaining_s: float | None = None
    notice: str | None = None
    baseline_changed: bool = False


class MonitorState:
    """待命 → 倒數 → 取基準 → 監測。"""

    def __init__(
        self, subject: str, window: int = 30, margin: float = 1.0,
        countdown_s: float = 5.0, baseline_s: float = 20.0,
        baseline: PostureBaseline | None = None, no_judge: bool = False,
    ):
        self._subject = subject
        self._countdown_s = float(countdown_s)
        self._baseline_s = float(baseline_s)
        self._baseline = baseline
        self._no_judge = no_judge
        self._margin = float(margin)

        self.ca_window = RollingAngle(window)
        self.sym_window = RollingAngle(window)
        self.drop_window = RollingAngle(window)
        self._judge = PostureJudge(margin_factor=margin)
        self._judge.watch_shoulder_drop(
            None if baseline is None else baseline.shoulder_drop_threshold_mm)
        self._misses = 0

        self._collector: BaselineCollector | None = None
        self._phase_ends_at: float | None = None
        self._collect_started_at: float | None = None
        self._mode = Mode.MONITORING if baseline is not None else Mode.WAITING
        self.stopped = False

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def baseline(self) -> PostureBaseline | None:
        return self._baseline

    def request_baseline(self, now: float) -> None:
        """按了 `c`。先倒數，讓人坐定再開始收。"""
        self._collector = None
        self._mode = Mode.COUNTDOWN
        self._phase_ends_at = now + self._countdown_s

    def stop(self) -> None:
        self.stopped = True

    def feed(self, measurement, now: float) -> FrameResult:
        """收下一幀。"""
        reason = unusable_reason(measurement)
        corrected = self._correct(measurement)

        if self._mode is Mode.COUNTDOWN:
            return self._countdown(corrected, reason, now)
        if self._mode is Mode.COLLECTING:
            return self._collect(measurement, corrected, now)
        if self._mode is Mode.MONITORING:
            return self._monitor(measurement, corrected, reason)
        return FrameResult(mode=self._mode, corrected=corrected, reason=reason,
                           notice=None)

    # ---- 各個狀態 ------------------------------------------------------

    def _countdown(self, corrected, reason, now: float) -> FrameResult:
        """倒數期間的幀**不進基準**。

        一按下去就開始收的話，收到的前幾秒是人還在調整姿勢的樣子，而那正是
        散佈最大的一段，會直接灌進零點。
        """
        remaining = self._phase_ends_at - now
        if remaining > 0:
            return FrameResult(mode=self._mode, corrected=corrected, reason=reason,
                               phase_remaining_s=remaining)
        self._collector = BaselineCollector()
        self._collect_started_at = now
        self._phase_ends_at = now + self._baseline_s
        self._mode = Mode.COLLECTING
        return FrameResult(mode=self._mode, corrected=corrected, reason=reason,
                           phase_remaining_s=self._baseline_s,
                           notice="開始取基準，請坐正保持不動")

    def _collect(self, measurement, corrected, now: float) -> FrameResult:
        reason = self._collector.add(measurement)
        remaining = self._phase_ends_at - now
        if remaining > 0:
            return FrameResult(mode=self._mode, corrected=corrected, reason=reason,
                               phase_remaining_s=remaining)
        return self._finish(measurement, now, corrected, reason)

    def _finish(self, measurement, now: float, corrected, reason) -> FrameResult:
        """時間到了，這份基準能不能用。"""
        collector, self._collector = self._collector, None
        duration = now - self._collect_started_at
        try:
            baseline = collector.finish(self._subject, duration)
        except ValueError as exc:
            return self._give_up(corrected, reason, f"{exc}\n按 c 再取一次")

        total = collector.count + collector.rejected
        rate = collector.rejected / total if total else 0.0
        if rate > REJECTION_LIMIT:
            # 誤差小不代表可信：留下來的幀可能全都偏向同一邊，而被擋掉的
            # 那些正好是另一種姿勢。這一條與幀數不足一樣是結構性的問題，
            # 所以不採用；其他警告只提醒，操作者自己決定要不要重取。
            return self._give_up(
                corrected, reason,
                f"略過 {collector.rejected}/{total} 幀（{rate * 100:.0f}%），"
                f"這份基準不採用。按 c 再取一次"
            )

        self._adopt(baseline)
        warnings = collector.quality_warnings(baseline)
        notice = (f"基準 θ_CA {with_error(baseline.theta_ca_deg, baseline.theta_ca_standard_error_deg)}"
                  f"，開始監測")
        if warnings:
            notice += f"\n  需要注意：{warnings[0]}"
        # 換了零點就要重算，否則這一幀顯示的是舊零點下的偏移。
        return FrameResult(mode=self._mode, corrected=self._correct(measurement),
                           reason=reason, phase_remaining_s=None,
                           notice=notice, baseline_changed=True)

    def _monitor(self, measurement, corrected, reason) -> FrameResult:
        drop = self.shoulder_drop(measurement)
        if reason is None:
            self._misses = 0
            if corrected[0] is not None:
                self.ca_window.add(corrected[0])
            if corrected[1] is not None:
                self.sym_window.add(corrected[1])
            if drop is not None:
                self.drop_window.add(drop)
        else:
            self._misses += 1
            if windows_are_stale(self._misses, self.ca_window.window):
                # 不清的話，受試者離開座位之後會對著空椅子繼續回報上一個狀態。
                self.ca_window.clear()
                self.sym_window.clear()
                self.drop_window.clear()

        verdict = None
        if not self._no_judge:
            *_, verdict = self._judge.update(self.ca_window, self.sym_window,
                                            self.drop_window)
        return FrameResult(mode=self._mode, corrected=corrected, reason=reason,
                           verdict=verdict)

    # ---- 小工具 --------------------------------------------------------

    def _give_up(self, corrected, reason, notice: str) -> FrameResult:
        self._mode = Mode.WAITING
        self._phase_ends_at = None
        self._collect_started_at = None
        return FrameResult(mode=self._mode, corrected=corrected, reason=reason,
                           notice=notice)

    def _adopt(self, baseline: PostureBaseline) -> None:
        """換零點就要把移動平均清掉。

        不清的話，換零點之後的前 N 幀平均是兩個零點的資料混在一起，而判定
        看的正是那個平均。
        """
        self._baseline = baseline
        self.ca_window.clear()
        self.sym_window.clear()
        self.drop_window.clear()
        self._judge = PostureJudge(margin_factor=self._margin)
        self._judge.watch_shoulder_drop(baseline.shoulder_drop_threshold_mm)
        self._misses = 0
        self._mode = Mode.MONITORING

    def shoulder_drop(self, measurement) -> float | None:
        """肩膀比端正坐姿低了多少。沒有基準或基準沒記肩高就不算。"""
        if self._baseline is None:
            return None
        return self._baseline.shoulder_drop_mm(measurement.shoulder_height_mm)

    @property
    def drop_threshold_mm(self) -> float | None:
        return (None if self._baseline is None
                else self._baseline.shoulder_drop_threshold_mm)

    def _correct(self, measurement) -> tuple[float | None, float | None]:
        """扣掉個人基準。還沒有基準就給原始角度。

        待命時顯示原始角度是刻意的：那時候該看的是鏡頭有沒有對好，
        而不是一個沒有零點的偏移量。
        """
        if self._baseline is None:
            return (measurement.theta_ca_deg, measurement.theta_sym_deg)
        return self._baseline.correct(measurement.theta_ca_deg, measurement.theta_sym_deg)
