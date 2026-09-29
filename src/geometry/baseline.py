"""個人坐姿基準（θ_offset）。

判定門檻不能直接套在原始角度上。頸部解剖結構因人而異，同樣坐得端正，
兩個人量到的 θ_CA 可以差十幾度，直接比 10° 的門檻會讓一個人永遠報警、
另一個人永遠不報。前作的自適應歸零就是處理這件事，實測讓誤報率從 18.7%
降到 4.2%。

做法是請受試者坐正、目視前方、保持不動一段時間，取這段時間的平均當作
這個人的零點，之後判定的是「相對自己端正坐姿偏了多少」。

取平均而非單幀，理由與執行期相同：θ_CA 的單幀雜訊在這個硬體上是 ±8° 上下，
拿單幀當基準等於把一個 ±8° 的偏移永久寫進後續所有判定。

**取樣至少 30 秒。** 早先算 10 秒就夠是因為用了 std/√N，而相鄰幀並不獨立
（見 uncertainty 模組）。實測 60 幀（約 10 秒）的基準，誤差是 ±3.1°，
佔 2026-09-24 量到的坐正與前傾差距 18° 的六分之一；要壓到 ±2° 以內需要
150 幀以上，在實機的 5fps 上下是 30 秒起跳。實際需要多久取決於當下的單幀
散佈，所以 `seconds_for_target_error` 會照實測算，不用固定值。

基準會連同距離與方位角一起存下來。這兩項不影響角度的正確性（解剖平面由雙肩
定義），但影響精度，所以記錄下來才知道這份基準是在什麼條件下取得的。

**一份基準只對當次 session 有效。** 2026-09-24 實機：同一個人、同一句指示，
相隔九分鐘取兩次「坐正」，θ_CA 基準是 +10.04° 與 −4.17°，差 14.21°。
把相關性算進去之後兩者的誤差是 ±2.5° 與 ±1.7°，差距仍有 4.7 個標準差，
所以這不是量測雜訊，是兩次真的坐得不一樣。對照當時量到的訊號
（前傾相對坐正 18.0°），基準的變異已經是訊號的八成。所以每個 session
開始前都要重取，`captured_at` 存下來就是為了事後看得出這份基準是什麼時候取的。
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .pipeline import (precision_advice, rejection_advice,  # noqa: F401
                       sitting_too_far, unusable_reason)
from .uncertainty import standard_error

# 低於這個幀數就不給出基準。實務上要的是 150 幀以上（約 30 秒），
# 40 幀是大幅放寬後的硬下限，用來擋住明顯沒在量的情況；
# 取樣不足的提醒交給 baseline_quality_warnings，那裡給得出實際的誤差數字。
_MINIMUM_FRAMES = 40
# 基準的標準誤差超過這個值就提醒重做。這個偏移會進到之後每一次判定，
# 相對 10° 的門檻，2° 已經是可觀的系統性偏差。
_BASELINE_ERROR_WARNING_DEG = 2.0
# 再久也不建議。基準的前提是「這段時間內姿勢不變」，而 2026-09-24 實測
# 相隔九分鐘的兩次坐正差 14°，拉太長只是把漂移平均進去，不會讓基準更準。
_LONGEST_USEFUL_SECONDS = 120.0
# 實測散佈超過理論值這個倍數，代表受試者在取基準的過程中動了。
_MOVEMENT_FACTOR = 1.8
# 散佈只是偏高、還沒到「在動」的程度。這時候誤差大多半是因為散佈大，
# 而把散佈壓一半等於把取樣時間砍到四分之一，比拉長取樣划算。
_ELEVATED_SPREAD_FACTOR = 1.3


def group_rejection_reason(reason: str) -> str:
    """把訊息裡的數字換成佔位符，讓同一類原因歸成一類。

    「深度 54mm」與「深度 61mm」是同一件事，分開計數會蓋掉「每一幀都因為同一個
    原因被擋掉」這個訊息，而那正是最該講出來的。回傳值只當分組用的鍵，
    顯示時要拿原句，不然使用者看到的是「深度 Nmm」。
    """
    return re.sub(r"-?\d+(\.\d+)?", "N", reason)


@dataclass(frozen=True)
class PostureBaseline:
    """一位受試者端正坐姿的零點。"""

    subject: str
    captured_at: str
    frames: int
    rejected: int
    duration_s: float
    theta_ca_deg: float
    theta_sym_deg: float
    theta_ca_std_deg: float
    theta_sym_std_deg: float
    theta_ca_standard_error_deg: float
    theta_sym_standard_error_deg: float
    distance_mm: float
    azimuth_deg: float

    def correct(self, theta_ca: float | None, theta_sym: float | None):
        """把原始角度換算成相對這個人端正坐姿的偏移量。"""
        return (
            None if theta_ca is None else theta_ca - self.theta_ca_deg,
            None if theta_sym is None else theta_sym - self.theta_sym_deg,
        )

    def save(self, path: Path, overwrite: bool = False) -> None:
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"{path} 已經存在。連續替幾位受試者取基準時很容易忘記換檔名，"
                f"蓋掉的話前一位的判定基準就沒了。換個檔名，"
                f"或確定要覆蓋的話加上 --overwrite"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> PostureBaseline:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        missing = {f for f in cls.__dataclass_fields__} - set(data)
        if missing:
            raise ValueError(
                f"{path} 缺少欄位 {sorted(missing)}，可能是舊版存的檔案。請重新取一次基準"
            )
        return cls(**{k: data[k] for k in cls.__dataclass_fields__})

    def describe(self) -> str:
        return (
            f"受試者 {self.subject}（{self.captured_at}）\n"
            f"  θ_CA  基準 {self.theta_ca_deg:+.2f}° ± {self.theta_ca_standard_error_deg:.2f}°"
            f"（單幀標準差 ±{self.theta_ca_std_deg:.1f}°）\n"
            f"  θ_sym 基準 {self.theta_sym_deg:+.2f}° ± {self.theta_sym_standard_error_deg:.2f}°"
            f"（單幀標準差 ±{self.theta_sym_std_deg:.1f}°）\n"
            f"  取樣 {self.frames} 幀 / {self.duration_s:.1f} 秒，略過 {self.rejected} 幀，"
            f"距離 {self.distance_mm:.0f} mm，方位角 {self.azimuth_deg:.0f}°"
        )


def seconds_for_target_error(baseline: PostureBaseline, target_deg: float) -> float | None:
    """要把基準誤差壓到 target_deg，這個取樣速率下需要多久。

    誤差隨獨立樣本數的平方根下降，所以時間要乘上 (現在的誤差/目標)²。
    這是一階估計，前提是姿勢在更長的時間裡仍然不變；漂移會讓實際效果打折，
    所以超過 _LONGEST_USEFUL_SECONDS 就不該再往上加。

    回傳 None 代表這條路走不通，該動的是別的東西。
    """
    error = baseline.theta_ca_standard_error_deg
    if baseline.duration_s <= 0 or error <= target_deg:
        return None
    needed = baseline.duration_s * (error / target_deg) ** 2
    return None if needed > _LONGEST_USEFUL_SECONDS else needed


def _lever(baseline: PostureBaseline, expected_deg: float | None = None) -> str:
    """誤差太大時，該動哪一個。

    坐太遠的話先講距離。誤差隨距離平方成長，那一項壓倒性地大，這時候建議
    取樣兩分鐘是在浪費受試者的時間，把椅子往前拉一次就解決。

    距離沒問題才看取樣長度，因為那是唯一不必移動硬體就能改的。算出來超過
    兩分鐘的話拉長取樣也不是解法，姿勢本身會先漂掉。

    先前這裡用幀數門檻判斷卻用固定秒數給建議，於是 142 幀 / 30 秒的基準
    收到「拉長到 30 秒以上」，自相矛盾而且沒有可執行的下一步。
    """
    if sitting_too_far(baseline.distance_mm):
        return precision_advice(baseline.distance_mm, baseline.azimuth_deg)
    # 散佈偏高時先講它。誤差與散佈成正比、與時間的平方根成反比，所以把散佈
    # 壓一半等於取樣時間砍到四分之一。實機看到的是剛坐下那十幾秒人還在調整，
    # 那段的散佈明顯比後面大。
    if (expected_deg is not None
            and baseline.theta_ca_std_deg > _ELEVATED_SPREAD_FACTOR * expected_deg):
        return (f"單幀散佈 ±{baseline.theta_ca_std_deg:.1f}° 比這個位置該有的 "
                f"±{expected_deg:.1f}° 高，先讓受試者坐定再開始"
                f"（--countdown 拉長），比拉長取樣有效")
    seconds = seconds_for_target_error(baseline, _BASELINE_ERROR_WARNING_DEG)
    if seconds is not None:
        return (f"再取一次，這次取樣 {seconds:.0f} 秒"
                f"（目前的速率是每秒 {baseline.frames / baseline.duration_s:.1f} 幀）")
    return ("拉長取樣已經補不回來（再久姿勢本身就會漂）。"
            + precision_advice(baseline.distance_mm, baseline.azimuth_deg))


def baseline_quality_warnings(
    baseline: PostureBaseline, expected_single_frame_error_deg: float | None = None
) -> list[str]:
    """這份基準有沒有問題。

    寫成獨立函式而不是收集器的方法，因為**載入時也要檢查**。取基準當下看過一次
    就過去了，而這個偏移會固定留在之後每一次判定裡；存檔之後不再提醒的話，
    一份明知不好的基準會默默污染整場量測。

    expected_single_frame_error_deg 是這個距離與方位下該有的量測雜訊。
    實測散佈遠大於它，多出來的那部分只可能來自受試者本人。
    """
    warnings: list[str] = []
    if baseline.theta_ca_standard_error_deg > _BASELINE_ERROR_WARNING_DEG:
        warnings.append(
            f"基準誤差 ±{baseline.theta_ca_standard_error_deg:.1f}°"
            f"（單幀 ±{baseline.theta_ca_std_deg:.1f}°，{baseline.frames} 幀 / "
            f"{baseline.duration_s:.0f} 秒），這個偏移會留在之後每一次判定裡。"
            + _lever(baseline, expected_single_frame_error_deg)
        )
    if (expected_single_frame_error_deg is not None
            and baseline.theta_ca_std_deg > _MOVEMENT_FACTOR * expected_single_frame_error_deg):
        warnings.append(
            f"θ_CA 散佈 ±{baseline.theta_ca_std_deg:.1f}°，"
            f"這個位置該有的是 ±{expected_single_frame_error_deg:.1f}°。"
            f"受試者動了，請保持不動再取一次"
        )
    return warnings


class BaselineCollector:
    """把一段時間內的量測收集起來，算出這個人的零點。

    壞幀要在這裡就擋掉。基準算錯的後果比執行期單幀算錯嚴重得多：
    執行期的雜訊會被平均掉，基準的偏差會固定地留在之後每一次判定裡。
    """

    def __init__(self, minimum_frames: int = _MINIMUM_FRAMES):
        self._minimum = minimum_frames
        self._ca: list[float] = []
        self._sym: list[float] = []
        self._distance: list[float] = []
        self._azimuth: list[float] = []
        self._precision: list[float] = []
        self._reasons: Counter[str] = Counter()
        self._reason_examples: dict[str, str] = {}
        self.rejected = 0

    @property
    def count(self) -> int:
        return len(self._ca)

    def add(self, measurement) -> str | None:
        """收下一幀。無法使用時回傳原因字串並計入 rejected。"""
        reason = unusable_reason(measurement)
        if reason is None and (
            measurement.theta_ca_deg is None or measurement.theta_sym_deg is None
        ):
            reason = "有角度算不出來"
        if reason is not None:
            self.rejected += 1
            # 數字換成佔位符再計數，否則「深度 54mm」「深度 61mm」會被當成兩種原因，
            # 而「每一幀都因為同一件事被擋掉」正是最該講出來的訊息。
            # 佔位符只當分組用的鍵，顯示時要拿原句，不然使用者看到的是「深度 Nmm」。
            key = group_rejection_reason(reason)
            self._reasons[key] += 1
            self._reason_examples.setdefault(key, reason)
            return reason

        self._ca.append(measurement.theta_ca_deg)
        self._sym.append(measurement.theta_sym_deg)
        self._distance.append(measurement.reference_depth_mm)
        self._azimuth.append(measurement.camera_azimuth_deg)
        if measurement.theta_ca_precision_deg is not None:
            self._precision.append(measurement.theta_ca_precision_deg)
        return None

    @property
    def main_rejection(self) -> tuple[str, int] | None:
        """最常見的略過原因與次數。全部因為同一件事被擋掉時，那件事就是問題本身。

        回傳的是第一次遇到這個原因時的原句，數字保留著，方便直接看量級。
        """
        if not self._reasons:
            return None
        key, count = self._reasons.most_common(1)[0]
        return self._reason_examples[key], count

    def finish(self, subject: str, duration_s: float) -> PostureBaseline:
        if self.count < self._minimum:
            raise ValueError(
                f"只收到 {self.count} 幀有效資料（至少要 {self._minimum} 幀），"
                f"略過了 {self.rejected} 幀。\n{self._diagnose()}"
            )
        ca = np.asarray(self._ca)
        sym = np.asarray(self._sym)
        return PostureBaseline(
            subject=subject,
            captured_at=datetime.now().isoformat(timespec="seconds"),
            frames=self.count,
            rejected=self.rejected,
            duration_s=float(duration_s),
            theta_ca_deg=float(ca.mean()),
            theta_sym_deg=float(sym.mean()),
            theta_ca_std_deg=float(ca.std()),
            theta_sym_std_deg=float(sym.std()),
            theta_ca_standard_error_deg=float(standard_error(ca)),
            theta_sym_standard_error_deg=float(standard_error(sym)),
            distance_mm=float(np.mean(self._distance)),
            azimuth_deg=float(np.mean(self._azimuth)),
        )

    def _diagnose(self) -> str:
        """把略過的原因整理成一句能往下查的話，而不是一句通用的提醒。"""
        top = self.main_rejection
        if top is None:
            return "一幀都沒收到，確認受試者在畫面內、光線足夠。"
        reason, count = top
        lines = [f"略過的原因幾乎都是同一個（{count}/{self.rejected} 幀）：{reason}"]
        advice = rejection_advice(reason)
        if advice:
            lines.append(advice)
        return "\n".join(lines)

    def quality_warnings(self, baseline: PostureBaseline) -> list[str]:
        """這份基準有沒有問題。取基準時沒發現的話，之後每一次判定都帶著它。"""
        expected = float(np.mean(self._precision)) if self._precision else None
        warnings = baseline_quality_warnings(baseline, expected)
        if self.rejected > self.count:
            warnings.append(
                f"略過 {self.rejected} 幀比收下的 {self.count} 還多，偵測不穩定，"
                f"這份基準的代表性有限"
            )
        return warnings
