"""個人坐姿基準（θ_offset）。

判定門檻不能直接套在原始角度上。頸部解剖結構因人而異，同樣坐得端正，
兩個人量到的 θ_CA 可以差十幾度，直接比 10° 的門檻會讓一個人永遠報警、
另一個人永遠不報。前作的自適應歸零就是處理這件事，實測讓誤報率從 18.7%
降到 4.2%。

做法是請受試者坐正、目視前方、保持不動約 10 秒，取這段時間的平均當作
這個人的零點，之後判定的是「相對自己端正坐姿偏了多少」。

取平均而非單幀，理由與執行期相同：θ_CA 的單幀雜訊在這個硬體上是 ±10° 上下，
拿單幀當基準等於把一個 ±10° 的偏移永久寫進後續所有判定。10 秒約 60 幀，
標準誤差降到 ±1.5° 以內。

基準會連同距離與方位角一起存下來。這兩項不影響角度的正確性（解剖平面由雙肩
定義），但影響精度，所以記錄下來才知道這份基準是在什麼條件下取得的。
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .pipeline import precision_advice, unusable_reason

# 低於這個幀數就不給出基準。10 秒在實機的 6.4fps 下約 60 幀，
# 20 幀是大幅放寬後的下限，再少的話平均本身就不可信。
_MINIMUM_FRAMES = 20
# 基準的標準誤差超過這個值就提醒重做。這個偏移會進到之後每一次判定，
# 相對 10° 的門檻，2° 已經是可觀的系統性偏差。
_BASELINE_ERROR_WARNING_DEG = 2.0
# 實測散佈超過理論值這個倍數，代表受試者在取基準的過程中動了。
_MOVEMENT_FACTOR = 1.8


def _without_numbers(reason: str) -> str:
    """把訊息裡的數字換成佔位符，讓同一類原因歸成一類。"""
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
            f"θ_CA 基準的誤差是 ±{baseline.theta_ca_standard_error_deg:.1f}°"
            f"（單幀 ±{baseline.theta_ca_std_deg:.1f}°，平均 {baseline.frames} 幀）。"
            f"這個偏移會固定留在之後每一次判定裡，相對 10° 的門檻已經可觀。"
            + precision_advice(baseline.distance_mm, baseline.azimuth_deg)
            + "延長取樣時間只能開根號地改善，先處理上面那一項。"
        )
    if (expected_single_frame_error_deg is not None
            and baseline.theta_ca_std_deg > _MOVEMENT_FACTOR * expected_single_frame_error_deg):
        warnings.append(
            f"θ_CA 的散佈 ±{baseline.theta_ca_std_deg:.1f}° 明顯大於這個距離"
            f"該有的量測雜訊 ±{expected_single_frame_error_deg:.1f}°，"
            f"受試者在取基準的過程中應該動了。請他保持不動再取一次"
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
            key = _without_numbers(reason)
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
            theta_ca_standard_error_deg=float(ca.std() / np.sqrt(len(ca))),
            theta_sym_standard_error_deg=float(sym.std() / np.sqrt(len(sym))),
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
        if "深度" in reason:
            lines.append(
                "深度是從視差回推的，落在合理範圍外代表左右兩眼對到了不同的位置，"
                "不是受試者真的坐在那個距離。用 "
                "python posture.py once --all-keypoints 看左右像素座標差多少。"
            )
        elif "垂直視差" in reason:
            lines.append("對極線校正後左右的 y 應該幾乎相同，差太多代表標定不準或左右配對錯誤。")
        elif "共同關鍵點" in reason:
            lines.append("左右能同時看到的部位太少，多半是遮擋或其中一眼的畫面有問題。")
        return "\n".join(lines)

    def quality_warnings(self, baseline: PostureBaseline) -> list[str]:
        """這份基準有沒有問題。取基準時沒發現的話，之後每一次判定都帶著它。"""
        expected = float(np.mean(self._precision)) if self._precision else None
        warnings = baseline_quality_warnings(baseline, expected)
        if self.rejected > self.count:
            warnings.append(
                f"略過的幀（{self.rejected}）比收下的（{self.count}）還多，"
                f"偵測本身就不穩定，這份基準的代表性有限"
            )
        return warnings
