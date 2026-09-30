"""超標判定。

這一層把角度變成「要不要提醒」。門檻套在**扣除個人基準之後**的角度上，
因為頸部結構因人而異：2026-09-24 實測，同一個人坐正的原始 θ_CA 是 +10.04°，
剛好壓在前作 10° 的門檻上，直接拿絕對值判定的話這個人正常坐著就會持續報警。

判定看的是移動平均而非單幀。2026-09-23 實機量測中受試者保持不動，θ_CA 的單幀值
掃過 50 度，而門檻是 10 度。

## 遲滯的寬度不是常數

單一門檻在門檻附近會來回觸發。常見做法是進入用一個值、解除用另一個值，中間的
寬度寫死；這裡改成從當下的量測誤差算。理由是誤差本身會變：距離、方位角、
受試者穩不穩都會影響它，2026-09-24 那兩份記錄裡 30 幀視窗的標準誤差從
±0.8° 到 ±3° 都有。寬度寫死的話，誤差大的時候擋不住抖動，誤差小的時候
又反應得不必要地慢。

規則只有一條，把平均值與門檻的距離拿來跟誤差比：

    平均 − k·誤差 > 門檻   →   超標
    平均 + k·誤差 < 門檻   →   正常
    兩者之間               →   維持原狀

第三條就是遲滯。「判斷不出來的時候不要改變狀態」與「進入和解除用不同的值」
是同一件事，只是這個寫法的寬度會自己跟著誤差調整。

k 是信賴邊界。放大會減少誤報但反應變慢、不動作的區間變寬。預設 1.0，
相當於單邊約 84% 的信賴水準。

## 沒有資料的時候不能說正常

視窗還沒填滿、或整段都被略過時，狀態是「未知」而不是「正常」。回報正常的話，
「相機根本沒看到人」會被讀成「這個人坐得很好」，而裝置會安靜地什麼都不做。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# 前作的門檻，套在扣除個人基準之後的偏移量上。
_THETA_CA_THRESHOLD_DEG = 10.0
# θ_sym 的門檻。肩膀高低差往左往右都算歪，所以這一項取絕對值。
_THETA_SYM_THRESHOLD_DEG = 5.0
# 信賴邊界的倍數。
_MARGIN_FACTOR = 1.0
# 遲滯寬度的下限。受試者很穩的時候標準誤差會趨近 0，寬度跟著趨近 0，
# 那就退化成單一門檻，真實的微幅移動也會讓狀態翻來翻去。
_MINIMUM_MARGIN_DEG = 1.0
# 肩高那一項的遲滯下限，單位毫米。理由與上面一樣，只是換了單位：
# 1mm 大約是這個距離下單幀定位誤差換算出來的量級。
_MINIMUM_DROP_MARGIN_MM = 2.0


def windows_are_stale(consecutive_misses: int, window: int) -> bool:
    """移動平均裡的值是不是全都過期了。

    `RollingAngle` 記的是最近 N 個**有效**值，沒有時間概念。偵測連續失敗時
    它會原樣保留舊值，判定就會拿一段已經過去的資料繼續回報狀態：受試者離開座位、
    或光線變暗讓偵測整段失效，裝置還是對著空椅子持續警告。

    連續失敗達到視窗長度時，視窗裡的每一個值都比一個視窗的時間跨度更舊，
    這時候該把視窗清掉讓狀態回到未知，而不是繼續用它判定。
    """
    return window > 0 and consecutive_misses >= window


class Posture(Enum):
    """判定結果。"""

    UNKNOWN = "未知"
    OK = "正常"
    OVER = "超標"


@dataclass(frozen=True)
class Judgement:
    """一次判定的結果與它的依據。

    把依據一起帶出來，因為「為什麼判成這樣」在調參數與寫報告時都要用到，
    事後從一個列舉值反推不出來。
    """

    posture: Posture
    changed: bool           # 與上一次判定相比狀態有沒有變，觸發回饋看的是這個
    frames_in_state: int    # 維持現在這個狀態多久了
    reason: str

    @property
    def should_warn(self) -> bool:
        return self.posture is Posture.OVER


class AngleJudge:
    """單一角度的判定，含遲滯。

    只處理一個角度。θ_CA 與 θ_sym 的門檻、方向、量測誤差都不同，
    共用一個狀態機會把兩者的遲滯綁在一起。
    """

    def __init__(
        self,
        threshold_deg: float,
        name: str,
        two_sided: bool = False,
        margin_factor: float = _MARGIN_FACTOR,
        minimum_margin: float = _MINIMUM_MARGIN_DEG,
        unit: str = "°",
    ):
        self._threshold = float(threshold_deg)
        self._name = name
        self._two_sided = two_sided
        self._factor = float(margin_factor)
        self._floor = float(minimum_margin)
        # 這個類別的數學與單位無關，只有印出來的字要對。肩高那一項用毫米。
        self._unit = unit
        self._posture = Posture.UNKNOWN
        self._frames_in_state = 0

    @property
    def threshold_deg(self) -> float:
        return self._threshold

    @property
    def posture(self) -> Posture:
        return self._posture

    def _settle(self, posture: Posture, reason: str) -> Judgement:
        changed = posture is not self._posture
        self._frames_in_state = 1 if changed else self._frames_in_state + 1
        self._posture = posture
        return Judgement(posture, changed, self._frames_in_state, reason)

    def update(
        self, mean_deg: float | None, standard_error_deg: float | None, ready: bool = True
    ) -> Judgement:
        """收下當下的移動平均，回傳判定。

        ready 是「視窗夠不夠」。還沒填滿時算出來的平均降噪不足，這時候判定
        等於拿一個比宣稱更吵的數字去比門檻。
        """
        if mean_deg is None:
            return self._settle(Posture.UNKNOWN, "沒有可用的量測")
        if not ready:
            return self._settle(Posture.UNKNOWN, "移動平均的視窗還沒填滿")

        value = abs(mean_deg) if self._two_sided else mean_deg
        # 誤差算不出來（視窗裡只有一個值）時退回下限，不能當成零。
        margin = max(self._floor, self._factor * (standard_error_deg or 0.0))
        shown = (f"{value:+.1f}{self._unit} ± {margin:.1f}{self._unit}，"
                 f"門檻 {self._threshold:.0f}{self._unit}")

        if value - margin > self._threshold:
            return self._settle(Posture.OVER, f"{self._name} {shown}")
        if value + margin < self._threshold:
            return self._settle(Posture.OK, f"{self._name} {shown}")
        # 判斷不出來，維持原狀。這就是遲滯。
        return self._settle(self._posture, f"{self._name} {shown}，在門檻的誤差範圍內")


class PostureJudge:
    """θ_CA 與 θ_sym 一起判定。

    任一項超標就算超標，因為兩者量的是不同的問題：θ_CA 是前傾，θ_sym 是肩膀歪。
    """

    def __init__(
        self,
        theta_ca_threshold_deg: float = _THETA_CA_THRESHOLD_DEG,
        theta_sym_threshold_deg: float = _THETA_SYM_THRESHOLD_DEG,
        margin_factor: float = _MARGIN_FACTOR,
    ):
        # θ_CA 只看前傾。往後靠不是這個系統要提醒的事，前作的門檻也是單邊的。
        self.theta_ca = AngleJudge(
            theta_ca_threshold_deg, "θ_CA", two_sided=False, margin_factor=margin_factor
        )
        self._factor = margin_factor
        # θ_sym 往左往右都算歪。
        self.theta_sym = AngleJudge(
            theta_sym_threshold_deg, "θ_sym", two_sided=True, margin_factor=margin_factor
        )
        # 肩高是第三項，門檻由個人基準的晃動量決定，所以建構時還不知道。
        # 沒給門檻就完全不參與判定，於是舊的基準檔與舊的 CSV 行為不變。
        self._drop: AngleJudge | None = None
        self._posture = Posture.UNKNOWN
        self._frames_in_state = 0

    @property
    def posture(self) -> Posture:
        return self._posture

    @property
    def shoulder_drop(self) -> AngleJudge | None:
        return self._drop

    def watch_shoulder_drop(self, threshold_mm: float | None) -> None:
        """把肩高這一項打開，門檻取自這個人自己的晃動量。

        呼叫端只有拿得到基準的時候才會呼叫，所以「沒有基準」與「基準沒有
        肩高」兩種情況都自動退化成不判這一項。
        """
        self._drop = None if threshold_mm is None else AngleJudge(
            threshold_mm, "肩高", two_sided=False,
            margin_factor=self._factor,
            minimum_margin=_MINIMUM_DROP_MARGIN_MM, unit="mm",
        )

    def update(self, ca_window, sym_window, drop_window=None):
        """收下移動平均，回傳 (θ_CA, θ_sym, 肩高, 合併)。

        肩高那一項在沒開或沒資料時是 None，而 None 不參與合併，所以既有的
        記錄重播出來的結果與加這一項之前完全一樣。

        回傳四個而不是三個，呼叫端一律寫 `*_, verdict =`：再加第四項判定
        的時候就不必回頭改每一個呼叫點。
        """
        ca = self.theta_ca.update(
            ca_window.mean, ca_window.standard_error, ca_window.is_full
        )
        sym = self.theta_sym.update(
            sym_window.mean, sym_window.standard_error, sym_window.is_full
        )
        drop = None
        if self._drop is not None and drop_window is not None:
            drop = self._drop.update(
                drop_window.mean, drop_window.standard_error, drop_window.is_full
            )

        parts = [j for j in (ca, sym, drop) if j is not None]
        if any(j.posture is Posture.OVER for j in parts):
            combined, reason = Posture.OVER, "、".join(
                j.reason for j in parts if j.posture is Posture.OVER
            )
        elif all(j.posture is Posture.OK for j in parts):
            combined, reason = Posture.OK, f"{len(parts)} 項都在門檻內"
        else:
            # 有一項還不知道就不能說正常，另一項正常不代表整體正常。
            combined, reason = Posture.UNKNOWN, "、".join(
                j.reason for j in parts if j.posture is Posture.UNKNOWN
            )

        changed = combined is not self._posture
        self._frames_in_state = 1 if changed else self._frames_in_state + 1
        self._posture = combined
        return ca, sym, drop, Judgement(combined, changed, self._frames_in_state, reason)
