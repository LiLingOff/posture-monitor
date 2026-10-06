"""錄製一段量測時累積的東西，以及三件與硬體打交道的政策。

這裡沒有相機、沒有終端機、沒有 argparse。錄製的迴圈本身留在 `posture.py`，
因為它要同時抓相機、畫終端機、寫檔，那是應用層的工作；但它累積的資料、
以及「相機不見了算不算」「同一個原因連續略過要不要提示」「多久存一張畫面」
這三項判斷與 CLI 無關，放在這裡才測得到。

搬過來的直接理由：`Recording` 用 `@dataclass` 宣告在一個用檔案路徑載入的腳本
裡會踩到 `sys.modules` 的問題（見 `tests/posture_loader.py`），而那個 shim
本來是為了測檔名與編號邏輯才存在的。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from calibration.capture import CameraReadError, describe_camera_loss

from .baseline import group_rejection_reason
from .judgement import windows_are_stale
from .pipeline import rejection_advice
from .smoothing import RollingAngle

# 相機掉線之後每次讀取都會立刻失敗，不擋的話迴圈會全速空轉。實機那次在
# 110 秒內寫了 162485 筆空記錄。每次失敗之間停一下，累積到這個次數就停止量測。
READ_RETRY_PAUSE_S = 0.1
MAX_CONSECUTIVE_READ_FAILURES = 25
# 同一個原因連續略過這麼多幀就提示一次。實機約 5fps，15 幀是三秒，
# 短到還來得及調整，長到不會被零星的失誤觸發。
STUCK_AFTER_FRAMES = 15


class Session:
    """整段量測收下的角度。

    移動視窗只留最近 N 個，結尾的摘要需要的是全部。CSV 也有，但要看一眼結果
    就得再開一個程式，那不合理。
    """

    def __init__(self):
        self.ca: list[float] = []
        self.sym: list[float] = []

    def add(self, corrected) -> None:
        if corrected[0] is not None:
            self.ca.append(corrected[0])
        if corrected[1] is not None:
            self.sym.append(corrected[1])

    @property
    def mean_ca(self) -> float | None:
        """整段的 θ_CA 平均。沒有資料時回傳 None 而不是 0，0 是合法的角度。"""
        return float(np.mean(self.ca)) if self.ca else None


@dataclass
class Recording:
    """錄製一段之後留下來的東西。

    回傳而不是直接印出來，因為 `study` 要把好幾段收集起來最後一起比較，
    而 `live` 只要印一段。判斷「這一段能不能用」的邏輯也只該寫一次。
    """

    session: Session
    ca_window: RollingAngle
    sym_window: RollingAngle
    rejected: int
    frames: int
    transitions: list = field(default_factory=list)
    camera_lost: str | None = None
    log_path: Path | None = None
    condition: str = ""      # study 用，live 留空
    trial: int | None = None
    # 可用幀的方位角。拿來跟基準比，看受試者在這一段裡有沒有轉身。
    azimuths: list = field(default_factory=list)

    @property
    def rejection_rate(self) -> float:
        """略過的比例。分母是實際處理過的幀數。

        不用「略過 + 算得出角度的幀數」：通過品質檢查但 θ_CA 算不出來的幀
        兩邊都不算，分母會少掉它們，比例就偏高。同一次錄製曾經因此印出
        9% 與 10% 兩個數字。
        """
        return 0.0 if self.frames == 0 else self.rejected / self.frames

    @property
    def usable(self) -> int:
        return len(self.session.ca)


class CameraRetry:
    """連續讀取失敗的計數，以及「還要不要再試」這個判斷。

    這段先前在三個迴圈裡各寫了一次（錄製、取基準、monitor）。改了其中一個的
    重試預算，另外兩個不會跟著動，而那個預算是 2026-09-29 空轉寫下 162485 筆
    空記錄之後才訂出來的。

    「相機不見了要怎麼辦」留給呼叫端：錄製那邊要保住已經收到的資料所以跳出
    迴圈，取基準那邊沒有資料可保所以直接結束程式。
    """

    def __init__(self, index, pause_s: float = READ_RETRY_PAUSE_S):
        self._index = index
        # 間隔開放給測試調成 0。寫死的話光是驗證重試預算就要真的睡十秒。
        self._pause_s = pause_s
        self._failures = 0

    @property
    def failures(self) -> int:
        return self._failures

    def ok(self) -> None:
        """讀到了。偶爾掉一幀是正常的，所以要歸零而不是累加。"""
        self._failures = 0

    def failed(self, exc) -> str | None:
        """收下這一幀的例外。相機不見了就回傳說明，還有機會就停一下並回傳 None。

        不是讀取失敗（例如偵測不到人）的話不計數：那種失敗重試沒有意義，
        但也不代表相機掉線。
        """
        if not isinstance(exc, CameraReadError):
            return None
        self._failures += 1
        lost = camera_lost(self._index, self._failures)
        if lost is None:
            # 相機掉線之後每次讀取都會立刻失敗，不停一下的話迴圈會全速空轉。
            time.sleep(self._pause_s)
        return lost


def camera_lost(index, failures: int) -> str | None:
    """連續讀取失敗到這個程度就當成相機不見了，回傳該印的話。

    偶爾掉一幀是正常的，相機不見了則重試多少次都一樣。分不開的話只有兩種
    壞法：太早放棄，或像實機那次一樣空轉到把有效資料埋掉。
    """
    if failures < MAX_CONSECUTIVE_READ_FAILURES:
        return None
    return describe_camera_loss(index, failures, failures * READ_RETRY_PAUSE_S)


class StuckWatcher:
    """一直因為同一件事被略過時提示一次該怎麼辦。

    逐幀那一行只說「哪裡不對」，而且會被下一幀蓋掉，所以連續略過三十秒看到的
    是同一句話重複閃動，沒有人知道該動什麼。原因分組時把數字換掉，因為
    「垂直視差 8.9px」與「8.4px」是同一件事。

    提示每一段只印一次。反覆印會把它變成跟原本那一行一樣的背景雜訊。
    """

    def __init__(self, after: int = STUCK_AFTER_FRAMES):
        self._after = after
        self._key: str | None = None
        self._run = 0
        self._told = False

    def saw(self, reason: str | None) -> str | None:
        """收下這一幀的略過原因（沒被略過就傳 None），回傳該印的提示。"""
        if reason is None:
            self._key, self._run, self._told = None, 0, False
            return None
        key = group_rejection_reason(reason)
        if key != self._key:
            self._key, self._run, self._told = key, 1, False
        else:
            self._run += 1
        # 門檻只在這裡判斷一次。分到兩個分支去判斷的話，一段的第一幀會被
        # 漏掉，提示就永遠晚一幀，而 after=1 這種設定會完全不合預期。
        if self._told or self._run < self._after:
            return None
        self._told = True
        advice = rejection_advice(reason)
        head = f"連續 {self._run} 幀都是同一個原因：{reason}"
        return head + ("\n  " + advice if advice else "")


def snapshot_name(frame: int, elapsed_s: float, theta_ca_deg: float | None,
                  posture: str | None) -> str:
    """快照的檔名。要能不開 CSV 就看出這一張是什麼時候、量到多少、判成什麼。

    檔名排序要與時間順序一致，所以幀號補零；角度帶正負號，因為 θ_CA 的符號
    就是前傾與後仰的差別。角度算不出來時寫 na，不寫 0，0 是合法的角度。
    """
    angle = "na" if theta_ca_deg is None else f"{theta_ca_deg:+06.1f}"
    verdict = {"超標": "over", "正常": "ok", "未知": "unknown"}.get(posture or "", "none")
    return f"{frame:06d}_t{elapsed_s:05.1f}s_ca{angle}_{verdict}.png"


class Snapshots:
    """每隔一段時間存一張左眼畫面。

    姿勢事後查證不了，所以要留下畫面。2026-09-29 連續兩次量測的結論都被推翻：
    一次是方位角在過程中漂了三十幾度，一次是受試者不自覺前傾而事後才想起來。
    兩次都不是程式的問題，而是「請坐正兩分鐘」這件事沒有留下任何獨立記錄，
    所以量到的數字既不能當誤報率、也不能當正確偵測的證據。

    存左眼就夠。要確認的是姿勢而不是立體配對，而且一半的張數換一半的磁碟。
    """

    def __init__(self, out_dir: Path | None, every_s: float):
        self._dir = out_dir
        self._every = every_s
        self._next_at = 0.0

    @property
    def enabled(self) -> bool:
        return self._dir is not None and self._every > 0

    def maybe_save(self, frame_index, elapsed_s, left_frame, left_kp,
                   theta_ca_deg, posture) -> str | None:
        """時間到了就存一張，回傳檔名。"""
        if not self.enabled or elapsed_s < self._next_at:
            return None
        self._next_at = elapsed_s + self._every
        name = snapshot_name(frame_index, elapsed_s, theta_ca_deg, posture)
        write_frame(left_frame, self._dir / name, left_kp)
        return name


def write_frame(frame, path: Path, keypoints=None) -> None:
    """把一張畫面寫成 png，骨架與關鍵點疊上去。

    繪製與即時視窗共用 `view.overlay`，所以事後翻出來的快照與當場看到的畫面
    長得一樣。快照存在的理由就是當作獨立的記錄，兩邊畫得不一樣的話對不起來。
    存檔的版本多標名字，因為它是拿來查「哪一點跑掉」的。

    cv2 與 view 都延遲匯入。`analyse` 與 `cohort` 不碰畫面，不該為了它們付
    OpenCV 的載入時間。
    """
    import cv2

    from view import overlay

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = frame.copy()
    overlay.draw_bones(canvas, keypoints, 1.0)
    overlay.draw_keypoints(canvas, keypoints, 1.0)
    overlay.label_keypoints(canvas, keypoints, 1.0)
    cv2.imwrite(str(path), canvas)


def forget_if_stale(
    ca_window: RollingAngle, sym_window: RollingAngle, consecutive_misses: int,
    *more: RollingAngle,
) -> None:
    """偵測連續失敗久了就把移動平均清掉，讓狀態回到未知。

    不清的話，受試者離開座位之後裝置會對著空椅子繼續回報上一個狀態。

    後面的視窗用 *more 收：判定軸會再增加，而漏掉其中一個的後果是那一項
    自己繼續用過期的資料判定，合併之後看起來完全正常。
    """
    if windows_are_stale(consecutive_misses, ca_window.window):
        ca_window.clear()
        sym_window.clear()
        for window in more:
            window.clear()
