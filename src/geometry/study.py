"""實驗流程裡不碰硬體的那一半：檔案要放哪、第幾次、指導語怎麼念、哪一段不能用。

`posture.py study` 的迴圈需要相機、模型與終端機，那留在應用層。這裡是它的
規則部分，沒有相機、沒有 argparse，所以測得起來不必載入整個 CLI。

流程本身就是資料品質的一部分。2026-09-29 那晚四次量測作廢三次，沒有一次是
程式算錯：解析度不符、基準隔了九分鐘、量測中轉頭看螢幕、受試者不自覺前傾。
把流程寫進程式，做錯就變難，而寫在這裡的每一條都對應那晚的一次失敗。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .baseline import REJECTION_LIMIT

# 方位角相對基準漂超過這麼多度就算轉身。方位角是從雙肩連線算的，受試者轉身
# 就會跟著變；轉身之後扣基準不再有意義，而且遠側肩膀開始被擋住。
# 2026-09-29 與 2026-10-06 各有一段因為這個作廢，兩次都是事後才發現。
TURN_LIMIT_DEG = 10.0

# 每種姿勢的指導語。2026-10-06 定案的五種：upright、head-forward、head-back、
# left-shoulder-up、right-shoulder-up。forward 不在清單裡，留著是因為 2026-10-01
# 的資料用了它，而且它是唯一會動到肩部垂直位移的條件。清單以外的名稱照樣
# 跑得動，只是指導語要口頭給。
#
# 這裡**不要**用 Markdown 的強調語法。這幾句是逐字念給受試者聽的，而終端機
# 不會把 ** 變成粗體，只會原樣印出來，念的人就跟著把星號念進去了。
_BRIEFS = {
    "upright": "請坐正：背部貼著椅背、雙腳平放地面、下巴微收、"
               "視線看向前方牆上的固定點。全程不要看螢幕。",
    "forward": "請刻意前傾：上半身往前，但不要轉身，視線仍然看向同一個固定點。",
    # 這一條與 forward 量的不是同一件事，2026-10-01 的實機資料逼出來的：
    # forward 是軀幹整個倒下去，肩膀跟著下沉 58mm，但耳朵與肩膀一起移動，
    # 所以 θ_CA 幾乎沒反應。θ_CA 量的是頭相對於軀幹，要驗它就需要一個
    # 肩膀不動、只有頭往前跑的條件。
    "head-forward": "肩膀不要動，只把下巴往前推出去，像看不清楚螢幕那樣。"
                    "背部維持貼著椅背，視線仍然看向同一個固定點。",
    # 頭部後仰。θ_CA 往負的方向走，肩膀不動，與 head-forward 對稱。
    "head-back": "肩膀不要動，把頭往後仰、下巴往上抬，像要看牆上比較高的地方。"
                 "背部維持貼著椅背。",
    # 左右肩較高各一個條件。θ_sym 有正負號（右肩較高為正、左肩較高為負），
    # 兩邊各量一次才看得出符號對不對，也才知道離相機遠的那一側會不會因為
    # 遮擋而量不準。
    #
    # 用單側聳肩而不是「手撐著頭側傾」：驗一軸的靈敏度要把那一軸單獨挑出來。
    # 手撐著頭會同時改變頭的位置、擋住一邊的肩膀，三軸一起動就分不出是哪一項
    # 在反應。
    "left-shoulder-up": "請把左邊肩膀往上聳起來，右邊肩膀不要動。"
                        "頭與上半身維持原位，視線仍然看向同一個固定點。",
    "right-shoulder-up": "請把右邊肩膀往上聳起來，左邊肩膀不要動。"
                         "頭與上半身維持原位，視線仍然看向同一個固定點。",
}


def subject_dir(root: Path, subject: str) -> Path:
    """一位受試者的資料夾。"""
    return Path(root) / subject


def baseline_path(root: Path, subject: str) -> Path:
    """這位受試者這次的基準檔。每個 session 一份，與姿勢無關。"""
    return subject_dir(root, subject) / "baseline.json"


def session_paths(root: Path, subject: str, condition: str, trial: int):
    """一次量測的 CSV 與快照資料夾。

    檔名自己說得出是誰、哪種姿勢、第幾次，因為彙整時靠中繼資料而不是檔名，
    但人在檔案總管裡找東西還是靠檔名。

    只回傳真的跟 condition/trial 有關的兩個路徑。基準檔是每位受試者一份，
    先前把它塞在同一個回傳值裡，取基準那邊只好傳假的 condition 進來拿它。
    """
    folder = subject_dir(root, subject)
    stem = f"{condition}-{trial}"
    return folder / f"{stem}.csv", folder / f"{stem}-shots"


def next_trial(root: Path, subject: str, condition: str) -> int:
    """這個人這種姿勢已經量過幾次了。

    自動接續而不是每次都從 1 開始，因為撞名是 2026-09-29 真的發生過的事，
    而當時受試者正坐著等。
    """
    folder = subject_dir(root, subject)
    if not folder.is_dir():
        return 1
    used = set()
    for path in folder.glob(f"{condition}-*.csv"):
        tail = path.stem[len(condition) + 1:]
        if tail.isdigit():
            used.add(int(tail))
    return max(used) + 1 if used else 1


def parse_conditions(text: str) -> list[str]:
    """把 `--conditions` 的逗號清單拆開。

    空的就是設定錯了，早點說比開了相機才說好。順序保留：受試者的姿勢是按
    指定的順序量的，而疲勞會隨時間累積，所以順序本身是實驗設計的一部分。
    """
    conditions = [c.strip() for c in text.split(",") if c.strip()]
    if not conditions:
        raise ValueError("至少要有一種姿勢，例如 upright,forward")
    return conditions


def condition_brief(condition: str) -> str:
    """這種姿勢的受試者指導語。"""
    return _BRIEFS.get(
        condition, f"請維持「{condition}」的姿勢，視線看向前方的固定點。"
    )


@dataclass(frozen=True)
class SegmentReview:
    """一段量完之後，這一段能不能用。"""

    condition: str
    trial: int | None
    theta_ca_deg: float | None
    usable: int
    frames: int
    rejection_rate: float
    # 這一段的平均方位角減掉基準的方位角。沒有基準或量不到方位角時是 None。
    turned_deg: float | None = None
    camera_lost: bool = False
    # 相機掉線之後同一種姿勢在這一趟裡又量了一次，所以這一段是殘缺的那份。
    retaken: bool = False

    @property
    def too_many_rejected(self) -> bool:
        return self.rejection_rate > REJECTION_LIMIT

    @property
    def turned_too_far(self) -> bool:
        return self.turned_deg is not None and abs(self.turned_deg) > TURN_LIMIT_DEG

    @property
    def needs_redo(self) -> bool:
        """要叫人重量。已經在這一趟裡重量過的那份不算，重量的是後面那一份。"""
        if self.retaken:
            return False
        return self.too_many_rejected or self.turned_too_far or self.camera_lost

    @property
    def usable_data(self) -> bool:
        """這一段的 CSV 值不值得丟進 analyse。

        殘缺的那份不值得：相機掉線時往往只留下幾秒，跟後面完整的那份平均在
        一起只會把結果拉偏。
        """
        return not self.retaken


def review(recordings, baseline=None) -> list[SegmentReview]:
    """把幾段錄製整理成「哪一段能用」。

    判斷寫在這裡而不是印出來的地方，因為同一條政策（略過率上限）另外有三個
    使用者：取基準時的警告、`cohort` 的統計、報表上的 ⚠ 標記。四處各自寫一遍
    的話，總結可以說某一段乾淨而彙整其實把它排除掉了。「哪一段被重量過」
    同理：先前它寫在印出來的地方，而那正是這段說明在講的事。
    """
    later = [r.condition for r in recordings]
    return [
        SegmentReview(
            condition=r.condition, trial=r.trial,
            theta_ca_deg=r.session.mean_ca, usable=r.usable,
            frames=r.frames, rejection_rate=r.rejection_rate,
            turned_deg=_turned(r, baseline),
            camera_lost=r.camera_lost is not None,
            retaken=(r.camera_lost is not None
                     and r.condition in later[i + 1:]),
        )
        for i, r in enumerate(recordings)
    ]


def _turned(recording, baseline) -> float | None:
    """這一段相對取基準時轉了多少。

    方位角取的是絕對值（0~90），轉過正面到另一側只會先變小再變大，所以
    這個差值看得出轉了多少，看不出往哪邊轉。要知道方向，看是哪一側肩膀在
    配錯。
    """
    if baseline is None or not recording.azimuths:
        return None
    return float(sum(recording.azimuths) / len(recording.azimuths)
                 - baseline.azimuth_deg)
