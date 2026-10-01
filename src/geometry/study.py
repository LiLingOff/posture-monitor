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

# 每種姿勢的指導語。只有坐正與前傾寫死，其他名稱照樣跑得動，只是指導語要
# 口頭給。姿勢種類還沒定案，程式不該先把清單釘死。
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
    # θ_sym 的門檻 5° 沿用自前作，而到 2026-10-01 為止我們自己一次都沒有驗過
    # 它會不會動：坐正 +1.05°、軀幹前傾 +2.56°，兩種條件都沒有刻意讓肩膀一高
    # 一低。三軸裡只有肩高有靈敏度證據，就是因為沒有一個條件瞄準 θ_sym。
    #
    # 用單側聳肩而不是「手撐著頭側傾」，理由與 head-forward 相同：驗一軸的
    # 靈敏度要把那一軸單獨挑出來。手撐著頭會同時改變頭的位置、擋住一邊的
    # 肩膀，三軸一起動，分不出是哪一項在反應。真實的不良坐姿之後可以另外
    # 加一個條件，但那是驗「抓不抓得到」，不是驗靈敏度。
    #
    # 固定抬右肩，因為 θ_sym 有正負號，同一側才比得了跨受試者的符號一致性。
    # 哪一側離相機遠會影響遮擋，所以模組擺的那一側要一併記錄。
    "shoulder-tilt": "請把右邊肩膀往上聳起來，左邊肩膀不要動。"
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

    @property
    def too_many_rejected(self) -> bool:
        return self.rejection_rate > REJECTION_LIMIT


def review(recordings) -> list[SegmentReview]:
    """把幾段錄製整理成「哪一段能用」。

    判斷寫在這裡而不是印出來的地方，因為同一條政策（略過率上限）另外有三個
    使用者：取基準時的警告、`cohort` 的統計、報表上的 ⚠ 標記。四處各自寫一遍
    的話，總結可以說某一段乾淨而彙整其實把它排除掉了。
    """
    return [
        SegmentReview(
            condition=r.condition, trial=r.trial,
            theta_ca_deg=r.session.mean_ca, usable=r.usable,
            frames=r.frames, rejection_rate=r.rejection_rate,
        )
        for r in recordings
    ]
