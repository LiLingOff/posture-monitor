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
# left-shoulder-up、right-shoulder-up。2026-11-17 重排趟次之後再加三種：
# natural（自然坐，排在取基準之前）、tilt-left、tilt-right（側傾，量角器當真值）。
# forward 現在是軀幹前傾條件（趟 1），用來驗 θ_KA 與肩部垂直位移：軀幹整個
# 往前倒時這兩項會一起動。head-back 已不列入趟次，但程式保留，舊資料用得到。
# 清單以外的名稱照樣跑得動，只是指導語要口頭給。
#
# 這裡**不要**用 Markdown 的強調語法。這幾句是逐字念給受試者聽的，而終端機
# 不會把 ** 變成粗體，只會原樣印出來，念的人就跟著把星號念進去了。
_BRIEFS = {
    "upright": "請坐正：背部貼著椅背、雙腳平放地面、下巴微收、"
               "視線看向前方牆上的固定點。全程不要看螢幕。",
    # 軀幹前傾。驗 θ_KA（雙髖中點到頸）與肩部垂直位移，兩者在這個條件下會一起動，
    # 所以它們不是兩份獨立的證據。
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
    # 自然坐。排在取基準之前（--before-baseline），不扣基準，與同一趟的端正坐配對。
    # 任務固定為抄寫一段文字、60 秒，螢幕固定在受試者正前方：面對螢幕本身就是
    # 固定點，方位角才穩。要抄的文字由操作者準備，不寫在程式裡。
    "natural": "請像平常一樣使用電腦，面對螢幕抄寫這段文字，不用刻意坐好。",
    # 側傾。操作者用量角器把上半身調到目標角度，記下實測值（--measured-angle-deg，
    # 或開始前程式會問）。θ_sym 右肩較高為正，所以往左側彎時右肩較高，預期為正；
    # 往右側彎預期為負。
    "tilt-left": "請把上半身往左邊側彎，頭與肩膀一起動，不要轉身，"
                 "視線維持在螢幕。我會用量角器幫你調到要的角度，調好之後請保持不動。",
    "tilt-right": "請把上半身往右邊側彎，頭與肩膀一起動，不要轉身，"
                  "視線維持在螢幕。我會用量角器幫你調到要的角度，調好之後請保持不動。",
}

# 端正坐的條件名稱。自然坐靠它找同一趟的對照，轉身檢查也拿它當參考。
UPRIGHT = "upright"
# 桌面相對手肘高度的三種做法。座椅與桌子都不可調，偏高、偏低是墊出來的。
DESK_LEVELS = ("base", "desk-up", "seat-up")
# 自然坐沒給 --task 時記的任務名稱。
DEFAULT_TASK = "copy-text"


def is_natural(condition: str) -> bool:
    """自然坐。名稱以 natural 開頭都算（例如 natural、natural-2）。"""
    return condition == "natural" or condition.startswith("natural-")


def is_tilt(condition: str) -> bool:
    """側傾條件：tilt-left、tilt-right。"""
    return condition in ("tilt-left", "tilt-right")


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


def parse_optional_conditions(text: str | None) -> list[str]:
    """`--before-baseline` 與（有給 --before-baseline 時的）`--conditions`，可以是空的。

    趟 5（表4）只有基準前的段落，`--conditions` 是空的；趟 4 沒有自然坐，
    `--before-baseline` 是空的。兩個都空由呼叫端擋。
    """
    if text is None:
        return []
    return [c.strip() for c in text.split(",") if c.strip()]


def parse_number_list(text: str | None, name: str) -> tuple[float, ...]:
    """逗號分隔的數字清單。不給是空的；有一項不是數字就說是哪一項。"""
    if text is None or not str(text).strip():
        return ()
    out = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            out.append(float(item))
        except ValueError:
            raise ValueError(f"{name} 裡的 {item!r} 不是數字")
    return tuple(out)


@dataclass(frozen=True)
class TripOptions:
    """一趟 study 的中繼資料，寫進每一段 CSV 的標頭最後面。

    trip_id 是同一趟共用的代號（study 開始的時間）。自然坐沒有基準，
    不能靠 baseline_captured_at 找對照，所以配對靠它。

    target_angles、measured_angles、measured_end_angles 依序對應這一趟裡的側傾段落
    （measured_end_angles 是量完那一段之後再讀一次量角器的值，60 秒內上身可能
    慢慢滑掉，只記開始值看不出來）；azimuth_targets 只有一個值時套用到每一段，否則依序對應每一段。
    """

    trip_id: str
    desk_level: str | None = None
    lift_cm: float | None = None
    desk_offset_cm: float | None = None
    task: str | None = None
    target_angles: tuple = ()
    measured_angles: tuple = ()
    measured_end_angles: tuple = ()
    azimuth_targets: tuple = ()


def build_trip_options(trip_id: str, segments: list[str], *, desk_level=None,
                       lift_cm=None, desk_offset_cm=None, task=None,
                       target_angles=(), measured_angles=(),
                       azimuth_targets=(), measured_end_angles=()) -> TripOptions:
    """檢查各選項與段落對不對得上，對不上就丟 ValueError（在開相機之前）。

    segments 是這一趟依序要量的所有段落（基準前的與基準後的接在一起）。
    """
    if desk_level is not None and desk_level not in DESK_LEVELS:
        raise ValueError(f"--desk-level 要是 {'、'.join(DESK_LEVELS)} 其中之一，"
                         f"收到 {desk_level!r}")
    if desk_level == "base":
        if lift_cm not in (None, 0, 0.0):
            raise ValueError("--desk-level base 是原桌椅，--lift-cm 只能是 0 或不給")
        lift_cm = 0.0
    if lift_cm is not None and lift_cm < 0:
        raise ValueError("--lift-cm 是墊高幾公分，不能是負的")
    tilts = sum(1 for c in segments if is_tilt(c))
    for name, values in (("--target-angle-deg", target_angles),
                         ("--measured-angle-deg", measured_angles),
                         ("--measured-end-angle-deg", measured_end_angles)):
        if values and len(values) != tilts:
            raise ValueError(f"{name} 給了 {len(values)} 個值，但這一趟有 {tilts} 段"
                             "側傾（tilt-left、tilt-right）。依序一段一個")
        if any(v < 0 for v in values):
            raise ValueError(f"{name} 記的是量角器上的角度大小，不帶正負號")
    if azimuth_targets and len(azimuth_targets) not in (1, len(segments)):
        raise ValueError(f"--azimuth-target-deg 給了 {len(azimuth_targets)} 個值，"
                         f"要嘛一個（每段都一樣），要嘛 {len(segments)} 個（依序每段一個）")
    return TripOptions(
        trip_id=trip_id, desk_level=desk_level, lift_cm=lift_cm,
        desk_offset_cm=desk_offset_cm, task=task,
        target_angles=tuple(target_angles), measured_angles=tuple(measured_angles),
        measured_end_angles=tuple(measured_end_angles),
        azimuth_targets=tuple(azimuth_targets),
    )


def _g(value: float) -> str:
    return f"{value:g}"


def segment_metadata(options: TripOptions, condition: str, segment_index: int,
                     tilt_index: int | None,
                     measured_angle: float | None = None) -> dict[str, str]:
    """一段的標頭中繼資料，順序固定。沒有值的鍵不放（標頭裡整列不寫）。

    segment_index 是這一段在整趟裡的位置（從 0 起算），tilt_index 是它在側傾
    段落裡的位置（不是側傾就是 None）。measured_angle 是當場問到的實測值，
    沒給 --measured-angle-deg 時用它。
    """
    meta: dict[str, str] = {"trip_id": options.trip_id}
    if options.desk_level is not None:
        meta["desk_level"] = options.desk_level
    if options.lift_cm is not None:
        meta["lift_cm"] = _g(options.lift_cm)
    if options.desk_offset_cm is not None:
        meta["desk_offset_cm"] = _g(options.desk_offset_cm)
    if is_natural(condition):
        meta["task"] = options.task or DEFAULT_TASK
    if tilt_index is not None:
        if options.target_angles:
            meta["target_angle_deg"] = _g(options.target_angles[tilt_index])
        measured = (options.measured_angles[tilt_index] if options.measured_angles
                    else measured_angle)
        if measured is not None:
            meta["measured_angle_deg"] = _g(measured)
    if options.azimuth_targets:
        value = (options.azimuth_targets[0] if len(options.azimuth_targets) == 1
                 else options.azimuth_targets[segment_index])
        meta["azimuth_target_deg"] = _g(value)
    return meta


def end_angle_for(options: TripOptions, tilt_index: int | None) -> float | None:
    """--measured-end-angle-deg 事先給的結束值；沒給或不是側傾段落就是 None。

    量角器的結束值是量完那一段之後才讀得到，所以多半是當場問（posture.py），
    這裡只管事先給的那一種。
    """
    if tilt_index is None or not options.measured_end_angles:
        return None
    return options.measured_end_angles[tilt_index]


def end_angle_footer(value: float) -> str:
    """寫在 CSV 檔尾的一列。標頭在開始時就寫完了，結束值是事後才知道的，
    與 camera_lost 同一個做法：`read_rows` 收任何位置的 `# key=value`。"""
    return f"# measured_end_angle_deg={_g(value)}\n"


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
    # θ_KA（扣基準）的平均。基準沒有這一項、或髖一直被擋住時是 None。
    theta_ka_deg: float | None = None
    # 方位角差要不要拿來判轉身。趟 5（表4）是刻意搬動相機，差多少都不算轉身。
    turn_checked: bool = True

    @property
    def too_many_rejected(self) -> bool:
        return self.rejection_rate > REJECTION_LIMIT

    @property
    def turned_too_far(self) -> bool:
        return (self.turn_checked and self.turned_deg is not None
                and abs(self.turned_deg) > TURN_LIMIT_DEG)

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
    out = []
    for i, r in enumerate(recordings):
        reference, checked = _reference_azimuth(r, recordings, baseline)
        out.append(SegmentReview(
            condition=r.condition, trial=r.trial,
            theta_ca_deg=r.session.mean_ca,
            theta_ka_deg=r.session.mean_ka, usable=r.usable,
            frames=r.frames, rejection_rate=r.rejection_rate,
            turned_deg=_turned(r, reference),
            camera_lost=r.camera_lost is not None,
            retaken=(r.camera_lost is not None
                     and r.condition in later[i + 1:]),
            turn_checked=checked,
        ))
    return out


def _mean_azimuth(recording) -> float | None:
    if not recording.azimuths:
        return None
    return float(sum(recording.azimuths) / len(recording.azimuths))


def _reference_azimuth(recording, recordings, baseline) -> tuple[float | None, bool]:
    """轉身檢查的參考方位角，以及這個差值要不要拿來判轉身。

    - 基準之後的段落：取基準時的方位角（與先前相同）。
    - 基準之前的段落（自然坐）沒有基準：改用同一趟端正坐第一段的平均方位角；
      這一趟沒有端正坐（或它一幀方位角都沒有）就不標轉身。
    - 整趟沒有基準（趟 5，表4）：用這一趟第一段的方位角當參考，差值照列，
      但不判轉身，因為方位角改變是刻意搬動相機造成的。
    """
    if not getattr(recording, "before_baseline", False):
        return (None if baseline is None else baseline.azimuth_deg), True
    if baseline is None:
        # 整趟沒有基準：就算這一趟裡有 upright 也一樣不判轉身（GPT 審查第 1 條）。
        first = next((r for r in recordings if r.azimuths), None)
        return (None if first is None else _mean_azimuth(first)), False
    for i, other in enumerate(recordings):
        if other.condition != UPRIGHT or not other.azimuths:
            continue
        # 掉線之後同一種姿勢又重量了，殘缺的那份不當參考（審查第 5 條）。
        if other.camera_lost is not None and any(
                r.condition == UPRIGHT for r in recordings[i + 1:]):
            continue
        return _mean_azimuth(other), True
    return None, True


def _turned(recording, reference: float | None) -> float | None:
    """這一段相對參考方位角轉了多少。

    方位角取的是絕對值（0~90），轉過正面到另一側只會先變小再變大，所以
    這個差值看得出轉了多少，看不出往哪邊轉。要知道方向，看是哪一側肩膀在
    配錯。
    """
    mean = _mean_azimuth(recording)
    if reference is None or mean is None:
        return None
    return float(mean - reference)
