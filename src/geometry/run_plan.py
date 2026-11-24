"""引導模式（`posture.py run-plan`）裡不碰硬體的那一半：計畫檔、當場檢查、作廢。

2026-11-19 S01 的自測暴露的問題幾乎都是操作，而且都是事後判讀才發現，那時人
已經走了：側傾六段都沒有輸入量角器、兩趟的 `desk_offset_cm` 都記成 10、
側傾轉身、基準和接著的端正坐差十幾度卻沒有當場重取。組員一次只能來一小時，
所以這些檢查要在量完那一段的當下就做，而且要讓操作者照著畫面就能跑完，
不必翻手冊、不必記指令。

計畫檔只描述「要跑哪幾趟」，每一趟最後還是組成一條 `study` 指令，量測方法
與 `study` 完全相同；這裡只多了流程控制。門檻沿用既有的值，不新訂：
略過率 15%（`baseline.REJECTION_LIMIT`）、轉身 10°（`study.TURN_LIMIT_DEG`），
基準與端正坐差 5° 是文件 09 與自測手冊既有的重取規則。
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .baseline import REJECTION_LIMIT
from .study import DESK_LEVELS, TURN_LIMIT_DEG, UPRIGHT, is_tilt

# 取完基準接著量的那段端正坐，扣基準後的 θ_CA 超過這麼多度就建議重取基準。
# 文件 09「下次量測前」與自測手冊第四節的規則（設計值，沒有實機依據）。
# 2026-11-18 掃 25 份基準，只有 9 份在 5° 內，所以這一條會常常觸發，
# 每一趟只問一次，避免把一小時的組員時間全花在重取上。
BASELINE_UPRIGHT_LIMIT_DEG = 5.0

# 一段最多當場重量幾次。重量一次還是不過，多半是受試者或架設的問題，
# 再量也只是讓下一位受試者等，記下來交給事後判讀。
MAX_REDO_PER_SEGMENT = 1


@dataclass(frozen=True)
class PlanTrip:
    """計畫裡的一趟：同一次坐下、不起身、自己取一份基準。"""

    trip_id: str
    title: str
    before: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()
    target_angles: tuple[float, ...] = ()
    desk_level: str | None = None
    task: str | None = None
    setup: str = ""                    # 開始前操作者要擺好的東西
    subject_does: str = ""             # 念給受試者聽、或讓受試者自己看的
    optional: bool = False
    optional_note: str = ""            # 什麼情況下可以跳過

    @property
    def segments(self) -> tuple[str, ...]:
        return self.before + self.conditions

    @property
    def tilts(self) -> int:
        return sum(1 for c in self.segments if is_tilt(c))


@dataclass(frozen=True)
class Plan:
    name: str
    trips: tuple[PlanTrip, ...]
    seconds_each: float = 60.0
    countdown: int = 20
    baseline_seconds: float = 30.0
    snapshot_raw: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)


def _strings(value, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{where} 要是字串清單")
    out = tuple(v.strip() for v in value)
    if any(not v for v in out):
        raise ValueError(f"{where} 裡有空的名稱")
    return out


def _numbers(value, where: str) -> tuple[float, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{where} 要是數字清單")
    out = []
    for v in value:
        # bool 是 int 的子類別，true 不能被當成 1°。
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{where} 要是數字清單，收到 {v!r}")
        out.append(float(v))
    return tuple(out)


def _text(value, where: str) -> str:
    if value is None:
        return ""
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return "\n".join(value)
    if not isinstance(value, str):
        raise ValueError(f"{where} 要是字串")
    return value


def _flag(value, default: bool, where: str) -> bool:
    """JSON 的 true／false。寫成字串 "false" 的話 bool() 會變成 True，意思整個反過來，
    所以不轉型，直接擋（GPT 審查第 6 條）。"""
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{where} 要寫 true 或 false（不加引號），收到 {value!r}")
    return value


def _optional_name(value, where: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} 要是非空字串，收到 {value!r}")
    return value.strip()


def _positive(value, default, where: str):
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{where} 要是正數，收到 {value!r}")
    return value


def parse_plan(data) -> Plan:
    """把計畫檔的 JSON 內容轉成 Plan。內容不對就丟 ValueError，訊息說哪裡錯。

    檢查都在開相機之前做完：受試者坐下之後才發現第三趟的角度少寫一個，
    浪費的是組員的時間。
    """
    if not isinstance(data, dict):
        raise ValueError("計畫檔最外層要是一個物件（{ ... }）")
    raw_trips = data.get("trips")
    if not isinstance(raw_trips, list) or not raw_trips:
        raise ValueError("計畫檔要有 trips，而且至少一趟")
    trips, seen = [], set()
    for n, raw in enumerate(raw_trips, start=1):
        where = f"第 {n} 趟"
        if not isinstance(raw, dict):
            raise ValueError(f"{where} 要是一個物件")
        trip_id = raw.get("id")
        if not isinstance(trip_id, str) or not trip_id.strip():
            raise ValueError(f"{where} 缺 id")
        trip_id = trip_id.strip()
        if trip_id in seen:
            raise ValueError(f"趟次 id {trip_id!r} 重複")
        seen.add(trip_id)
        where = f"趟次 {trip_id}"
        before = _strings(raw.get("before"), f"{where} 的 before")
        conditions = _strings(raw.get("conditions"), f"{where} 的 conditions")
        if not before and not conditions:
            raise ValueError(f"{where} 的 before 與 conditions 都是空的")
        desk_level = raw.get("desk_level")
        if desk_level is not None and desk_level not in DESK_LEVELS:
            raise ValueError(f"{where} 的 desk_level 要是 {'、'.join(DESK_LEVELS)} 其中之一")
        trip = PlanTrip(
            trip_id=trip_id,
            title=_text(raw.get("title"), f"{where} 的 title") or trip_id,
            before=before, conditions=conditions,
            target_angles=_numbers(raw.get("target_angle_deg"),
                                   f"{where} 的 target_angle_deg"),
            desk_level=desk_level,
            task=_optional_name(raw.get("task"), f"{where} 的 task"),
            setup=_text(raw.get("setup"), f"{where} 的 setup"),
            subject_does=_text(raw.get("subject_does"), f"{where} 的 subject_does"),
            optional=_flag(raw.get("optional"), False, f"{where} 的 optional"),
            optional_note=_text(raw.get("optional_note"), f"{where} 的 optional_note"),
        )
        if trip.target_angles and len(trip.target_angles) != trip.tilts:
            raise ValueError(f"{where} 有 {trip.tilts} 段側傾，target_angle_deg 卻有 "
                             f"{len(trip.target_angles)} 個")
        if trip.tilts and not trip.target_angles:
            raise ValueError(f"{where} 有側傾段落，要給 target_angle_deg")
        trips.append(trip)
    return Plan(
        name=_text(data.get("name"), "name") or "未命名計畫",
        trips=tuple(trips),
        seconds_each=float(_positive(data.get("seconds_each"), 60.0, "seconds_each")),
        countdown=int(_positive(data.get("countdown"), 20, "countdown")),
        baseline_seconds=float(_positive(data.get("baseline_seconds"), 30.0,
                                         "baseline_seconds")),
        snapshot_raw=_flag(data.get("snapshot_raw"), True, "snapshot_raw"),
        notes=tuple(_strings(data.get("notes"), "notes")),
    )


def load_plan(path: Path) -> Plan:
    """讀計畫檔（JSON，UTF-8）。"""
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ValueError(f"讀不到計畫檔 {path}：{exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"計畫檔 {path} 不是合法的 JSON：第 {exc.lineno} 行，"
                         f"{exc.msg}") from exc
    return parse_plan(data)


def _number_text(value: float) -> str:
    return f"{value:g}"


def trip_argv(plan: Plan, trip: PlanTrip, subject: str,
              desk_offset_cm: float | None) -> list[str]:
    """這一趟等同的 `study` 參數（不含 "study" 本身）。

    組成真正的指令列再交給同一個 parser，而不是自己填 Namespace：study 的預設值
    只寫在 parser 裡一處，這裡另抄一份遲早會對不上。也因此操作者看到的
    「等同指令」就是實際跑的那一條。
    """
    argv = ["--subject", subject,
            "--conditions", ",".join(trip.conditions),
            "--seconds", _number_text(plan.baseline_seconds),
            "--seconds-each", _number_text(plan.seconds_each),
            "--countdown", str(plan.countdown)]
    if trip.before:
        argv += ["--before-baseline", ",".join(trip.before)]
    if trip.desk_level is not None:
        argv += ["--desk-level", trip.desk_level]
        if desk_offset_cm is not None:
            argv += ["--desk-offset-cm", _number_text(desk_offset_cm)]
    if trip.task is not None:
        argv += ["--task", trip.task]
    if trip.target_angles:
        argv += ["--target-angle-deg",
                 ",".join(_number_text(a) for a in trip.target_angles)]
    if plan.snapshot_raw:
        argv.append("--snapshot-raw")
    return argv


def estimated_minutes(plan: Plan, trip: PlanTrip) -> float:
    """一趟大概要幾分鐘（不含重量與換人）。給操作者排時間用，只是估計。

    每段：倒數或念指導語約 30 秒、量測 seconds_each；基準：倒數加取樣。
    側傾每段另加約 60 秒量角器（量開始值與結束值）。
    """
    seconds = len(trip.segments) * (30 + plan.seconds_each)
    if trip.conditions:
        seconds += plan.countdown + plan.baseline_seconds
    seconds += trip.tilts * 60
    return seconds / 60.0


@dataclass(frozen=True)
class SegmentCheck:
    """一段量完當場的檢查結果。"""

    problems: tuple[str, ...]
    retake_baseline: bool = False

    @property
    def ok(self) -> bool:
        return not self.problems


def check_segment(review, *, first_upright_after_baseline: bool) -> SegmentCheck:
    """`study.review()` 給的一段，加上「基準與第一段端正坐」這一條。

    相機掉線不在這裡處理：`study` 本身就會重新找相機、同一段重量。
    基準前的段落（自然坐）在量的當下還沒有同趟端正坐可比，轉身要到總結才知道。
    """
    if review.camera_lost:
        return SegmentCheck(())
    problems = []
    if review.too_many_rejected:
        problems.append(f"略過率 {review.rejection_rate * 100:.0f}%，"
                        f"超過 {REJECTION_LIMIT * 100:.0f}%。"
                        "請受試者把肩膀打開、上身不要轉，頭髮不要蓋住耳朵")
    if review.turned_too_far:
        problems.append(f"方位角和參考差 {review.turned_deg:+.0f}°，超過 "
                        f"{TURN_LIMIT_DEG:.0f}°，受試者轉身了。"
                        "重量前提醒他臉和肩膀朝前、盯著牆上的點")
    retake = False
    if (first_upright_after_baseline and review.condition == UPRIGHT
            and review.theta_ca_deg is not None
            and abs(review.theta_ca_deg) > BASELINE_UPRIGHT_LIMIT_DEG):
        retake = True
        problems.append(f"這段端正坐和剛取的基準差 {review.theta_ca_deg:+.1f}°，"
                        f"超過 {BASELINE_UPRIGHT_LIMIT_DEG:.0f}°，這份基準不可靠。"
                        "建議重取基準再量一次這段（受試者不用起身）")
    return SegmentCheck(tuple(problems), retake)


def discard_dir(root: Path, subject: str) -> Path:
    """作廢的段落搬去哪裡。

    不能放在 `root` 底下：`cohort data/sessions` 會遞迴找所有 CSV，作廢的段落
    會被偷偷算進去。放在旁邊的 `<root>-discarded/` 就不會，而且沒刪，
    事後要查「為什麼重量」還找得到。
    """
    root = Path(root)
    return root.parent / f"{root.name}-discarded" / subject


def discard_segment(csv_path: Path, root: Path, subject: str) -> Path:
    """把一段的 CSV 與快照資料夾搬到作廢區，回傳搬過去的 CSV 路徑。

    搬走之後 `next_trial` 會給回同一個編號，重量的那一段沿用原本的檔名，
    趟 1 的漂移檢查那段才會照舊叫 upright-2。
    作廢區已有同名檔（同一段第二次作廢）時加上序號，不覆蓋。
    """
    csv_path = Path(csv_path)
    target_dir = discard_dir(root, subject)
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem
    n = 1
    target = target_dir / csv_path.name
    while target.exists() or (target_dir / f"{target.stem}-shots").exists():
        n += 1
        target = target_dir / f"{stem}.discarded{n}{csv_path.suffix}"
    if csv_path.exists():
        shutil.move(str(csv_path), str(target))
    shots = csv_path.with_name(f"{stem}-shots")
    if shots.is_dir():
        shutil.move(str(shots), str(target_dir / f"{target.stem}-shots"))
    return target


def parse_offset(text: str) -> float | None:
    """操作者輸入的桌面減手肘（公分）。不是數字回傳 None。"""
    text = text.strip().replace("＋", "+").replace("－", "-").replace("cm", "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def describe_offset(value: float) -> str:
    """把正負號講成白話，讓操作者確認沒有寫反。"""
    if value > 0:
        return f"桌面比手肘高 {value:g} 公分"
    if value < 0:
        return f"桌面比手肘低 {-value:g} 公分"
    return "桌面和手肘一樣高"


# ---- 只跑幾趟、接續之前的資料（2026-11-23）--------------------------------
#
# 1123 那天量到一半中斷，晚上要接著做側傾，但 run-plan 每次都從趟 1 開始：
# 整個重來不但多花時間，自然坐也會變成第二次（受試者已經知道流程，不算自然）。
# 所以加 `--trips` 只跑指定的趟。刻意不做「自動從斷掉的地方接」：哪一趟算做完
# 要人判斷（例如做完了但有一段建議重量），程式猜錯的話現場不會有人發現。
# 改成同一個代號已經有資料時先停下來，列出每一趟的狀況與下一步該打的指令。

def select_trips(plan: Plan, text: str) -> tuple[PlanTrip, ...]:
    """`--trips trip4,tiltA` → 計畫裡的那幾趟，照計畫檔的順序。"""
    wanted = [part.strip() for part in text.split(",") if part.strip()]
    if not wanted:
        raise ValueError("--trips 是空的，例如 --trips tiltA")
    known = [t.trip_id for t in plan.trips]
    unknown = [w for w in wanted if w not in known]
    if unknown:
        raise ValueError(f"計畫裡沒有 {'、'.join(unknown)}；"
                         f"可以用的有 {'、'.join(known)}")
    return tuple(t for t in plan.trips if t.trip_id in wanted)


@dataclass(frozen=True)
class ExistingSegment:
    """資料夾裡已經有的一段。rejection_rate 讀不出來時是 None。"""

    condition: str
    trial: int
    rejection_rate: float | None = None
    # 側傾段標頭裡的 target_angle_deg。側傾 A（20°）與側傾 B（10°、30°）的檔名
    # 都是 tilt-left／tilt-right，只能靠它分。
    target_angle: float | None = None


def segment_key(condition: str, target: float | None) -> str:
    """一段的身分：姿勢，側傾再加目標角度（tilt-left@10）。"""
    if is_tilt(condition) and target is not None:
        return f"{condition}@{target:g}"
    return condition


def _trip_keys(trip: PlanTrip) -> tuple[str, ...]:
    """這一趟每一段的身分，照段落順序。側傾段依序對應 target_angle_deg。"""
    keys = []
    tilt_index = 0
    for condition in trip.segments:
        target = None
        if is_tilt(condition):
            if tilt_index < len(trip.target_angles):
                target = trip.target_angles[tilt_index]
            tilt_index += 1
        keys.append(segment_key(condition, target))
    return tuple(dict.fromkeys(keys))


def _label(key: str) -> str:
    """給人看的名稱：tilt-left@10 → tilt-left 10°。"""
    name, _, target = key.partition("@")
    return f"{name} {target}°" if target else name


def _own_keys(plan: Plan, trip: PlanTrip) -> tuple[str, ...]:
    """只有這一趟才有的段。upright 每一趟都有，看不出是哪一趟量的。"""
    others = {k for t in plan.trips if t is not trip for k in _trip_keys(t)}
    keys = _trip_keys(trip)
    own = tuple(k for k in keys if k not in others)
    return own or keys


@dataclass(frozen=True)
class TripProgress:
    trip: PlanTrip
    present: tuple[str, ...]
    missing: tuple[str, ...]
    high_skip: tuple[str, ...]        # 「right-shoulder-up #1 略過 17%」

    @property
    def done(self) -> bool:
        return not self.missing

    @property
    def started(self) -> bool:
        return bool(self.present)


def trip_progress(plan: Plan, segments) -> list[TripProgress]:
    """每一趟做到哪裡。同一種姿勢量過好幾次時看最新的那一次。"""
    latest: dict[str, ExistingSegment] = {}
    for seg in segments:
        key = segment_key(seg.condition, seg.target_angle)
        if key not in latest or seg.trial > latest[key].trial:
            latest[key] = seg
    out = []
    for trip in plan.trips:
        own = _own_keys(plan, trip)
        present = tuple(_label(k) for k in own if k in latest)
        missing = tuple(_label(k) for k in own if k not in latest)
        high = tuple(
            f"{_label(k)} #{latest[k].trial} 略過 {latest[k].rejection_rate:.0%}"
            for k in own
            if k in latest and latest[k].rejection_rate is not None
            and latest[k].rejection_rate > REJECTION_LIMIT)
        out.append(TripProgress(trip, present, missing, high))
    return out


def resume_lines(plan: Plan, progress: list[TripProgress], subject: str,
                 command: str) -> list[str]:
    """同一個代號已經有資料時印給操作者看的話，含下一步可以照打的指令。

    `command` 是不含 --trips 的那一段，例如 `python posture.py run-plan --subject S02`。
    """
    lines = [f"{subject} 已經有量過的資料："]
    for p in progress:
        name = f"{p.trip.title}（{p.trip.trip_id}）"
        if not p.started:
            state = "還沒做"
        elif p.done:
            state = "做完了"
        else:
            state = f"沒做完，還缺 {'、'.join(p.missing)}"
        if p.high_skip:
            state += f"；{'、'.join(p.high_skip)}，建議整趟重量"
        lines.append(f"  ・{name}：{state}")

    todo = [p.trip.trip_id for p in progress if not p.done]
    redo = [p.trip.trip_id for p in progress if p.done and p.high_skip]
    lines.append("")
    if todo:
        lines.append("只補還沒做完的：")
        lines.append(f"  {command} --trips {','.join(todo)}")
    if redo:
        both = [p.trip.trip_id for p in progress
                if p.trip.trip_id in todo or p.trip.trip_id in redo]
        lines.append("連建議重量的一起做：")
        lines.append(f"  {command} --trips {','.join(both)}")
    if not todo and not redo:
        lines.append("每一趟都做完了。真的要重量某一趟，加 --trips，例如：")
        lines.append(f"  {command} --trips {plan.trips[-1].trip_id}")
    if any(p.started and any(c.startswith("natural") for c in p.trip.segments)
           for p in progress):
        lines.append("自然坐已經量過，那一趟非必要不要重量：受試者知道流程後就不算自然坐了。")
    lines.append("如果其實是另一位受試者，換一個沒用過的代號。")
    return lines
