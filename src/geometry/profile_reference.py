"""同步正側面參考：用手機在受試者正側 90° 拍的畫面，算每一段 θ_CA 的獨立參考值。

為什麼需要：同一個人「端正坐」在段與段之間 θ_CA 差 10～20°，光看雙目的數字，
分不出是人的姿勢變了，還是儀器有誤差。正側面時，耳與肩膀的左右間距沿著視線方向，
被投影掉，影像上的耳肩向量就在矢狀面內，它與鉛垂線的夾角就是 θ_CA 要量的東西。
每一段同時有這個參考值，「3D 減參考」裡人的姿勢變動就抵消了。

限制（也寫在報告裡）：
- 參考值用的是同一個關鍵點偵測模型，模型對耳朵位置的一致偏差，兩邊會一起偏，
  這個方法抓不到；它抓得到的是三角測量與雙目幾何造成的誤差。
- 手機要擺在正側 90°，相機水平。擺歪的量用 `--plumb` 給畫面裡鉛垂線的兩點修正，
  不能修正擺位的方位角誤差（偏離正側的角度會讓左右間距漏進影像）。
- 近側要與雙目那邊用的一側一致。

這個模組不碰 cv2、終端機與 argparse，影片讀取在 posture.py。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pose.keypoints import PersonKeypoints

from .session_analysis import read_rows
from .uncertainty import standard_error

SIDES = ("right", "left")

# 影像座標 y 向下，所以「上」是 (0, -1)。
UP = np.array([0.0, -1.0])

# 耳肩向量短於這個長度（像素）時角度沒有意義。手機在 2m 處，耳肩距離約有 100px 以上。
_MIN_VECTOR_PX = 10.0
# 鼻與耳的水平差短於這個長度時，判斷不出受試者面向哪一邊。
_MIN_FACING_PX = 3.0
# 一段裡面向同一個方向的幀要佔這個比例以上，否則標警告（可能轉頭或偵測不穩）。
_MIN_FACING_AGREEMENT = 0.8
# 至少要有這麼多可用幀，一段的平均才算數。
_MIN_USABLE_FRAMES = 5


def parse_plumb(text: str | None) -> np.ndarray:
    """畫面裡鉛垂線的兩個點，"x1,y1,x2,y2"，上面的點在前。回傳指向上方的單位向量。

    不給就是影像的正上方 (0, -1)，等於假設手機沒有翻滾。
    """
    if text is None or not text.strip():
        return UP.copy()
    try:
        x1, y1, x2, y2 = (float(v) for v in text.split(","))
    except ValueError:
        raise ValueError(f"--plumb 要寫成 x1,y1,x2,y2（上面的點在前），收到 {text!r}")
    up = np.array([x1 - x2, y1 - y2])
    length = float(np.linalg.norm(up))
    if length < 20.0:
        raise ValueError("--plumb 的兩個點太近，畫不出鉛垂線的方向")
    return up / length


def _horizontal(up: np.ndarray) -> np.ndarray:
    """與「上」垂直的水平軸。上是 (0,-1) 時是 (1,0)，也就是影像右方。"""
    return np.array([-up[1], up[0]])


def facing_sign(nose_xy, ear_xy, up: np.ndarray = UP) -> float | None:
    """受試者在影像裡面向哪一邊：+1 是面向水平軸的正方向（預設是影像右方），-1 相反。

    看鼻子相對耳朵的位置。兩點沒有都偵測到、或水平差太小時回傳 None。
    """
    if nose_xy is None or ear_xy is None:
        return None
    horizontal = float(np.dot(np.asarray(nose_xy, float) - np.asarray(ear_xy, float),
                              _horizontal(up)))
    if abs(horizontal) < _MIN_FACING_PX:
        return None
    return 1.0 if horizontal > 0 else -1.0


def profile_angle_deg(ear_xy, shoulder_xy, facing: float,
                      up: np.ndarray = UP) -> float | None:
    """耳肩向量與「上」的帶號夾角（度），頭往受試者前方伸為正、往後仰為負。

    與 3D θ_CA 同號：耳在肩正上方是 0°。facing 是 +1 或 -1（見 facing_sign）。
    任何一點缺值、或向量太短都回傳 None，不回傳 0。
    """
    if ear_xy is None or shoulder_xy is None:
        return None
    if facing not in (1.0, -1.0):
        raise ValueError(f"facing 要是 +1 或 -1，收到 {facing!r}")
    vector = np.asarray(ear_xy, float) - np.asarray(shoulder_xy, float)
    if not np.all(np.isfinite(vector)) or float(np.linalg.norm(vector)) < _MIN_VECTOR_PX:
        return None
    forward = facing * _horizontal(up)
    return math.degrees(math.atan2(float(np.dot(vector, forward)),
                                   float(np.dot(vector, up))))


def pick_person(persons: list[PersonKeypoints], side: str = "right",
                min_confidence: float = 0.0) -> PersonKeypoints | None:
    """從一幀偵測到的人裡挑出近側耳與肩都有的那一位，信心度總和最高的。

    正側面的畫面通常只有受試者一個人，背景有人走動時才需要挑。
    """
    if side not in SIDES:
        raise ValueError(f"side 要是 {SIDES}，收到 {side!r}")
    from pose.topology import keypoint_index

    ear, shoulder = f"{side}_ear", f"{side}_shoulder"
    best, best_score = None, -1.0
    for person in persons:
        if person.get(ear) is None or person.get(shoulder) is None:
            continue
        confidence = (float(person.confidences[keypoint_index(ear)])
                      + float(person.confidences[keypoint_index(shoulder)]))
        if min(float(person.confidences[keypoint_index(ear)]),
               float(person.confidences[keypoint_index(shoulder)])) < min_confidence:
            continue
        if confidence > best_score:
            best, best_score = person, confidence
    return best


@dataclass
class FrameSample:
    """一個取樣幀要留下的資料。座標是影像像素，缺值是 None。"""
    t_s: float
    ear: tuple[float, float] | None = None
    shoulder: tuple[float, float] | None = None
    nose: tuple[float, float] | None = None
    reason: str = ""


def sample_from_person(t_s: float, person: PersonKeypoints | None,
                       side: str = "right") -> FrameSample:
    """把 pick_person 挑出的人轉成 FrameSample。沒有人時留下原因，不丟掉那一幀。"""
    if person is None:
        return FrameSample(t_s=t_s, reason=f"沒有偵測到近側（{side}）耳與肩都有的人")

    def point(name: str):
        p = person.get(name)
        return None if p is None else (float(p[0]), float(p[1]))

    return FrameSample(t_s=t_s, ear=point(f"{side}_ear"),
                       shoulder=point(f"{side}_shoulder"), nose=point("nose"))


@dataclass
class WindowSummary:
    """一個時間窗（對應雙目的一段）的參考值。"""
    name: str
    start_s: float
    end_s: float
    frames: int
    usable: int
    facing: float | None
    facing_agreement: float | None
    mean_deg: float | None
    sd_deg: float | None
    se_deg: float | None
    notes: list[str] = field(default_factory=list)


def window_facing(samples: list[FrameSample], up: np.ndarray = UP
                  ) -> tuple[float | None, float | None]:
    """整個時間窗的面向：每幀判一次，取多數。回傳 (面向, 同意的比例)。"""
    votes = [facing_sign(s.nose, s.ear, up) for s in samples]
    votes = [v for v in votes if v is not None]
    if not votes:
        return None, None
    positive = sum(1 for v in votes if v > 0)
    sign = 1.0 if positive * 2 >= len(votes) else -1.0
    agree = (positive if sign > 0 else len(votes) - positive) / len(votes)
    return sign, agree


def summarize_window(name: str, samples: list[FrameSample], start_s: float,
                     end_s: float, up: np.ndarray = UP,
                     facing_override: float | None = None) -> WindowSummary:
    """這個時間窗內的 θ_CA 參考值：平均、單幀散佈、批次平均法的平均值誤差。

    面向在整個時間窗只判一次再套用到每一幀，每幀各判一次的話，偶爾鼻子偵測抖動
    就會讓個別幀的正負號翻轉。
    """
    inside = [s for s in samples if start_s <= s.t_s < end_s]
    notes: list[str] = []
    if facing_override is not None:
        facing, agree = float(facing_override), None
    else:
        facing, agree = window_facing(inside, up)
    summary = WindowSummary(name=name, start_s=start_s, end_s=end_s, frames=len(inside),
                            usable=0, facing=facing, facing_agreement=agree,
                            mean_deg=None, sd_deg=None, se_deg=None, notes=notes)
    if not inside:
        notes.append("時間窗內沒有任何取樣幀，檢查起訖秒數與影片長度")
        return summary
    if facing is None:
        notes.append("判斷不出受試者面向哪一邊（鼻子沒偵測到或與耳的水平差太小），"
                     "可用 --facing right|left 指定")
        return summary
    if agree is not None and agree < _MIN_FACING_AGREEMENT:
        notes.append(f"面向只有 {agree:.0%} 的幀一致，受試者可能轉頭或偵測不穩，"
                     "這一段的參考值要保守看")
    angles = [profile_angle_deg(s.ear, s.shoulder, facing, up) for s in inside]
    values = np.array([a for a in angles if a is not None], dtype=float)
    summary.usable = int(values.size)
    if values.size < _MIN_USABLE_FRAMES:
        notes.append(f"可用幀只有 {values.size}，不報平均")
        return summary
    summary.mean_deg = float(values.mean())
    summary.sd_deg = float(values.std(ddof=1))
    summary.se_deg = standard_error(values)
    if values.size < 0.5 * len(inside):
        notes.append(f"可用幀只佔 {values.size}/{len(inside)}，受試者可能被擋或偵測不穩")
    return summary


def parse_windows(text: str | None, duration_s: float | None = None
                  ) -> list[tuple[str, float, float]]:
    """"名稱=起:訖,名稱=起:訖"，單位秒。不給就是整段影片一個時間窗 all。"""
    if text is None or not text.strip():
        if duration_s is None:
            raise ValueError("沒有 --windows 時需要知道影片長度")
        return [("all", 0.0, float(duration_s))]
    windows, seen = [], set()
    for part in text.split(","):
        name, sep, span = part.strip().partition("=")
        name = name.strip()
        start_text, sep2, end_text = span.partition(":")
        if not name or not sep or not sep2:
            raise ValueError(f"--windows 的格式是 名稱=起:訖（秒），收到 {part!r}")
        try:
            start, end = float(start_text), float(end_text)
        except ValueError:
            raise ValueError(f"--windows 的起訖要是數字，收到 {part!r}")
        if not start < end:
            raise ValueError(f"--windows 的 {name} 起點要早於訖點：{start}～{end}")
        if name in seen:
            raise ValueError(f"--windows 的名稱重複：{name}")
        seen.add(name)
        windows.append((name, start, end))
    return windows


@dataclass
class StereoSegment:
    """雙目那一段的 3D θ_CA 平均（只用 usable 幀）。"""
    path: str
    frames: int
    mean_deg: float | None
    se_deg: float | None


def stereo_segment(path: Path) -> StereoSegment:
    """讀雙目的逐幀 CSV，取 usable 幀的 3D θ_CA 原始值平均與批次平均誤差。"""
    rows, _ = read_rows(Path(path))
    values = []
    for row in rows:
        if row.get("usable") != "1":
            continue
        text = (row.get("theta_ca_deg") or "").strip()
        if text:
            try:
                values.append(float(text))
            except ValueError:
                pass
    if len(values) < _MIN_USABLE_FRAMES:
        return StereoSegment(str(path), len(values), None, None)
    arr = np.array(values, dtype=float)
    return StereoSegment(str(path), len(values), float(arr.mean()), standard_error(arr))


@dataclass
class Comparison:
    name: str
    reference_deg: float | None
    stereo_deg: float | None
    difference_deg: float | None      # 3D 減參考
    error_deg: float | None           # 兩邊各自的批次平均誤差合成
    note: str = ""


def compare(summary: WindowSummary, stereo: StereoSegment) -> Comparison:
    """3D 減正側面參考。兩邊任何一邊沒有平均就不比。"""
    if summary.mean_deg is None:
        return Comparison(summary.name, None, stereo.mean_deg, None, None,
                          "參考值沒有平均（" + "；".join(summary.notes) + "）")
    if stereo.mean_deg is None:
        return Comparison(summary.name, summary.mean_deg, None, None, None,
                          f"雙目可用幀只有 {stereo.frames}，不報平均")
    parts = [e for e in (summary.se_deg, stereo.se_deg) if e is not None]
    error = float(math.sqrt(sum(e * e for e in parts))) if len(parts) == 2 else None
    return Comparison(summary.name, summary.mean_deg, stereo.mean_deg,
                      stereo.mean_deg - summary.mean_deg, error)


def report_lines(summaries: list[WindowSummary],
                 comparisons: list[Comparison] | None = None) -> list[str]:
    """印給人看的文字：每個時間窗的參考值，有雙目 CSV 的再加 3D 減參考。"""

    def number(value: float | None, digits: int = 2, signed: bool = True) -> str:
        if value is None:
            return "—"
        return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"

    lines = ["正側面參考的 θ_CA（前傾為正，單位度）", ""]
    lines.append(f"{'時間窗':12s} {'秒':>11s} {'幀':>5s} {'可用':>5s} {'面向':>5s} "
                 f"{'參考平均':>10s} {'單幀SD':>8s} {'平均誤差':>9s}")
    for s in summaries:
        facing = "—" if s.facing is None else ("+" if s.facing > 0 else "−")
        lines.append(f"{s.name:12s} {s.start_s:5.1f}～{s.end_s:<5.1f} {s.frames:5d} {s.usable:5d} "
                     f"{facing:>5s} {number(s.mean_deg):>10s} {number(s.sd_deg, 2, False):>8s} "
                     f"{number(s.se_deg, 2, False):>9s}")
        for note in s.notes:
            lines.append(f"    注意：{note}")
    if comparisons:
        lines += ["", "雙目 3D θ_CA 減正側面參考（誤差是兩邊各自的批次平均誤差合成）", ""]
        lines.append(f"{'時間窗':12s} {'參考':>9s} {'3D':>9s} {'3D−參考':>10s} {'誤差':>8s}")
        for c in comparisons:
            lines.append(f"{c.name:12s} {number(c.reference_deg):>9s} {number(c.stereo_deg):>9s} "
                         f"{number(c.difference_deg):>10s} {number(c.error_deg, 2, False):>8s}"
                         + (f"   {c.note}" if c.note else ""))
        lines += ["", "參考值與 3D 共用同一個關鍵點偵測模型，偵測對耳朵位置的一致偏差兩邊會一起偏。",
                  "手機擺位偏離正側 90° 時，耳肩的左右間距會漏進影像，參考值會偏。"]
    return lines
