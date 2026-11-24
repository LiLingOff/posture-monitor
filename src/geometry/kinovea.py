"""Kinovea 人工標註：匯出標註包、匯入標註、對 3D θ_sym 算相關。

說明書表4 與 3.6 寫「與 Kinovea 相關係數」。做法：從側傾段挑幾張快照給標註者，
標註者在 Kinovea 裡量兩肩連線與水平線的夾角，填進一張表；這裡把填好的角度對回
同一幀的 3D θ_sym。

## 為什麼沿用盲評包

標註者不能看到系統的角度，不然會被帶著走。快照檔名帶著系統的 θ_CA 與判定，疊加版畫著
系統認的骨架，所以和盲評一樣：只收 `raw/` 原圖、檔名改成亂數、圖片只留像素，key 放在
包外面。匯出直接用 `rating_pack.write_pack`，這裡只決定挑哪些圖、README 寫什麼。

## 對回哪一幀

快照檔名開頭的幀號就是逐幀 CSV 的 `frame` 欄。key 記了 `segment_csv` 與 `frame`，
匯入時照它回到那一列，取**原始**的 `theta_sym_deg`：Kinovea 量的是影像上的絕對角度，
不能拿扣過基準的值去比。

## 這個比較能說明什麼

Kinovea 量的是斜角影像上的 2D 角度，會受透視影響（相機不是正對受試者），所以
「3D 減 Kinovea」不是 3D 的誤差，3D 的真值仍然是量角器。這裡的相關係數說的是：
3D θ_sym 與人工在影像上量到的肩線傾斜，隨姿勢變動的方向與幅度一不一致。

這個模組不碰相機、終端機與 argparse。
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

import numpy as np

from .rating_import import find_rating_files, read_key
from .rating_pack import PackResult, find_items, write_pack
from .rating_stats import Interval, bootstrap
from .session_analysis import correlation, read_rows
from .study import is_tilt

COLUMN = "kinovea_shoulder_deg"
LIMIT_DEG = 90.0
# 受試者少於這個數時不給相關係數的信賴區間：以受試者為單位重抽，人太少沒有意義。
MIN_SUBJECTS_FOR_INTERVAL = 5
DEFAULT_PER_SEGMENT = 3

README = """\
Kinovea 標註說明

做什麼：每一張照片，在 Kinovea 裡量「兩側肩膀連線」與「水平線」的夾角，把角度填進
rating_sheet.csv 的 kinovea_shoulder_deg 欄。

怎麼量
1. 照片依檔名開頭的編號順序處理，一張量完再下一張。
2. 水平線以畫面的橫軸為準（照片的上下邊是水平的），不用修正透視、不用管地板或牆面。
3. 肩膀的點取兩側肩膀關節的中心，大約是衣服上肩線與手臂交界的位置。
4. 角度取到小數點一位，單位是度。

正負號（很重要）
- 受試者面對相機，所以他的右肩在照片的左邊、左肩在照片的右邊。
- 受試者的右肩比左肩高：填正數。受試者的左肩比右肩高：填負數。
  白話：兩肩連線從照片左邊往右邊是往下走（左高右低）就填正數。
- 兩邊一樣高就填 0。

其他
- 每一格都要填。真的看不清楚肩膀在哪裡的照片，欄位留空，並在 note 寫原因。
- 不要改 rater_id 與 blind_id 兩欄，也不要改照片的檔名。
- 量的時候不用管這張照片是誰、做什麼姿勢，也不需要知道系統量到幾度。
"""


class KinoveaImportError(ValueError):
    """標註表有問題。problems 是全部的問題。"""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("標註表有問題，整批沒有匯入：\n  " + "\n  ".join(problems))


def tilt_only(meta: dict[str, str]) -> bool:
    return is_tilt(meta.get("condition", ""))


def export_pack(paths: list[Path], out_dir: Path, annotators: list[str],
                per_segment: int = DEFAULT_PER_SEGMENT, seed: int | None = None,
                key_path: Path | None = None,
                allow_overlay: bool = False) -> tuple[PackResult, list[str], list[str]]:
    """只收側傾段的原圖，每段平均挑 per_segment 張。回傳 (結果, 略過的段落, 用了疊加版的段落)。"""
    items, skipped, overlay = find_items(paths, per_segment, allow_overlay,
                                         include=tilt_only, kendall=False)
    result = write_pack(items, out_dir, annotators,
                        sheet_columns=["rater_id", "blind_id", COLUMN],
                        seed=seed, key_path=key_path, readme=README)
    return result, skipped, overlay


# ---- 匯入 --------------------------------------------------------------

@dataclass(frozen=True)
class Annotation:
    annotator: str
    blind_id: str
    angle: float


@dataclass
class AnnotationSet:
    annotations: list[Annotation]
    annotators: list[str]
    files: list[Path]
    # 標註者 → 還沒填的 blind_id
    missing: dict[str, list[str]] = field(default_factory=dict)


def _data_lines(path: Path) -> list[str]:
    return [line for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if not line.startswith("#")]


def read_annotations(paths: list[Path], key: dict) -> AnnotationSet:
    """讀標註表。問題一次全列出：退回去補的時候不用一次只講一個。"""
    files = find_rating_files(paths)
    if not files:
        raise KinoveaImportError(["沒有找到標註表（標頭要有 rater_id 與 blind_id 兩欄）"])
    problems: list[str] = []
    annotations: list[Annotation] = []
    seen: dict[tuple[str, str], str] = {}
    missing: dict[str, list[str]] = {}
    for path in files:
        rows = list(csv.DictReader(_data_lines(path)))
        columns = set(rows[0]) if rows else set()
        absent = [c for c in ("rater_id", "blind_id", COLUMN) if rows and c not in columns]
        if absent:
            problems.append(f"{path.name} 缺欄：{'、'.join(absent)}")
            continue
        for line_no, row in enumerate(rows, start=2):
            where = f"{path.name} 第 {line_no} 列"
            rater = (row.get("rater_id") or "").strip()
            blind = (row.get("blind_id") or "").strip()
            raw = (row.get(COLUMN) or "").strip()
            if not rater or not blind:
                problems.append(f"{where}：rater_id 或 blind_id 是空的")
                continue
            if blind not in key:
                problems.append(f"{where}：blind_id {blind} 不在 key 裡")
                continue
            if not raw:
                missing.setdefault(rater, []).append(blind)
                continue
            try:
                angle = float(raw)
            except ValueError:
                problems.append(f"{where}：{COLUMN} 要是數字，收到 {raw!r}")
                continue
            if not math.isfinite(angle) or abs(angle) > LIMIT_DEG:
                problems.append(f"{where}：{COLUMN} 是 {raw}，要在 ±{LIMIT_DEG:g}° 以內")
                continue
            if (rater, blind) in seen:
                problems.append(f"{where}：{rater} 把 {blind} 填了兩次（另一次在 {seen[(rater, blind)]}）")
                continue
            seen[(rater, blind)] = where
            annotations.append(Annotation(rater, blind, angle))
    if problems:
        raise KinoveaImportError(problems)
    annotators = sorted({a.annotator for a in annotations} | set(missing))
    return AnnotationSet(annotations, annotators, files, missing)


@dataclass(frozen=True)
class Image:
    """一張標過的圖，連同對回的那一幀。reason 空白代表納入統計。"""

    blind_id: str
    subject: str
    segment: str
    condition: str
    target: float | None
    frame: int
    kinovea: float                 # 多位標註者時是平均
    annotators: int
    system: float | None = None    # 同一幀的原始 θ_sym
    protractor: float | None = None
    reason: str = ""

    @property
    def label(self) -> str:
        return (f"{self.condition} {self.target:g}°" if self.target is not None
                else self.condition)


def _number(text) -> float | None:
    if text is None or str(text).strip() == "":
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _signed_protractor(meta: dict[str, str], condition: str) -> float | None:
    """量角器的帶號實測值：往左傾 θ_sym 為正，往右傾為負（與 trips.TiltRow 同號）。
    有開始與結束兩個值時取平均。"""
    values = [v for v in (_number(meta.get("measured_angle_deg")),
                          _number(meta.get("measured_end_angle_deg"))) if v is not None]
    if not values:
        return None
    mean = sum(values) / len(values)
    return mean if condition == "tilt-left" else -mean


def match_images(annotated: AnnotationSet, key: dict) -> list[Image]:
    """每張標過的圖回到 CSV 的那一幀，取原始 θ_sym。多位標註者標同一張時取平均。"""
    by_blind: dict[str, list[float]] = {}
    for item in annotated.annotations:
        by_blind.setdefault(item.blind_id, []).append(item.angle)
    cache: dict[str, tuple[dict[int, dict], dict[str, str]] | str] = {}
    images = []
    for blind, angles in sorted(by_blind.items()):
        row_key = key[blind]
        path = row_key.segment_csv
        if path not in cache:
            try:
                rows, meta = read_rows(Path(path))
                cache[path] = ({int(float(r["frame"])): r for r in rows if r.get("frame")},
                               meta)
            except (OSError, ValueError, KeyError) as exc:
                cache[path] = f"讀不到逐幀 CSV（{exc}）"
        loaded = cache[path]
        target = None
        base = dict(blind_id=blind, subject=row_key.subject, segment=row_key.segment,
                    condition=row_key.condition, frame=int(row_key.frame or 0),
                    kinovea=sum(angles) / len(angles), annotators=len(angles))
        if isinstance(loaded, str):
            images.append(Image(target=None, reason=loaded, **base))
            continue
        frames, meta = loaded
        target = _number(meta.get("target_angle_deg"))
        protractor = _signed_protractor(meta, row_key.condition)
        row = frames.get(base["frame"])
        reason, system = "", None
        if row is None:
            reason = "CSV 裡沒有這一幀"
        elif str(row.get("usable", "")).strip().lower() not in ("1", "true"):
            reason = "系統那一幀被略過（沒有有效角度）"
        else:
            system = _number(row.get("theta_sym_deg"))
            if system is None:
                reason = "系統那一幀沒有 θ_sym"
        images.append(Image(target=target, system=system, protractor=protractor,
                            reason=reason, **base))
    return images


# ---- 統計 --------------------------------------------------------------

@dataclass(frozen=True)
class CellStat:
    label: str
    images: int
    subjects: int
    kinovea: float
    system: float
    protractor: float | None

    @property
    def difference(self) -> float:
        """3D − Kinovea。"""
        return self.system - self.kinovea


@dataclass
class KinoveaReport:
    annotators: list[str]
    images: list[Image]
    missing: dict[str, list[str]]
    r: float | None = None
    interval: Interval | None = None
    bias: float | None = None
    sd: float | None = None
    cells: list[CellStat] = field(default_factory=list)

    @property
    def used(self) -> list[Image]:
        return [i for i in self.images if not i.reason]

    @property
    def subjects(self) -> int:
        return len({i.subject for i in self.used})

    @property
    def limits(self) -> tuple[float, float] | None:
        if self.bias is None or self.sd is None:
            return None
        return self.bias - 1.96 * self.sd, self.bias + 1.96 * self.sd


def _cell_order(label: str):
    name, _, angle = label.partition(" ")
    try:
        value = float(angle.rstrip("°")) if angle else 0.0
    except ValueError:
        value = 0.0
    return (0 if name == "tilt-left" else 1, value)


def analyse(annotated: AnnotationSet, key: dict, reps: int | None = None) -> KinoveaReport:
    images = match_images(annotated, key)
    report = KinoveaReport(annotated.annotators, images, annotated.missing)
    used = report.used
    if used:
        k = np.array([i.kinovea for i in used])
        s = np.array([i.system for i in used])
        report.r = correlation(k, s)
        diff = s - k
        report.bias = float(diff.mean())
        report.sd = float(diff.std(ddof=1)) if diff.size >= 2 else None
        if report.r is not None and report.subjects >= MIN_SUBJECTS_FOR_INTERVAL:
            pairs = list(zip(k.tolist(), s.tolist()))

            def statistic(sample):
                return correlation(np.array([p[0] for p in sample]),
                                   np.array([p[1] for p in sample]))

            kwargs = {} if reps is None else {"reps": reps}
            report.interval = bootstrap(pairs, [i.subject for i in used], statistic,
                                        **kwargs)
    groups: dict[str, list[Image]] = {}
    for image in used:
        groups.setdefault(image.label, []).append(image)
    for label in sorted(groups, key=_cell_order):
        members = groups[label]
        protractors = [m.protractor for m in members if m.protractor is not None]
        report.cells.append(CellStat(
            label, len(members), len({m.subject for m in members}),
            float(np.mean([m.kinovea for m in members])),
            float(np.mean([m.system for m in members])),
            float(np.mean(protractors)) if protractors else None))
    return report


# ---- 報告 --------------------------------------------------------------

def _f(value, digits=1, signed=True) -> str:
    if value is None:
        return "—"
    return f"{value:+.{digits}f}°" if signed else f"{value:.{digits}f}°"


def report_text(report: KinoveaReport) -> str:
    used = report.used
    skipped = [i for i in report.images if i.reason]
    out = [f"標註者：{'、'.join(report.annotators)}（{len(report.annotators)} 位）"
           + ("；多位標註者標同一張時取平均" if len(report.annotators) > 1 else ""),
           f"標了 {len(report.images)} 張，納入 {len(used)} 張（{report.subjects} 位受試者），"
           f"沒納入 {len(skipped)} 張"]
    reasons: dict[str, int] = {}
    for image in skipped:
        reasons[image.reason] = reasons.get(image.reason, 0) + 1
    out += [f"  沒納入的原因：{reason}，{count} 張" for reason, count in sorted(reasons.items())]
    for annotator, blinds in sorted(report.missing.items()):
        out.append(f"  {annotator} 還有 {len(blinds)} 張沒填")
    if not used:
        out += ["", "沒有可以對的圖，不算相關係數。"]
        return "\n".join(out)
    out.append("")
    if report.r is None:
        out.append("相關係數算不出來：納入的圖不到 3 張，或其中一邊完全沒有變化。")
    else:
        out.append(f"Pearson r（Kinovea 對 3D θ_sym）= {report.r:.3f}，n = {len(used)} 張")
        if report.interval is not None and report.interval.low is not None:
            out.append(f"  95% 區間（以{report.interval.unit}為單位重抽）："
                       f"{report.interval.low:.3f} ～ {report.interval.high:.3f}")
        else:
            out.append(f"  受試者不到 {MIN_SUBJECTS_FOR_INTERVAL} 位，不給信賴區間。")
    limits = report.limits
    out.append(f"3D − Kinovea：平均差 {_f(report.bias)}，差的標準差 {_f(report.sd, signed=False)}"
               + ("" if limits is None else
                  f"，一致界限 {_f(limits[0])} ～ {_f(limits[1])}"))
    out += ["", "依側傾方向與目標角度", ""]
    header = f"{'側傾':<16}{'量角器':>9}{'Kinovea':>10}{'3D θ_sym':>10}{'3D − K':>9}{'張':>5}{'人':>4}"
    out += [header, "-" * len(header)]
    for cell in report.cells:
        out.append(f"{cell.label:<16}{_f(cell.protractor):>9}{_f(cell.kinovea):>10}"
                   f"{_f(cell.system):>10}{_f(cell.difference):>9}{cell.images:>5}{cell.subjects:>4}")
    out += ["",
            "  Kinovea 量的是斜角影像上的 2D 角度，會受透視影響，所以「3D − K」不是 3D 的誤差；",
            "  3D 的真值是量角器。相關係數說的是：3D θ_sym 與人工在影像上量到的肩線傾斜，",
            "  隨姿勢變動的方向與幅度一不一致。",
            "  量角器欄是帶號的實測值（往左傾為正、往右傾為負，開始與結束的平均）。"]
    return "\n".join(out)


def images_csv(report: KinoveaReport) -> str:
    fields = ["blind_id", "subject", "segment", "condition", "target_angle_deg", "frame",
              "kinovea_deg", "annotators", "system_theta_sym_deg", "protractor_deg",
              "included", "reason"]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()

    def r(value, digits=3):
        return "" if value is None else round(value, digits)

    for i in report.images:
        writer.writerow({
            "blind_id": i.blind_id, "subject": i.subject, "segment": i.segment,
            "condition": i.condition,
            "target_angle_deg": "" if i.target is None else f"{i.target:g}",
            "frame": i.frame,
            "kinovea_deg": r(i.kinovea), "annotators": i.annotators,
            "system_theta_sym_deg": r(i.system), "protractor_deg": r(i.protractor),
            "included": int(not i.reason), "reason": i.reason,
        })
    return buffer.getvalue()


def load_key(path: Path) -> dict:
    """key 檔就是盲評包那一份，直接沿用它的讀法與檢查。"""
    return read_key(path)


# ---- 給指令列用的整包流程 ----------------------------------------------

def export_summary(paths: list[Path], out_dir: Path, annotators: list[str],
                   per_segment: int = DEFAULT_PER_SEGMENT, seed: int | None = None,
                   key_path: Path | None = None, allow_overlay: bool = False) -> str:
    """匯出標註包並回傳摘要。出錯時丟 RatingPackError。"""
    result, skipped, overlay = export_pack(paths, out_dir, annotators, per_segment, seed,
                                           key_path, allow_overlay)
    lines = [f"Kinovea 標註包寫到 {out_dir}",
             f"  {result.items} 張側傾段的原圖（每段 {per_segment} 張）",
             f"  給標註者：{out_dir}/<代號>/（照片在 images/，填 rating_sheet.csv 的 {COLUMN}）",
             f"  key 檔：{result.key_path}（在包外面），不要給標註者",
             "  標註的方法在每個資料夾的 README.txt"]
    if overlay:
        lines.append("  用了疊加骨架的快照（--allow-overlay）：" + "、".join(overlay))
    for note in skipped:
        lines.append(f"  略過 {note}")
    return "\n".join(lines)


def import_report(key_path: Path, sheets: list[Path], out_dir: Path | None = None,
                  reps: int | None = None) -> str:
    """讀回標註、對回逐幀 CSV、回傳報告。給了 out_dir 就寫報告與逐張 CSV。

    問題都以 KinoveaImportError 或 ValueError／FileNotFoundError 丟出，呼叫端轉成訊息。
    """
    key = load_key(key_path)
    if not sheets:
        return "沒有標註檔。標註者填好 rating_sheet.csv 之後用 --sheets 指給我。"
    annotated = read_annotations(sheets, key)
    report = analyse(annotated, key, reps)
    text = report_text(report)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "kinovea.txt").write_text(text + "\n", encoding="utf-8")
        (out_dir / "kinovea_images.csv").write_text(images_csv(report), encoding="utf-8")
        text += f"\n\n已寫到 {out_dir / 'kinovea.txt'} 與 {out_dir / 'kinovea_images.csv'}"
    return text
