"""外部評分者盲評的設定檔：量表、評分項目、Kendall 檢核項目與各個門檻。

量表範圍、評分項目、Kendall 檢核項目到 2026-11-18 都還沒定案，所以這裡
**不寫死任何一套量表**。程式只認設定檔裡寫的東西；沒有設定檔就不做評分分析。
有預設值的只有任務指定的兩個門檻（評分下降 2 分、絕對標準 θ_CA ±10°）
與幾個比例，這些也都可以在設定檔裡改。

設定檔格式（JSON）：

    {
      "score": {
        "items": [{"name": "posture", "min": 1, "max": 7, "higher_is_better": true}],
        "primary": "posture"
      },
      "kendall": {
        "items": [{"name": "head_forward", "bad_values": ["1"], "good_values": ["0"]}],
        "min_failed_items": 1
      },
      "agreement": {"min_bad_share": 0.5},
      "rating_drop_threshold": 2,
      "absolute_thresholds_deg": {"theta_ca_deg": 10},
      "alarm_share": 0.5,
      "reference_condition": "upright"
    }

評分表的欄位名稱由項目名稱決定：主觀評分是 `score_<name>`，Kendall 是
`kendall_<name>`。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# 任務指定的預設值。改的話改設定檔，不要改這裡。
DEFAULT_RATING_DROP = 2.0
# 絕對標準的預設門檻是 θ_CA ±10°。這是本專案的設計值，不是 Hansraj 或其他
# 文獻的數字；說明書要引文獻門檻的話，從設定檔改。
DEFAULT_ABSOLUTE_THRESHOLDS = {"theta_ca_deg": 10.0}
# 一段之中有多少比例的幀判超標，這一段就算「系統警報」。
DEFAULT_ALARM_SHARE = 0.5
# 一段之中有多少比例的（評分者 × 照片）Kendall 判不良，這一段就算 Kendall 判不良。
DEFAULT_KENDALL_BAD_SHARE = 0.5
DEFAULT_REFERENCE_CONDITION = "upright"

# 絕對標準只能用逐幀 CSV 裡的原始角度欄。扣過基準的欄位是相對基準線那一套。
_ABSOLUTE_COLUMNS = ("theta_ca_deg", "theta_sym_deg", "theta_ka_deg")
_NAME = re.compile(r"^[A-Za-z0-9_\-]+$")


class RatingConfigError(ValueError):
    """設定檔有問題。訊息列出全部的問題，不是只報第一個。"""


@dataclass(frozen=True)
class ScoreItem:
    name: str
    minimum: float
    maximum: float
    higher_is_better: bool = True

    @property
    def column(self) -> str:
        return f"score_{self.name}"


@dataclass(frozen=True)
class KendallItem:
    """一個 Kendall 檢核項目與它的二分規則。

    值的比對不分大小寫、去掉前後空白。bad 與 good 以外的值一律當成填錯，
    不猜：「?」或「不確定」默默算成通過的話，Kendall 不良率就被低估了。
    """

    name: str
    bad_values: tuple[str, ...]
    good_values: tuple[str, ...]

    @property
    def column(self) -> str:
        return f"kendall_{self.name}"

    def is_bad(self, text: str) -> bool | None:
        """不良回傳 True、通過回傳 False、不認得回傳 None。"""
        value = text.strip().casefold()
        if value in self.bad_values:
            return True
        if value in self.good_values:
            return False
        return None


@dataclass(frozen=True)
class RatingConfig:
    score_items: tuple[ScoreItem, ...]
    primary: str
    kendall_items: tuple[KendallItem, ...]
    min_failed_items: int = 1
    kendall_bad_share: float = DEFAULT_KENDALL_BAD_SHARE
    rating_drop_threshold: float = DEFAULT_RATING_DROP
    absolute_thresholds_deg: dict = field(
        default_factory=lambda: dict(DEFAULT_ABSOLUTE_THRESHOLDS))
    alarm_share: float = DEFAULT_ALARM_SHARE
    reference_condition: str = DEFAULT_REFERENCE_CONDITION

    @property
    def primary_item(self) -> ScoreItem:
        return next(i for i in self.score_items if i.name == self.primary)

    @property
    def columns(self) -> list[str]:
        """評分表要有的欄位，依序。"""
        return (["rater_id", "blind_id"]
                + [i.column for i in self.score_items]
                + [i.column for i in self.kendall_items])

    def kendall_photo_bad(self, flags: list[bool]) -> bool:
        """一位評分者對一張照片的 Kendall 總判定：不良的項目數達到門檻就是不良。"""
        return sum(flags) >= self.min_failed_items


def _values(raw, where: str, problems: list[str]) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        problems.append(f"{where} 要是非空的清單")
        return ()
    return tuple(str(v).strip().casefold() for v in raw)


def parse_config(data: dict) -> RatingConfig:
    """把 JSON 讀出來的字典轉成設定。所有問題一次列出。"""
    problems: list[str] = []
    if not isinstance(data, dict):
        raise RatingConfigError("設定檔最外層要是一個物件")

    score = data.get("score") or {}
    score_items = []
    for n, raw in enumerate(score.get("items") or []):
        where = f"score.items[{n}]"
        name = str(raw.get("name", "")).strip()
        if not _NAME.match(name):
            problems.append(f"{where}.name 只能用英數、底線、連字號，收到 {name!r}")
            continue
        try:
            low, high = float(raw["min"]), float(raw["max"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"{where} 要有數字的 min 與 max")
            continue
        if not low < high:
            problems.append(f"{where} 的 min 要小於 max")
            continue
        score_items.append(ScoreItem(name, low, high,
                                     bool(raw.get("higher_is_better", True))))
    if not score_items:
        problems.append("score.items 至少要有一個評分項目")
    primary = str(score.get("primary", score_items[0].name if score_items else ""))
    if score_items and primary not in {i.name for i in score_items}:
        problems.append(f"score.primary {primary!r} 不在 score.items 裡")

    kendall = data.get("kendall") or {}
    kendall_items = []
    for n, raw in enumerate(kendall.get("items") or []):
        where = f"kendall.items[{n}]"
        name = str(raw.get("name", "")).strip()
        if not _NAME.match(name):
            problems.append(f"{where}.name 只能用英數、底線、連字號，收到 {name!r}")
            continue
        bad = _values(raw.get("bad_values"), f"{where}.bad_values", problems)
        good = _values(raw.get("good_values"), f"{where}.good_values", problems)
        if set(bad) & set(good):
            problems.append(f"{where} 的 bad_values 與 good_values 有重疊")
        kendall_items.append(KendallItem(name, bad, good))
    if not kendall_items:
        problems.append("kendall.items 至少要有一個檢核項目")
    min_failed = kendall.get("min_failed_items", 1)
    if not isinstance(min_failed, int) or not 1 <= min_failed <= max(1, len(kendall_items)):
        problems.append("kendall.min_failed_items 要是 1 到檢核項目數之間的整數")

    names = [i.column for i in score_items] + [i.column for i in kendall_items]
    if len(names) != len(set(names)):
        problems.append("評分項目或 Kendall 項目的名稱重複")

    def _share(value, where):
        try:
            value = float(value)
        except (TypeError, ValueError):
            problems.append(f"{where} 要是數字")
            return 0.5
        if not 0 < value <= 1:
            problems.append(f"{where} 要在 0 到 1 之間（不含 0）")
        return value

    agreement = data.get("agreement") or {}
    bad_share = _share(agreement.get("min_bad_share", DEFAULT_KENDALL_BAD_SHARE),
                       "agreement.min_bad_share")
    alarm_share = _share(data.get("alarm_share", DEFAULT_ALARM_SHARE), "alarm_share")

    try:
        drop = float(data.get("rating_drop_threshold", DEFAULT_RATING_DROP))
        if drop <= 0:
            problems.append("rating_drop_threshold 要大於 0")
    except (TypeError, ValueError):
        problems.append("rating_drop_threshold 要是數字")
        drop = DEFAULT_RATING_DROP

    thresholds = data.get("absolute_thresholds_deg", DEFAULT_ABSOLUTE_THRESHOLDS)
    absolute = {}
    if not isinstance(thresholds, dict) or not thresholds:
        problems.append("absolute_thresholds_deg 要是 {欄位: 度數} 的非空物件")
    else:
        for column, value in thresholds.items():
            if column not in _ABSOLUTE_COLUMNS:
                problems.append(f"absolute_thresholds_deg 只收原始角度欄 "
                                f"{', '.join(_ABSOLUTE_COLUMNS)}，收到 {column!r}")
                continue
            try:
                absolute[column] = float(value)
            except (TypeError, ValueError):
                problems.append(f"absolute_thresholds_deg.{column} 要是數字")
                continue
            if absolute[column] <= 0:
                problems.append(f"absolute_thresholds_deg.{column} 要大於 0")

    reference = str(data.get("reference_condition", DEFAULT_REFERENCE_CONDITION)).strip()
    if not reference:
        problems.append("reference_condition 不能是空的")

    if problems:
        raise RatingConfigError("設定檔有問題：\n  " + "\n  ".join(problems))
    return RatingConfig(
        score_items=tuple(score_items), primary=primary,
        kendall_items=tuple(kendall_items), min_failed_items=min_failed,
        kendall_bad_share=bad_share, rating_drop_threshold=drop,
        absolute_thresholds_deg=absolute, alarm_share=alarm_share,
        reference_condition=reference,
    )


def load_config(path: Path) -> RatingConfig:
    path = Path(path)
    if not path.is_file():
        raise RatingConfigError(f"找不到設定檔 {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RatingConfigError(f"{path} 不是合法的 JSON：{exc}") from exc
    return parse_config(data)
