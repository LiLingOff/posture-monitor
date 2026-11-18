"""讀回盲評包的 key 檔與評分者填好的評分表，並檢查。

評分表（CSV）至少要有 rater_id、blind_id、設定檔裡每個評分項目的
`score_<name>` 與每個 Kendall 項目的 `kendall_<name>`。多出來的欄位（order、
note）照收不管。

檢查一次列出全部問題，不是遇到第一個就停：評分者交回來的表要退回去補，
一次只說一個問題就要來回好幾趟。

- 錯誤（整批不收）：缺欄、blind_id 不在 key 裡、同一位評分者同一張評了兩次、
  只填了一部分的列、分數超出量表、Kendall 值不在設定的二分規則裡。
- 缺漏（照樣分析，但明確列出）：整列評分欄都是空白（還沒評），或評分者
  根本沒有那一張的列。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from .rating_config import RatingConfig
from .rating_pack import KEY_COLUMNS


class RatingImportError(ValueError):
    """評分表有問題。problems 是全部的問題。"""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("評分表有問題，整批沒有匯入：\n  " + "\n  ".join(problems))


@dataclass(frozen=True)
class KeyRow:
    blind_id: str
    subject: str
    trip_id: str
    condition: str
    trial: str
    segment: str
    kind: str
    frame: str
    elapsed_s: str
    system_posture: str
    theta_ca_deg: str
    segment_csv: str
    source: str


@dataclass(frozen=True)
class Rating:
    rater: str
    blind_id: str
    scores: dict
    kendall: dict           # 項目名稱 → True（不良）／False（通過）


@dataclass
class RatingSet:
    ratings: list[Rating]
    raters: list[str]
    files: list[Path]
    # 評分者 → 還沒評的 blind_id
    missing: dict[str, list[str]] = field(default_factory=dict)


def _data_lines(path: Path) -> list[str]:
    return [line for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if not line.startswith("#")]


def read_key(path: Path) -> dict[str, KeyRow]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到 key 檔 {path}")
    rows = list(csv.DictReader(_data_lines(path)))
    missing = [c for c in KEY_COLUMNS if rows and c not in rows[0]]
    if not rows or missing:
        raise ValueError(f"{path} 不是盲評包的 key 檔（缺欄：{', '.join(missing) or '沒有資料列'}）")
    key = {}
    problems = []
    for line_no, row in enumerate(rows, start=2):
        blind = (row.get("blind_id") or "").strip()
        if not blind:
            problems.append(f"第 {line_no} 筆資料 blind_id 是空的")
        elif blind in key:
            # 後一筆蓋掉前一筆的話，評分會被歸到別人的段落，報告上完全看不出來。
            problems.append(f"第 {line_no} 筆資料 blind_id {blind} 重複")
        else:
            key[blind] = KeyRow(**{c: (row.get(c) or "") for c in KEY_COLUMNS})
    if problems:
        raise ValueError(f"{path} 的 key 檔有問題：\n  " + "\n  ".join(problems))
    return key


def _looks_like_sheet(path: Path) -> bool:
    """標頭同時有 rater_id 與 blind_id 才算評分表。"""
    for line in _data_lines(path)[:1]:
        header = next(csv.reader([line]), [])
        return "rater_id" in header and "blind_id" in header
    return False


def find_rating_files(paths: list[Path]) -> list[Path]:
    """評分檔可以給檔案或資料夾。

    直接指定的檔案照收（格式不對會在匯入時報缺欄）。資料夾只收標頭有
    rater_id 與 blind_id 的 CSV：同一個資料夾裡可能有上一次 `ratings --out`
    寫出的 rating_segments.csv 或 key 檔，不該被當成評分表。
    """
    files = []
    for path in paths:
        path = Path(path)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(p for p in sorted(path.rglob("*.csv")) if _looks_like_sheet(p))
    return files


def read_ratings(files: list[Path], config: RatingConfig,
                 key: dict[str, KeyRow]) -> RatingSet:
    problems: list[str] = []
    ratings: list[Rating] = []
    seen: dict[tuple[str, str], str] = {}
    blank: dict[str, set[str]] = {}
    rating_columns = [i.column for i in config.score_items] + \
                     [i.column for i in config.kendall_items]

    for path in files:
        lines = _data_lines(path)
        reader = csv.DictReader(lines)
        header = reader.fieldnames or []
        lacking = [c for c in config.columns if c not in header]
        if lacking:
            problems.append(f"{path.name}：缺欄 {', '.join(lacking)}")
            continue
        for line_no, row in enumerate(reader, start=2):
            where = f"{path.name} 第 {line_no} 列"
            rater = (row.get("rater_id") or "").strip()
            blind = (row.get("blind_id") or "").strip()
            if not rater or not blind:
                problems.append(f"{where}：rater_id 或 blind_id 是空的")
                continue
            if blind not in key:
                problems.append(f"{where}：blind_id {blind} 不在 key 檔裡")
                continue
            cells = {c: (row.get(c) or "").strip() for c in rating_columns}
            empty = [c for c, v in cells.items() if not v]
            if len(empty) == len(cells):
                blank.setdefault(rater, set()).add(blind)
                continue
            if empty:
                problems.append(f"{where}（{rater}/{blind}）：只填了一部分，空白的是 "
                                f"{', '.join(empty)}")
                continue
            pair = (rater, blind)
            if pair in seen:
                problems.append(f"{where}：{rater} 對 {blind} 重複評分"
                                f"（另一筆在 {seen[pair]}）")
                continue
            seen[pair] = where
            scores, kendall, ok = {}, {}, True
            for item in config.score_items:
                text = cells[item.column]
                try:
                    value = float(text)
                except ValueError:
                    problems.append(f"{where}：{item.column} 不是數字（{text!r}）")
                    ok = False
                    continue
                if not item.minimum <= value <= item.maximum:
                    problems.append(f"{where}：{item.column}={value:g} 超出量表 "
                                    f"{item.minimum:g}～{item.maximum:g}")
                    ok = False
                    continue
                scores[item.name] = value
            for item in config.kendall_items:
                flag = item.is_bad(cells[item.column])
                if flag is None:
                    problems.append(
                        f"{where}：{item.column}={cells[item.column]!r} 不在設定的值裡"
                        f"（不良：{'/'.join(item.bad_values)}；"
                        f"通過：{'/'.join(item.good_values)}）")
                    ok = False
                    continue
                kendall[item.name] = flag
            if ok:
                ratings.append(Rating(rater, blind, scores, kendall))

    if problems:
        raise RatingImportError(problems)

    raters = sorted({r.rater for r in ratings} | set(blank))
    rated = {(r.rater, r.blind_id) for r in ratings}
    missing = {}
    for rater in raters:
        lacking = sorted(b for b in key if (rater, b) not in rated)
        if lacking:
            missing[rater] = lacking
    return RatingSet(ratings=ratings, raters=raters, files=list(files), missing=missing)
