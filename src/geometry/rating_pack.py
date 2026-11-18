"""給外部評分者的盲評包。

評分者是事後盲評，不能知道系統判定。現有快照的檔名就帶著判定與角度
（`000051_t010.0s_ca+008.2_ok.png`），資料夾名稱帶著姿勢條件與受試者代號，
所以不能把快照資料夾直接交出去。這裡做的事：

1. 從 `data/sessions/<subject>/` 找出每一段的快照與 Kendall 檢核照片。
2. 每一張給一個隨機的 blind_id，複製成 `<順序>_<blind_id>.<副檔名>`。
   blind_id 是亂數，不是從任何欄位雜湊出來的，所以猜不回原檔。
3. 圖片本身只留像素：PNG 只保留 IHDR、PLTE、tRNS、IDAT、IEND，文字、時間
   等附屬區塊全部拿掉；JPEG 拿掉 EXIF、XMP、註解等 APP 與 COM 區段（手機拍的
   Kendall 照片會有拍攝時間、機型、GPS）。
4. 快照預設只收 `raw/` 底下不疊加資訊的原圖。疊加版畫著系統偵測到的骨架與
   點名，雖然沒有判定文字，但評分者看得到系統把耳朵、肩膀認在哪裡，會被帶著
   走。沒有原圖的段落預設拒絕匯出，要明知道這一點才用 `allow_overlay`。
5. 另寫一份 key 檔（blind_id 對應受試者、趟次、條件、第幾次、時間、系統判定），
   放在 `key/`，與給評分者的 `raters/` 分開，**key 不給評分者**。
6. 每位評分者的順序各自隨機，並且盡量讓同一位受試者的照片不連續出現。
   做不到時（例如某位受試者的照片超過一半）照樣匯出，但回報還剩幾處相鄰。

Kendall 檢核照片的放法：與該段 CSV 同一層，資料夾叫 `<condition>-<trial>-kendall/`，
裡面放 png 或 jpg，檔名不拘（反正會被重新命名）。
"""
from __future__ import annotations

import csv
import random
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

from .recording import Snapshots

# snapshot_name 的格式：幀號_t秒數s_ca角度_判定.png
_SNAPSHOT = re.compile(
    r"^(?P<frame>\d+)_t(?P<t>[\d.]+)s_ca(?P<ca>na|[+-][\d.]+)_(?P<v>[a-z]+)\.png$")
_VERDICT = {"over": "超標", "ok": "正常", "unknown": "未知", "none": ""}
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
# 不含容易看錯的 0/o、1/l/i。
_ID_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"
_ID_LENGTH = 8

KEY_COLUMNS = ("blind_id", "subject", "trip_id", "condition", "trial", "segment",
               "kind", "frame", "elapsed_s", "system_posture", "theta_ca_deg",
               "segment_csv", "source")


class RatingPackError(ValueError):
    pass


@dataclass(frozen=True)
class PackItem:
    """要給評分者看的一張圖，以及只寫進 key 檔的身分。"""

    source: Path
    subject: str
    trip_id: str
    condition: str
    trial: str
    segment: str            # <受試者資料夾>/<段落 CSV 檔名>
    segment_csv: Path
    kind: str               # snapshot 或 kendall
    frame: str = ""
    elapsed_s: str = ""
    system_posture: str = ""
    theta_ca_deg: str = ""


@dataclass
class PackResult:
    out_dir: Path
    items: int
    raters: list[str]
    adjacent_same_subject: dict[str, int] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    used_overlay: list[str] = field(default_factory=list)
    stripped_jpeg_orientation: int = 0


def read_meta(path: Path) -> dict[str, str]:
    """只讀 CSV 標頭的 `# key=value`，不讀整份資料。"""
    meta = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            key, sep, value = line.lstrip("#").strip().partition("=")
            if sep:
                meta[key.strip()] = value.strip()
    return meta


def parse_snapshot_name(name: str) -> dict | None:
    match = _SNAPSHOT.match(name)
    if match is None:
        return None
    ca = match["ca"]
    return {
        "frame": str(int(match["frame"])),
        "elapsed_s": match["t"],
        "theta_ca_deg": "" if ca == "na" else str(float(ca)),
        "system_posture": _VERDICT.get(match["v"], match["v"]),
    }


def _evenly(items: list, limit: int) -> list:
    """從時間序列裡平均挑 limit 張。limit<=0 表示全部。"""
    if limit <= 0 or len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[int(i * step + step / 2)] for i in range(limit)]


def find_items(paths: list[Path], per_segment: int = 0,
               allow_overlay: bool = False) -> tuple[list[PackItem], list[str], list[str]]:
    """找出所有要匯出的圖。回傳 (圖, 略過的段落說明, 用了疊加版的段落)。

    沒有原圖而且沒允許疊加版時，丟出錯誤並列出全部缺原圖的段落，
    不要只匯出一部分讓人以為收齊了。
    """
    csvs: list[Path] = []
    for path in paths:
        path = Path(path)
        if path.is_file() and path.suffix == ".csv":
            csvs.append(path)
        elif path.is_dir():
            csvs.extend(sorted(path.rglob("*.csv")))
    items: list[PackItem] = []
    skipped, overlay, missing_raw = [], [], []
    for csv_path in sorted(set(csvs)):
        meta = read_meta(csv_path)
        subject = meta.get("subject") or csv_path.parent.name
        segment = f"{csv_path.parent.name}/{csv_path.name}"
        base = dict(subject=subject, trip_id=meta.get("trip_id", ""),
                    condition=meta.get("condition", ""), trial=meta.get("trial", ""),
                    segment=segment, segment_csv=csv_path.resolve())
        shots = csv_path.with_name(f"{csv_path.stem}-shots")
        raw = shots / Snapshots.RAW_DIR
        found = []
        source_dir = None
        if raw.is_dir() and any(raw.glob("*.png")):
            source_dir = raw
        elif shots.is_dir() and any(shots.glob("*.png")):
            if allow_overlay:
                source_dir = shots
                overlay.append(segment)
            else:
                missing_raw.append(segment)
        if source_dir is not None:
            for image in sorted(source_dir.glob("*.png")):
                parsed = parse_snapshot_name(image.name)
                if parsed is not None:
                    found.append((int(parsed["frame"]), image, parsed))
            found.sort(key=lambda f: f[0])
            for _, image, parsed in _evenly(found, per_segment):
                items.append(PackItem(source=image, kind="snapshot", **base, **parsed))
        kendall = csv_path.with_name(f"{csv_path.stem}-kendall")
        photos = []
        if kendall.is_dir():
            photos = sorted(p for p in kendall.iterdir()
                            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)
            for photo in photos:
                items.append(PackItem(source=photo, kind="kendall", **base))
        if not found and not photos and segment not in missing_raw:
            skipped.append(f"{segment}：沒有快照也沒有 Kendall 照片")
    if missing_raw:
        raise RatingPackError(
            "這些段落只有疊加骨架的快照，沒有原圖（量測時沒加 --snapshot-raw）：\n  "
            + "\n  ".join(missing_raw)
            + "\n疊加版沒有判定文字，但評分者看得到系統的骨架，會被帶著走。"
              "要用的話明確加上 --allow-overlay")
    return items, skipped, overlay


def new_blind_ids(count: int, rng: random.Random) -> list[str]:
    seen: set[str] = set()
    out = []
    while len(out) < count:
        candidate = "".join(rng.choice(_ID_ALPHABET) for _ in range(_ID_LENGTH))
        if candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


def spaced_order(groups: list[str], rng: random.Random) -> list[int]:
    """把項目排成隨機順序，盡量讓同一組（受試者）不相鄰。

    groups[i] 是第 i 個項目的組別，回傳項目索引的排列。每一步從「不是上一個
    組別」的組裡挑：若某一組剩下的數量已經佔一半以上，非先排它不可，否則依
    剩下的數量加權隨機挑一組，再從那組隨機挑一個。最大的一組不超過總數的
    一半（向上取整）時，結果一定沒有相鄰；超過時無解，只能盡量少。
    """
    pools: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        pools.setdefault(group, []).append(index)
    for pool in pools.values():
        rng.shuffle(pool)
    order: list[int] = []
    last = None
    while pools:
        total = sum(len(p) for p in pools.values())
        candidates = [g for g in pools if g != last] or [last]
        heavy = [g for g in candidates if 2 * len(pools[g]) >= total]
        if heavy:
            chosen = max(heavy, key=lambda g: len(pools[g]))
        else:
            pick = sorted(candidates)
            chosen = rng.choices(pick, weights=[len(pools[g]) for g in pick])[0]
        order.append(pools[chosen].pop())
        if not pools[chosen]:
            del pools[chosen]
        last = chosen
    return order


def count_adjacent(order: list[int], groups: list[str]) -> int:
    return sum(1 for a, b in zip(order, order[1:]) if groups[a] == groups[b])


# ---- 拿掉圖片的中繼資料 ----------------------------------------------------

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_KEEP = {b"IHDR", b"PLTE", b"tRNS", b"IDAT", b"IEND"}


def strip_png(data: bytes) -> bytes:
    """只留下畫出像素必要的區塊。tEXt、iTXt、zTXt、tIME、eXIf 等全部拿掉。"""
    if not data.startswith(_PNG_SIGNATURE):
        raise RatingPackError("不是 PNG 檔")
    out = [_PNG_SIGNATURE]
    pos = len(_PNG_SIGNATURE)
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        kind = data[pos + 4:pos + 8]
        end = pos + 12 + length
        if end > len(data):
            raise RatingPackError("PNG 檔被截斷")
        if kind in _PNG_KEEP:
            out.append(data[pos:end])
        pos = end
        if kind == b"IEND":
            break
    return b"".join(out)


def strip_jpeg(data: bytes) -> tuple[bytes, bool]:
    """拿掉 APP1～APP13、APP15 與 COM（EXIF、XMP、IPTC、註解）。

    留 APP0（JFIF）與 APP14（Adobe 色彩轉換，拿掉的話部分 CMYK 檔顏色會錯）。
    回傳 (新的位元組, 原檔是否帶 EXIF)：EXIF 的方向標記跟著拿掉，手機直拍的
    照片可能變成橫的，要在匯出前先轉正，文件 10 有寫。
    """
    if not data.startswith(b"\xff\xd8"):
        raise RatingPackError("不是 JPEG 檔")
    out = [b"\xff\xd8"]
    pos = 2
    had_exif = False
    while pos < len(data):
        if data[pos] != 0xFF:
            raise RatingPackError("JPEG 區段格式不對")
        marker = data[pos + 1]
        if marker == 0xFF:          # 填充位元組
            pos += 1
            continue
        if marker == 0xD9:          # EOI
            out.append(data[pos:pos + 2])
            break
        if 0xD0 <= marker <= 0xD7 or marker == 0x01:
            out.append(data[pos:pos + 2])
            pos += 2
            continue
        (length,) = struct.unpack(">H", data[pos + 2:pos + 4])
        end = pos + 2 + length
        if marker == 0xDA:          # SOS 之後是壓縮資料，原樣接到結尾
            out.append(data[pos:])
            break
        drop = (0xE1 <= marker <= 0xEF and marker != 0xEE) or marker == 0xFE
        if marker == 0xE1 and data[pos + 4:pos + 10] == b"Exif\x00\x00":
            had_exif = True
        if not drop:
            out.append(data[pos:end])
        pos = end
    return b"".join(out), had_exif


def clean_image(path: Path) -> tuple[bytes, str, bool]:
    """回傳 (去掉中繼資料的位元組, 副檔名, 原檔是否帶 EXIF)。"""
    data = Path(path).read_bytes()
    if data.startswith(_PNG_SIGNATURE):
        return strip_png(data), ".png", False
    if data.startswith(b"\xff\xd8"):
        cleaned, had_exif = strip_jpeg(data)
        return cleaned, ".jpg", had_exif
    raise RatingPackError(f"{path} 不是 PNG 或 JPEG，盲評包只收這兩種")


# ---- 寫出盲評包 ------------------------------------------------------------

_RATER_ID = re.compile(r"^[A-Za-z0-9_\-]+$")

_RATER_README = """\
評分說明（給評分者）

1. 照片依檔名開頭的編號順序看，一張評完再看下一張，不要回頭改。
2. 每一張照片在 rating_sheet.csv 裡有一列，blind_id 與檔名後半段相同，不要改動
   rater_id 與 blind_id 兩欄。
3. 評分項目與 Kendall 檢核項目的定義、量表範圍，以研究者另外提供的說明為準。
4. 每一格都要填。看不清楚的照片請在 note 欄註明，評分欄仍依判斷填寫。
"""

_KEY_WARNING = (
    "# 這是盲評包的 key 檔，對得回受試者與系統判定。不可給評分者，也不要進 git。"
)


def write_pack(items: list[PackItem], out_dir: Path, raters: list[str],
               sheet_columns: list[str] | None = None,
               seed: int | None = None) -> PackResult:
    """寫出盲評包。out_dir 必須不存在或是空的，免得新舊兩批混在一起。

    sheet_columns 是評分表的欄位（從設定檔來）；沒給就只有 rater_id、blind_id、
    order、note，評分欄等量表定案後再補。
    """
    out_dir = Path(out_dir)
    if not items:
        raise RatingPackError("沒有找到任何快照或 Kendall 照片，盲評包是空的")
    if not raters:
        raise RatingPackError("至少要有一位評分者")
    bad = [r for r in raters if not _RATER_ID.match(r)]
    if bad or len(set(raters)) != len(raters):
        raise RatingPackError(f"評分者代號只能用英數、底線、連字號且不能重複：{raters}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RatingPackError(f"{out_dir} 已經有東西。換一個空的資料夾，新舊兩批混在一起會對錯 key")

    base_rng = random.Random(seed) if seed is not None else random.SystemRandom()
    ids = new_blind_ids(len(items), base_rng)
    # 先打亂再寫 key，key 的列順序就不會透露原本的時間順序。
    key_order = list(range(len(items)))
    base_rng.shuffle(key_order)

    cleaned: dict[int, tuple[bytes, str]] = {}
    orientation_lost = 0
    for index, item in enumerate(items):
        data, suffix, had_exif = clean_image(item.source)
        cleaned[index] = (data, suffix)
        orientation_lost += int(had_exif)

    key_dir = out_dir / "key"
    key_dir.mkdir(parents=True, exist_ok=True)
    with (key_dir / "key.csv").open("w", newline="", encoding="utf-8") as handle:
        handle.write(_KEY_WARNING + "\n")
        writer = csv.writer(handle)
        writer.writerow(KEY_COLUMNS)
        for index in key_order:
            item = items[index]
            writer.writerow([
                ids[index], item.subject, item.trip_id, item.condition, item.trial,
                item.segment, item.kind, item.frame, item.elapsed_s,
                item.system_posture, item.theta_ca_deg, str(item.segment_csv),
                str(item.source),
            ])

    groups = [item.subject for item in items]
    columns = list(sheet_columns or ["rater_id", "blind_id"])
    for extra in ("order", "note"):
        if extra not in columns:
            columns.append(extra)
    adjacent = {}
    for rater in raters:
        rng = (random.Random(f"{seed}:{rater}") if seed is not None
               else random.SystemRandom())
        order = spaced_order(groups, rng)
        adjacent[rater] = count_adjacent(order, groups)
        folder = out_dir / "raters" / rater
        images = folder / "images"
        images.mkdir(parents=True, exist_ok=True)
        width = max(4, len(str(len(order))))
        rows = []
        for position, index in enumerate(order, start=1):
            data, suffix = cleaned[index]
            (images / f"{position:0{width}d}_{ids[index]}{suffix}").write_bytes(data)
            row = {c: "" for c in columns}
            row.update(rater_id=rater, blind_id=ids[index], order=position)
            rows.append(row)
        with (folder / "rating_sheet.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        (folder / "README.txt").write_text(_RATER_README, encoding="utf-8")

    return PackResult(out_dir=out_dir, items=len(items), raters=list(raters),
                      adjacent_same_subject=adjacent,
                      stripped_jpeg_orientation=orientation_lost)
