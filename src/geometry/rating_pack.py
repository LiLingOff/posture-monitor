"""給外部評分者的盲評包。

評分者是事後盲評，不能知道系統判定。現有快照的檔名就帶著判定與角度
（`000051_t010.0s_ca+008.2_ok.png`），資料夾名稱帶著姿勢條件與受試者代號，
所以不能把快照資料夾直接交出去。這裡做的事：

1. 從 `data/sessions/<subject>/` 找出每一段的快照與 Kendall 檢核照片。
2. 每一張給一個隨機的 blind_id，複製成 `<順序>_<blind_id>.png`。
   blind_id 是亂數，不是從任何欄位雜湊出來的，所以猜不回原檔。
3. 圖片本身只留像素：PNG 只保留 IHDR、PLTE、tRNS、IDAT、IEND，文字、時間
   等附屬區塊全部拿掉；JPEG（手機拍的 Kendall 照片會有拍攝時間、機型、GPS）
   依 EXIF 方向轉正後存成同樣只留像素的 PNG。
4. 快照預設只收 `raw/` 底下不疊加資訊的原圖。疊加版畫著系統偵測到的骨架與
   點名，雖然沒有判定文字，但評分者看得到系統把耳朵、肩膀認在哪裡，會被帶著
   走。沒有原圖的段落預設拒絕匯出，要明知道這一點才用 `allow_overlay`。
5. 另寫一份 key 檔（blind_id 對應受試者、趟次、條件、第幾次、時間、系統判定）。
   key 放在盲評包資料夾**之外**（預設是旁邊的 `<資料夾>-key/key.csv`）：放在裡面
   的話，把整包壓縮傳給評分者就連 key 一起送出去了。
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
    key_path: Path | None = None
    adjacent_same_subject: dict[str, int] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    used_overlay: list[str] = field(default_factory=list)
    reencoded_jpeg: int = 0


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


def _snapshot_names(folder: Path) -> set[str]:
    """資料夾裡檔名符合快照格式的 png。資料夾不存在就是空集合。"""
    if not folder.is_dir():
        return set()
    return {p.name for p in folder.glob("*.png") if parse_snapshot_name(p.name)}


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
        # 逐張比對：raw/ 只存了一部分（例如量到一半才加 --snapshot-raw）時，
        # 只看 raw/ 有沒有檔案的話，缺原圖的那幾張會被靜默漏掉。
        raw_names = _snapshot_names(raw)
        overlay_only = _snapshot_names(shots) - raw_names
        if overlay_only and not allow_overlay:
            missing_raw.append(f"{segment}（{len(overlay_only)} 張沒有原圖）")
            overlay_only = set()
        elif overlay_only:
            overlay.append(segment)
        found = []
        for name in raw_names | overlay_only:
            parsed = parse_snapshot_name(name)
            image = (raw if name in raw_names else shots) / name
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
        if not found and not photos and not any(m.startswith(segment) for m in missing_raw):
            skipped.append(f"{segment}：沒有快照也沒有 Kendall 照片")
    if missing_raw:
        raise RatingPackError(
            "這些段落有疊加骨架的快照卻沒有對應的原圖（量測時沒加 --snapshot-raw）：\n  "
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


def jpeg_to_png(data: bytes) -> bytes:
    """把 JPEG 解碼再存成 PNG，只留像素。

    手機直拍的照片常常是橫的像素加上 EXIF 方向標記。只拿掉 EXIF 的話照片會
    轉九十度，評分者看到的是躺著的人；OpenCV 解碼時會照方向標記轉正，所以
    先解碼、轉正，再存成不帶任何中繼資料的 PNG（EXIF、XMP、註解、GPS 都不留）。
    存 PNG 不存 JPEG，免得再壓縮一次又掉畫質。
    """
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RatingPackError("JPEG 檔解不開，可能被截斷")
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RatingPackError("JPEG 轉存 PNG 失敗")
    return strip_png(encoded.tobytes())


def clean_image(path: Path) -> tuple[bytes, bool]:
    """回傳 (只留像素的 PNG 位元組, 原檔是否為 JPEG)。"""
    data = Path(path).read_bytes()
    if data.startswith(_PNG_SIGNATURE):
        return strip_png(data), False
    if data.startswith(bytes((0xFF, 0xD8))):
        return jpeg_to_png(data), True
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


def default_key_path(out_dir: Path) -> Path:
    """key 檔的預設位置：盲評包資料夾旁邊的 `<資料夾名>-key/key.csv`。"""
    out_dir = Path(out_dir)
    return out_dir.parent / f"{out_dir.name}-key" / "key.csv"


def write_pack(items: list[PackItem], out_dir: Path, raters: list[str],
               sheet_columns: list[str] | None = None,
               seed: int | None = None, key_path: Path | None = None) -> PackResult:
    """寫出盲評包。out_dir 必須不存在或是空的，免得新舊兩批混在一起。

    out_dir 底下只有每位評分者一個資料夾，整包交出去也不會帶到 key；
    key_path 不能落在 out_dir 裡面，也不能覆蓋既有的 key。

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
    key_path = Path(key_path) if key_path is not None else default_key_path(out_dir)
    if out_dir.resolve() in key_path.resolve().parents:
        raise RatingPackError(f"key 檔 {key_path} 不能放在盲評包 {out_dir} 裡面，"
                              f"整包交給評分者時會連 key 一起送出去")
    if key_path.exists():
        raise RatingPackError(f"{key_path} 已經存在。換一個位置，蓋掉舊的 key 就對不回舊的評分")

    base_rng = random.Random(seed) if seed is not None else random.SystemRandom()
    ids = new_blind_ids(len(items), base_rng)
    # 先打亂再寫 key，key 的列順序就不會透露原本的時間順序。
    key_order = list(range(len(items)))
    base_rng.shuffle(key_order)

    cleaned: dict[int, tuple[bytes, str]] = {}
    reencoded = 0
    for index, item in enumerate(items):
        data, was_jpeg = clean_image(item.source)
        cleaned[index] = (data, ".png")
        reencoded += int(was_jpeg)

    # 兩個資料夾各放一個忽略全部的 .gitignore：放到 data/ 以外也不會被 git 收進
    # 這個 public repo（盲評包有受試者的臉，key 對得回受試者）。
    key_path.parent.mkdir(parents=True, exist_ok=True)
    (key_path.parent / ".gitignore").write_text("*\n", encoding="utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".gitignore").write_text("*\n", encoding="utf-8")
    with key_path.open("w", newline="", encoding="utf-8") as handle:
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
        folder = out_dir / rater
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

    return PackResult(out_dir=out_dir, items=len(items), raters=list(raters), key_path=key_path,
                      adjacent_same_subject=adjacent,
                      reencoded_jpeg=reencoded)
