"""外部評分者盲評：盲評包、設定檔、評分匯入、評分者信度、有效警報。

全部用合成資料與合成設定。量表、評分項目、Kendall 檢核項目都還沒定案，
這裡的設定只是測試用，不代表任何定案的量表。
"""
from __future__ import annotations

import csv
import json
import random
import re
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from geometry.rating_alarm import (absolute_call, compare_methods, relative_call,
                                   segment_results, summarise)
from geometry.rating_config import RatingConfigError, load_config, parse_config
from geometry.rating_import import (RatingImportError, find_rating_files, read_key,
                                    read_ratings)
from geometry.rating_pack import (RatingPackError, count_adjacent, find_items,
                                  parse_snapshot_name, spaced_order,
                                  strip_png, write_pack)
from geometry.rating_report import NO_RATINGS
from geometry.rating_stats import (TwoByTwo, cohen_kappa, fleiss_kappa, icc2,
                                   is_wide, kappa, mcnemar_exact)
from geometry.rating_workflow import analyse_ratings, export_pack
from geometry.recording import Snapshots, snapshot_name
from posture_loader import load_posture_module

cv2 = pytest.importorskip("cv2")

CONFIG = {
    "score": {"items": [{"name": "posture", "min": 1, "max": 7},
                        {"name": "comfort", "min": 0, "max": 10}],
              "primary": "posture"},
    "kendall": {"items": [{"name": "head", "bad_values": ["1"], "good_values": ["0"]},
                          {"name": "shoulder", "bad_values": ["1"], "good_values": ["0"]}],
                "min_failed_items": 1},
}


def _key(out: Path) -> Path:
    """key 檔的預設位置：盲評包旁邊的 <資料夾名>-key/key.csv。"""
    return out.parent / f"{out.name}-key" / "key.csv"


def _config_file(tmp_path, **overrides) -> Path:
    data = json.loads(json.dumps(CONFIG))
    data.update(overrides)
    path = tmp_path / "rating_config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---- 合成圖片 ---------------------------------------------------------------

def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))


def _png_with_text(seed: int, text: str) -> bytes:
    """一張小 PNG，IEND 前面塞一個會洩漏資訊的 tEXt 區塊。"""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, (24, 32, 3), dtype=np.uint8)
    data = cv2.imencode(".png", image)[1].tobytes()
    iend = data.rindex(b"IEND") - 4
    leak = _png_chunk(b"tEXt", b"Comment\x00" + text.encode())
    return data[:iend] + leak + data[iend:]


def _png_idat(data: bytes) -> bytes:
    out, pos = [], 8
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        if data[pos + 4:pos + 8] == b"IDAT":
            out.append(data[pos + 8:pos + 8 + length])
        pos += 12 + length
    return b"".join(out)


def _png_kinds(data: bytes) -> list[bytes]:
    kinds, pos = [], 8
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        kinds.append(data[pos + 4:pos + 8])
        pos += 12 + length
    return kinds


def _jpeg_with_exif(text: str) -> bytes:
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    data = cv2.imencode(".jpg", image)[1].tobytes()
    payload = b"Exif\x00\x00" + text.encode()
    app1 = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    comment = text.encode()
    com = b"\xff\xfe" + struct.pack(">H", len(comment) + 2) + comment
    return data[:2] + app1 + com + data[2:]


# ---- 合成 session -----------------------------------------------------------

def _segment(root, subject, condition, trial, *, posture="正常", ca=2.0, shots=3,
             raw=True, overlay=True, kendall=0, trip_id="t1", frames=20):
    """一段：逐幀 CSV、快照（疊加版與原圖）、可選的 Kendall 照片。"""
    folder = root / subject
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{condition}-{trial}"
    lines = [f"# subject={subject}", f"# condition={condition}", f"# trial={trial}"]
    if trip_id:
        lines.append(f"# trip_id={trip_id}")
    lines.append("frame,elapsed_s,usable,theta_ca_deg,posture")
    for i in range(frames):
        lines.append(f"{i + 1},{i * 0.2:.1f},1,{ca + (0.1 if i % 2 else -0.1)},{posture}")
    (folder / f"{stem}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    verdict = {"超標": "超標", "正常": "正常"}[posture]
    for n in range(shots):
        name = snapshot_name(n * 50 + 1, n * 10.0, ca, verdict)
        leak = f"{subject} {condition} {verdict} {ca}"
        if overlay:
            path = folder / f"{stem}-shots" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_png_with_text(zlib.crc32(f"{subject}{stem}{n}o".encode()), leak))
        if raw:
            path = folder / f"{stem}-shots" / "raw" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_png_with_text(zlib.crc32(f"{subject}{stem}{n}".encode()), leak))
    for n in range(kendall):
        path = folder / f"{stem}-kendall" / f"{subject}_{condition}_side{n}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_jpeg_with_exif(f"{subject} GPS 25.0N"))
    return folder / f"{stem}.csv"


# ---- 快照存原圖 -------------------------------------------------------------

def test_snapshots_keep_no_raw_copy_by_default(tmp_path):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    Snapshots(tmp_path, 10.0).maybe_save(1, 0.0, frame, None, 1.0, "正常")
    assert not (tmp_path / Snapshots.RAW_DIR).exists()


def test_the_raw_copy_has_no_overlay_and_the_same_name(tmp_path):
    """疊加版畫著骨架；原圖要與相機畫面逐像素相同。"""
    from pose.keypoints import PersonKeypoints

    frame = np.full((64, 64, 3), 40, dtype=np.uint8)
    points = np.full((18, 2), np.nan)
    points[0] = (10, 10)
    points[1] = (30, 30)
    kp = PersonKeypoints(points=points, confidences=np.ones(18))
    name = Snapshots(tmp_path, 10.0, keep_raw=True).maybe_save(1, 0.0, frame, kp, 1.0, "超標")
    raw = cv2.imread(str(tmp_path / Snapshots.RAW_DIR / name))
    drawn = cv2.imread(str(tmp_path / name))
    assert np.array_equal(raw, frame)
    assert not np.array_equal(drawn, frame)


def test_live_and_study_accept_the_raw_option_and_default_to_off():
    parser = load_posture_module().build_parser()
    assert parser.parse_args(["live"]).snapshot_raw is False
    assert parser.parse_args(["study", "--snapshot-raw"]).snapshot_raw is True


# ---- 設定檔 -----------------------------------------------------------------

def test_the_config_has_no_built_in_scale():
    """沒有評分項目就不收，程式不替使用者決定量表。"""
    with pytest.raises(RatingConfigError) as err:
        parse_config({})
    assert "score.items" in str(err.value) and "kendall.items" in str(err.value)


def test_the_defaults_are_the_two_agreed_thresholds(tmp_path):
    config = load_config(_config_file(tmp_path))
    assert config.rating_drop_threshold == 2
    assert config.absolute_thresholds_deg == {"theta_ca_deg": 10.0}
    assert config.columns == ["rater_id", "blind_id", "score_posture", "score_comfort",
                              "kendall_head", "kendall_shoulder"]


def test_config_problems_are_listed_together():
    bad = {"score": {"items": [{"name": "p", "min": 5, "max": 1}]},
           "kendall": {"items": [{"name": "h", "bad_values": ["1"], "good_values": ["1"]}]},
           "absolute_thresholds_deg": {"theta_ca_corrected_deg": 10}}
    with pytest.raises(RatingConfigError) as err:
        parse_config(bad)
    text = str(err.value)
    assert "min 要小於 max" in text and "重疊" in text and "原始角度" in text


# ---- 盲評包 -----------------------------------------------------------------

_NAME = re.compile(r"^\d{4}_[a-z2-9]{8}\.png$")


def _two_subjects(root):
    for subject in ("S01", "S02", "S03"):
        _segment(root, subject, "upright", 1, posture="正常", ca=1.0)
        _segment(root, subject, "head-forward", 1, posture="超標", ca=18.5, kendall=1)


def test_pack_file_names_say_nothing_about_the_verdict_or_angle(tmp_path):
    root = tmp_path / "sessions"
    _two_subjects(root)
    out = tmp_path / "pack"
    export_pack([root], out, ["R1", "R2"], _config_file(tmp_path), seed=1)
    files = [p for p in out.rglob("*") if p.is_file() and p.name != ".gitignore"]
    images = [p for p in files if p.parent.name == "images"]
    assert images and all(_NAME.match(p.name) for p in images)
    for path in files:
        rel = str(path.relative_to(out))
        # blind_id 的字母表沒有大寫、沒有 i、沒有標點，這些片段不可能碰巧出現。
        for leak in ("S0", "upright", "head-", "_over", "_ok", "超標", "正常", "18.5", "ca+"):
            assert leak not in rel


def test_pack_images_carry_only_pixels(tmp_path):
    """PNG 的 tEXt 與 JPEG 的 EXIF、註解都要拿掉，像素原封不動。"""
    root = tmp_path / "sessions"
    _two_subjects(root)
    out = tmp_path / "pack"
    export_pack([root], out, ["R1"], seed=2)
    key = read_key(_key(out))
    for image in (out / "R1" / "images").iterdir():
        data = image.read_bytes()
        # 用整段文字比對，短的片段可能碰巧出現在壓縮後的像素資料裡。
        for leak in (b"Comment", b"S01 head-forward", b"S02 upright", "超標 18.5".encode(),
                     b"GPS 25.0N"):
            assert leak not in data
        row = key[image.stem.split("_", 1)[1]]
        assert image.suffix == ".png"
        assert set(_png_kinds(data)) <= {b"IHDR", b"IDAT", b"IEND"}
        if row.kind == "snapshot":
            assert _png_idat(data) == _png_idat(Path(row.source).read_bytes())
            assert Path(row.source).parent.name == "raw", "只收原圖"
        else:
            assert b"Exif" not in data
            assert cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is not None


def test_blind_ids_match_the_key_and_the_sheets(tmp_path):
    root = tmp_path / "sessions"
    _two_subjects(root)
    out = tmp_path / "pack"
    export_pack([root], out, ["R1", "R2"], _config_file(tmp_path), seed=3)
    key = read_key(_key(out))
    assert len(key) == 3 * (3 + 3 + 1)
    for rater in ("R1", "R2"):
        images = {p.stem.split("_", 1)[1] for p in (out / rater / "images").iterdir()}
        with (out / rater / "rating_sheet.csv").open(encoding="utf-8") as f:
            sheet = list(csv.DictReader(f))
        assert images == set(key) == {r["blind_id"] for r in sheet}
        assert {r["rater_id"] for r in sheet} == {rater}
        assert "score_posture" in sheet[0] and "kendall_head" in sheet[0]
    row = next(r for r in key.values() if r.condition == "head-forward" and r.kind == "snapshot")
    assert row.system_posture == "超標" and float(row.theta_ca_deg) == 18.5
    assert row.trip_id == "t1" and row.subject.startswith("S0")
    assert not list(out.rglob("key*")), "key 檔不在盲評包裡"


def test_each_rater_gets_an_own_order_without_same_subject_neighbours(tmp_path):
    root = tmp_path / "sessions"
    _two_subjects(root)
    out = tmp_path / "pack"
    export_pack([root], out, ["R1", "R2"], seed=4)
    key = read_key(_key(out))
    orders = {}
    for rater in ("R1", "R2"):
        names = sorted(p.name for p in (out / rater / "images").iterdir())
        ids = [n.split("_", 1)[1].split(".")[0] for n in names]
        subjects = [key[i].subject for i in ids]
        assert all(a != b for a, b in zip(subjects, subjects[1:]))
        orders[rater] = ids
    assert orders["R1"] != orders["R2"]


def test_spaced_order_never_puts_neighbours_together_when_it_is_possible():
    rng = random.Random(0)
    for _ in range(300):
        sizes = [rng.randint(1, 6) for _ in range(rng.randint(2, 5))]
        groups = [f"g{i}" for i, n in enumerate(sizes) for _ in range(n)]
        if max(sizes) > (len(groups) + 1) // 2:
            continue
        order = spaced_order(groups, random.Random(rng.random()))
        assert sorted(order) == list(range(len(groups)))
        assert count_adjacent(order, groups) == 0


def test_an_impossible_spacing_still_exports_and_counts_the_neighbours():
    groups = ["a"] * 5 + ["b"]
    order = spaced_order(groups, random.Random(1))
    assert sorted(order) == list(range(6))
    assert count_adjacent(order, groups) == 3


def test_overlay_only_snapshots_are_refused_unless_allowed(tmp_path):
    root = tmp_path / "sessions"
    _segment(root, "S01", "upright", 1, raw=False)
    with pytest.raises(RatingPackError) as err:
        find_items([root])
    assert "S01/upright-1.csv" in str(err.value) and "--allow-overlay" in str(err.value)
    items, _, overlay = find_items([root], allow_overlay=True)
    assert len(items) == 3 and overlay == ["S01/upright-1.csv"]


def test_a_non_empty_output_folder_is_refused(tmp_path):
    root = tmp_path / "sessions"
    _two_subjects(root)
    out = tmp_path / "pack"
    out.mkdir()
    (out / "old.txt").write_text("x")
    items, _, _ = find_items([root])
    with pytest.raises(RatingPackError):
        write_pack(items, out, ["R1"])


def test_per_segment_picks_evenly_spread_snapshots(tmp_path):
    root = tmp_path / "sessions"
    _segment(root, "S01", "upright", 1, shots=9)
    items, _, _ = find_items([root], per_segment=3)
    assert [int(i.frame) for i in items] == [51, 201, 351]


def test_snapshot_names_parse_back():
    assert parse_snapshot_name(snapshot_name(51, 10.0, 8.2, "正常")) == {
        "frame": "51", "elapsed_s": "010.0", "theta_ca_deg": "8.2", "system_posture": "正常"}
    assert parse_snapshot_name(snapshot_name(1, 0.0, None, "超標"))["theta_ca_deg"] == ""
    assert parse_snapshot_name("notes.png") is None


def test_strippers_reject_other_formats():
    with pytest.raises(RatingPackError):
        strip_png(b"GIF89a")


# ---- 統計：已知答案 ---------------------------------------------------------

def test_cohen_kappa_known_values():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1
    # Wikipedia 的例子：20 同意是、15 同意否、5 與 10 不一致 → 0.4
    a = [1] * 25 + [0] * 25
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)
    assert cohen_kappa([1, 1, 1], [1, 1, 1]) is None, "只用一個類別時沒有定義"


def test_random_ratings_give_kappa_near_zero():
    rng = random.Random(5)
    a = [rng.random() < 0.5 for _ in range(5000)]
    b = [rng.random() < 0.5 for _ in range(5000)]
    assert abs(cohen_kappa(a, b)) < 0.05
    rows = [[rng.random() < 0.5 for _ in range(4)] for _ in range(3000)]
    assert abs(fleiss_kappa(rows)) < 0.05


def test_fleiss_kappa_matches_the_published_example():
    """Fleiss (1971) 常被引用的例子：10 個對象、14 位評分者、5 類，κ ≈ 0.210。"""
    counts = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0],
              [2, 2, 8, 1, 1], [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2],
              [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
    table = [[c for c, n in enumerate(row) for _ in range(n)] for row in counts]
    assert fleiss_kappa(table) == pytest.approx(0.210, abs=0.001)
    assert fleiss_kappa([[1, 1, 1], [0, 0, 0]]) == 1


def test_kappa_picks_cohen_for_two_and_fleiss_for_more():
    two = [[1, 0], [0, 0], [1, 1], [0, 1], [1, 1]]
    assert kappa(two) == cohen_kappa([r[0] for r in two], [r[1] for r in two])
    three = [[1, 0, 1], [0, 0, 0], [1, 1, 1]]
    assert kappa(three) == fleiss_kappa(three)


def test_icc_matches_shrout_and_fleiss():
    """Shrout & Fleiss (1979) 表2：6 個對象、4 位評分者，ICC(2,1)=0.29、ICC(2,4)=0.62。"""
    data = [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]]
    result = icc2(data)
    assert result.single == pytest.approx(0.29, abs=0.005)
    assert result.average == pytest.approx(0.62, abs=0.005)


def test_icc_perfect_and_random():
    perfect = [[v, v, v] for v in (1, 3, 4, 6, 7)]
    assert icc2(perfect).single == pytest.approx(1.0)
    rng = np.random.default_rng(6)
    noise = rng.integers(1, 8, (3000, 2))
    assert abs(icc2(noise).single) < 0.05
    assert icc2([[3, 3], [3, 3]]).single is None


def test_two_by_two_tests_known_values():
    chi, p, _ = TwoByTwo(10, 20, 20, 10).chi_square()
    assert chi == pytest.approx(6.667, abs=0.001) and p == pytest.approx(0.00982, abs=1e-4)
    assert TwoByTwo(1, 9, 11, 3).fisher_p() == pytest.approx(0.002759, abs=1e-5)
    assert mcnemar_exact(1, 9) == pytest.approx(22 / 1024)
    assert TwoByTwo(0, 0, 3, 4).chi_square() == (None, None, None)


def test_few_subjects_are_always_flagged_as_wide():
    from geometry.rating_stats import Interval
    assert is_wide(Interval(0.5, 0.6, "受試者", 2000), subjects=3)
    assert not is_wide(Interval(0.5, 0.6, "受試者", 2000), subjects=12)
    assert is_wide(Interval(None, None, "受試者", 0), subjects=12)


# ---- 評分匯入 ---------------------------------------------------------------

def _write_sheet(path, rows, columns=None):
    columns = columns or ["rater_id", "blind_id", "score_posture", "score_comfort",
                          "kendall_head", "kendall_shoulder", "note"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(rater, blind, posture=5, comfort=5, head="0", shoulder="0"):
    return dict(rater_id=rater, blind_id=blind, score_posture=posture,
                score_comfort=comfort, kendall_head=head, kendall_shoulder=shoulder)


@pytest.fixture
def pack(tmp_path):
    root = tmp_path / "sessions"
    _two_subjects(root)
    out = tmp_path / "pack"
    export_pack([root], out, ["R1", "R2"], _config_file(tmp_path), seed=7)
    return out, read_key(_key(out)), load_config(tmp_path / "rating_config.json")


def test_import_reports_every_problem_at_once(tmp_path, pack):
    _, key, config = pack
    ids = sorted(key)
    sheet = _write_sheet(tmp_path / "r1.csv", [
        _row("R1", ids[0]),
        _row("R1", ids[0]),                        # 重複
        _row("R1", "zzzzzzzz"),                    # 不在 key
        _row("R1", ids[1], posture=9),             # 超出量表
        _row("R1", ids[2], head="maybe"),          # 不在二分規則
        dict(_row("R1", ids[3]), score_comfort=""),  # 只填一部分
    ])
    with pytest.raises(RatingImportError) as err:
        read_ratings([sheet], config, key)
    text = str(err.value)
    for expected in ("重複評分", "不在 key 檔", "超出量表", "maybe", "只填了一部分"):
        assert expected in text
    assert len(err.value.problems) == 5


def test_a_missing_column_is_named(tmp_path, pack):
    _, key, config = pack
    sheet = _write_sheet(tmp_path / "r1.csv", [_row("R1", sorted(key)[0])],
                         columns=["rater_id", "blind_id", "score_posture"])
    with pytest.raises(RatingImportError) as err:
        read_ratings([sheet], config, key)
    assert "kendall_head" in str(err.value) and "score_comfort" in str(err.value)


def test_blank_rows_count_as_not_yet_rated(pack):
    out, key, config = pack
    sheet = out / "R1" / "rating_sheet.csv"
    rows = list(csv.DictReader(sheet.open(encoding="utf-8")))
    for row in rows[:5]:
        row.update(score_posture=4, score_comfort=4, kendall_head="0", kendall_shoulder="1")
    _write_sheet(sheet, rows, columns=list(rows[0]))
    ratings = read_ratings([sheet], config, key)
    assert len(ratings.ratings) == 5
    assert len(ratings.missing["R1"]) == len(key) - 5


def test_the_key_file_is_not_mistaken_for_a_rating_sheet(pack):
    out, _, _ = pack
    files = find_rating_files([out])
    assert all(p.name != "key.csv" for p in files)


# ---- 有效警報 ---------------------------------------------------------------

def _scenario(tmp_path):
    """一位受試者一趟，五段，答案事先知道：

    - upright：評分 6、Kendall 通過，系統正常 → 對照
    - head-forward：評分 3、Kendall 不良，相對超標、原始 θ_CA 20° → 兩種都有效
    - head-back：評分 6、Kendall 通過，相對超標、原始 3° → 相對誤報、絕對正確未警報
    - left-shoulder-up：評分 3、Kendall 不良，相對正常、原始 2° → 兩種都漏報
    - forward：評分 3、Kendall 通過，相對超標、原始 15° → 兩種都誤報（Kendall 不一致）
    """
    root = tmp_path / "sessions"
    plan = {"upright": ("正常", 1.0, 6, "0"), "head-forward": ("超標", 20.0, 3, "1"),
            "head-back": ("超標", 3.0, 6, "0"), "left-shoulder-up": ("正常", 2.0, 3, "1"),
            "forward": ("超標", 15.0, 3, "0")}
    for condition, (posture, ca, _, _) in plan.items():
        _segment(root, "S01", condition, 1, posture=posture, ca=ca)
    out = tmp_path / "pack"
    config_path = _config_file(tmp_path)
    export_pack([root], out, ["R1", "R2"], config_path, seed=8)
    key = read_key(_key(out))
    for rater, offset in (("R1", 0), ("R2", 0.5)):
        rows = [_row(rater, b, posture=plan[r.condition][2] + offset,
                     head=plan[r.condition][3]) for b, r in key.items()]
        _write_sheet(tmp_path / f"{rater}.csv", rows)
    return root, out, key, load_config(config_path), config_path


def test_effective_false_and_missed_alarms_are_classified(tmp_path):
    _, out, key, config, _ = _scenario(tmp_path)
    ratings = read_ratings([tmp_path / "R1.csv", tmp_path / "R2.csv"], config, key)
    results = {r.condition: r for r in segment_results(ratings, key, config)}
    expect = {"upright": ("正確未警報", "正確未警報"), "head-forward": ("有效", "有效"),
              "head-back": ("誤報", "正確未警報"), "left-shoulder-up": ("漏報", "漏報"),
              "forward": ("誤報", "誤報")}
    for condition, (rel, ab) in expect.items():
        assert results[condition].outcome("relative", config) == rel, condition
        assert results[condition].outcome("absolute", config) == ab, condition
    assert results["head-forward"].drop(config) == pytest.approx(3.0)

    summaries = summarise(list(results.values()), config)
    rel, ab = summaries["relative"], summaries["absolute"]
    assert (rel.alarms, rel.effective, rel.false_alarms, rel.misses) == (3, 1, 2, 1)
    assert rel.effective_rate == pytest.approx(1 / 3) and rel.miss_rate == pytest.approx(0.5)
    assert (ab.alarms, ab.effective, ab.false_alarms, ab.misses) == (2, 1, 1, 1)
    comparison = compare_methods(list(results.values()), config, summaries)
    assert (comparison.table.a, comparison.table.b, comparison.table.c,
            comparison.table.d) == (1, 2, 1, 1)
    assert comparison.only_absolute_correct == 1 and comparison.only_relative_correct == 0


def test_a_lower_is_better_scale_flips_the_drop(tmp_path):
    _, _, key, _, _ = _scenario(tmp_path)
    data = json.loads(json.dumps(CONFIG))
    data["score"]["items"][0]["higher_is_better"] = False
    config = parse_config(data)
    ratings = read_ratings([tmp_path / "R1.csv", tmp_path / "R2.csv"], config, key)
    results = {r.condition: r for r in segment_results(ratings, key, config)}
    assert results["head-forward"].drop(config) == pytest.approx(-3.0)
    assert results["head-forward"].rating_bad(config) is False


def test_the_absolute_threshold_comes_from_the_config():
    rows = [{"usable": "1", "theta_ca_deg": "12"}] * 3
    assert absolute_call(rows, {"theta_ca_deg": 10.0}, 0.5).alarm is True
    assert absolute_call(rows, {"theta_ca_deg": 15.0}, 0.5).alarm is False
    assert absolute_call([{"usable": "0", "theta_ca_deg": "30"}],
                         {"theta_ca_deg": 10.0}, 0.5).alarm is None


def test_a_segment_without_judgements_is_left_unclassified():
    assert relative_call([{"posture": ""}, {"posture": "未知"}], 0.5).alarm is None


def test_a_trip_without_upright_ratings_is_explained(tmp_path):
    root = tmp_path / "sessions"
    _segment(root, "S01", "head-forward", 1, posture="超標", ca=20)
    out = tmp_path / "pack"
    config_path = _config_file(tmp_path)
    export_pack([root], out, ["R1"], config_path, seed=9)
    key = read_key(_key(out))
    _write_sheet(tmp_path / "R1.csv", [_row("R1", b) for b in key])
    config = load_config(config_path)
    ratings = read_ratings([tmp_path / "R1.csv"], config, key)
    [result] = segment_results(ratings, key, config)
    assert result.outcome("relative", config) is None
    assert "同趟沒有 upright" in result.note


def test_the_full_report_states_unit_and_dependence(tmp_path):
    _, out, _, _, config_path = _scenario(tmp_path)
    report = analyse_ratings(_key(out), [tmp_path / "R1.csv", tmp_path / "R2.csv"],
                             config_path, out_dir=tmp_path / "result", reps=200)
    assert "統計單位：段落" in report and "不獨立" in report
    assert "Cohen's kappa" in report and "ICC(2,1)" in report and "ICC(2,k)" in report
    assert "區間很寬" in report, "一位受試者，區間一定要標寬"
    assert (tmp_path / "result" / "rating_segments.csv").is_file()


def test_three_raters_use_fleiss(tmp_path):
    _, out, key, _, config_path = _scenario(tmp_path)
    rows = [_row("R3", b, posture=4) for b in key]
    _write_sheet(tmp_path / "R3.csv", rows)
    report = analyse_ratings(_key(out),
                             [tmp_path / f"R{i}.csv" for i in (1, 2, 3)],
                             config_path, reps=100)
    assert "Fleiss' kappa" in report


# ---- 沒有評分檔、既有指令 ---------------------------------------------------

def test_no_rating_files_is_said_plainly_and_does_not_crash(tmp_path, pack):
    out, _, _ = pack
    empty = tmp_path / "nothing"
    empty.mkdir()
    report = analyse_ratings(_key(out), [empty], tmp_path / "rating_config.json")
    assert report.startswith(NO_RATINGS)


def test_the_ratings_command_without_files_exits_normally(tmp_path, pack, capsys):
    out, _, _ = pack
    module = load_posture_module()
    args = module.build_parser().parse_args(
        ["ratings", "--key", str(_key(out)),
         "--config", str(tmp_path / "rating_config.json")])
    module._run_ratings(args)
    assert NO_RATINGS in capsys.readouterr().out


def test_existing_commands_parse_as_before():
    parser = load_posture_module().build_parser()
    args = parser.parse_args(["analyse", "a.csv"])
    assert args.mode == "analyse" and args.window == 30
    args = parser.parse_args(["cohort", "data/sessions"])
    assert args.baseline_condition == "upright"


# ---- 審查後補的情境 ---------------------------------------------------------

def test_the_key_lives_outside_the_pack_and_both_are_git_ignored(tmp_path):
    """整包交給評分者時不能帶到 key。"""
    root = tmp_path / "sessions"
    _two_subjects(root)
    out = tmp_path / "pack"
    export_pack([root], out, ["R1"], seed=10)
    assert _key(out).is_file()
    assert out not in _key(out).parents
    assert (out / ".gitignore").read_text() == "*\n"
    assert (_key(out).parent / ".gitignore").read_text() == "*\n"


def test_a_key_inside_the_pack_or_over_an_old_key_is_refused(tmp_path):
    root = tmp_path / "sessions"
    _two_subjects(root)
    items, _, _ = find_items([root])
    with pytest.raises(RatingPackError, match="不能放在盲評包"):
        write_pack(items, tmp_path / "pack", ["R1"], key_path=tmp_path / "pack" / "k.csv")
    old = tmp_path / "old-key.csv"
    old.write_text("x")
    with pytest.raises(RatingPackError, match="已經存在"):
        write_pack(items, tmp_path / "pack2", ["R1"], key_path=old)


def test_a_quoted_boolean_does_not_flip_the_scale():
    """bool("false") 是 True，用 bool() 轉的話量表方向會被默默反過來。"""
    data = json.loads(json.dumps(CONFIG))
    data["score"]["items"][0]["higher_is_better"] = "false"
    with pytest.raises(RatingConfigError, match="higher_is_better"):
        parse_config(data)


def test_wrongly_shaped_config_is_reported_not_crashed():
    with pytest.raises(RatingConfigError) as err:
        parse_config({"score": [], "kendall": {"items": ["head"]}, "agreement": 3})
    text = str(err.value)
    assert "score 要是一個物件" in text and "kendall.items[0]" in text and "agreement" in text


def test_a_duplicated_blind_id_in_the_key_is_refused(tmp_path, pack):
    out, _, _ = pack
    lines = _key(out).read_text(encoding="utf-8").splitlines()
    _key(out).write_text("\n".join(lines + [lines[-1]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="重複"):
        read_key(_key(out))


def test_a_partial_raw_folder_is_not_silently_trimmed(tmp_path):
    root = tmp_path / "sessions"
    _segment(root, "S01", "upright", 1, shots=3)
    raw = root / "S01" / "upright-1-shots" / "raw"
    sorted(raw.iterdir())[0].unlink()
    with pytest.raises(RatingPackError, match="1 張沒有原圖"):
        find_items([root])
    items, _, overlay = find_items([root], allow_overlay=True)
    assert len(items) == 3 and overlay == ["S01/upright-1.csv"]
    assert sum(1 for i in items if i.source.parent.name == "raw") == 2


def _exif_orientation(value: int) -> bytes:
    """最小的 EXIF：一個 IFD、一個方向標記（0x0112，SHORT）。"""
    ifd = (b"MM\x00*" + struct.pack(">I", 8) + struct.pack(">H", 1)
           + struct.pack(">HHIHH", 0x0112, 3, 1, value, 0) + struct.pack(">I", 0))
    payload = b"Exif\x00\x00" + ifd
    return b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload


def test_jpeg_photos_are_turned_upright_before_the_exif_is_dropped(tmp_path):
    """手機直拍：像素是橫的，靠 EXIF 方向標記轉正。拿掉 EXIF 前要先轉。"""
    root = tmp_path / "sessions"
    _segment(root, "S01", "upright", 1, shots=0)
    image = np.zeros((8, 16, 3), dtype=np.uint8)
    data = cv2.imencode(".jpg", image)[1].tobytes()
    photo = root / "S01" / "upright-1-kendall" / "side.jpg"
    photo.parent.mkdir(parents=True)
    photo.write_bytes(data[:2] + _exif_orientation(6) + data[2:])
    out = tmp_path / "pack"
    export_pack([root], out, ["R1"], seed=11)
    [exported] = list((out / "R1" / "images").iterdir())
    decoded = cv2.imdecode(np.frombuffer(exported.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (16, 8), "轉正後變成直的"
    assert b"Exif" not in exported.read_bytes()


def test_output_files_in_the_ratings_folder_are_not_read_as_sheets(tmp_path, pack):
    _, key, _ = pack
    folder = tmp_path / "ratings"
    folder.mkdir()
    _write_sheet(folder / "R1.csv", [_row("R1", b) for b in key])
    (folder / "rating_segments.csv").write_text("segment,subject\nx,y\n", encoding="utf-8")
    assert find_rating_files([folder]) == [folder / "R1.csv"]


def test_the_drop_uses_only_raters_who_scored_both(tmp_path):
    """端正坐只有給分偏高的 R1 評、前傾只有給分偏低的 R2 評：沒有共同評分者，不分類。"""
    root = tmp_path / "sessions"
    _segment(root, "S01", "upright", 1, posture="正常", ca=1.0)
    _segment(root, "S01", "head-forward", 1, posture="超標", ca=20.0)
    out = tmp_path / "pack"
    config_path = _config_file(tmp_path)
    export_pack([root], out, ["R1", "R2"], config_path, seed=12)
    key = read_key(_key(out))
    config = load_config(config_path)
    rows = [_row("R1", b, posture=7) for b, r in key.items() if r.condition == "upright"]
    rows += [_row("R2", b, posture=4, head="1") for b, r in key.items()
             if r.condition == "head-forward"]
    _write_sheet(tmp_path / "all.csv", rows)
    ratings = read_ratings([tmp_path / "all.csv"], config, key)
    results = {r.condition: r for r in segment_results(ratings, key, config)}
    forward = results["head-forward"]
    assert forward.drop(config) is None
    assert forward.outcome("relative", config) is None
    assert "沒有評分者同時評過" in forward.note
