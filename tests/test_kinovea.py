"""Kinovea 人工標註：匯出只含側傾的原圖、匯入檢查、對回同一幀的 3D θ_sym 算相關。

CSV 與快照都是合成的，答案可以心算。快照檔名開頭的幀號要等於 CSV 的 frame 欄。
"""
import csv
import shutil
from pathlib import Path

import pytest

from geometry.kinovea import (COLUMN, KinoveaImportError, analyse, export_pack,
                              images_csv, load_key, read_annotations, report_text)
from geometry.rating_pack import RatingPackError
from geometry.recording import snapshot_name
from geometry.session_analysis import read_rows
from test_trips import _write

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

SNAPSHOT_FRAMES = (10, 50, 90, 130, 170, 190)       # 一段 60 秒，每 10 秒一張
BASELINE = (0.0, 3.0, 40.0)    # 扣基準後的 θ_sym 比原始少 3，用來證明匯入取的是原始值


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((12, 16, 3), 90, dtype=np.uint8))


def _segment(root, subject, condition, sym, *, target, measured=None, trial=1,
             shots=True, raw=True, frames=200, extra=None):
    folder = root / subject
    name = f"{condition}-{trial}"
    meta = {"target_angle_deg": target, "order_group": "A", **(extra or {})}
    if measured is not None:
        meta["measured_angle_deg"] = measured
        meta["measured_end_angle_deg"] = measured + 2
    path = _write(folder, f"{name}.csv", subject=subject, condition=condition,
                  trial=trial, trip="T1", sym=sym, frames=frames, baseline=BASELINE,
                  extra=meta)
    if shots:
        for frame in SNAPSHOT_FRAMES:
            shot = snapshot_name(frame, frame * 0.3, 1.0, "正常")
            _png(folder / f"{name}-shots" / shot)
            if raw:
                _png(folder / f"{name}-shots" / "raw" / shot)
    return path


def _upright(root, subject):
    """端正坐：只有疊加版快照、沒有原圖。Kinovea 包不收它，所以不該擋匯出。"""
    return _segment(root, subject, "upright", 0.0, target=None, raw=False,
                    extra={"target_angle_deg": ""})


def _subjects(root, count=2):
    for n in range(count):
        subject = f"s{n + 1}"
        _upright(root, subject)
        _segment(root, subject, "tilt-left", 10.0 + n, target=10, measured=9 + n)
        _segment(root, subject, "tilt-left", 20.0 + n, target=20, measured=19 + n, trial=2)
        _segment(root, subject, "tilt-right", -20.0 - n, target=20, measured=19 + n)
        _segment(root, subject, "tilt-right", -30.0 - n, target=30, measured=29 + n, trial=2)
    return root


def _export(tmp_path, root, annotators=("K1",), **kwargs):
    out = tmp_path / "pack"
    result, skipped, overlay = export_pack([root], out, list(annotators), seed=1,
                                           key_path=tmp_path / "key" / "key.csv", **kwargs)
    return out, result, tmp_path / "key" / "key.csv"


def _sheet(out, annotator="K1"):
    return out / annotator / "rating_sheet.csv"


def _fill(out, key_path, fill, annotator="K1", blank=()):
    """照 key 一張一張填：fill(對回的 CSV 那一幀的原始 θ_sym, key 列) → 角度。"""
    key = load_key(key_path)
    sheet = _sheet(out, annotator)
    rows = list(csv.DictReader(sheet.read_text(encoding="utf-8").splitlines()))
    for row in rows:
        k = key[row["blind_id"]]
        if row["blind_id"] in blank:
            continue
        frames = {int(float(r["frame"])): r for r in read_rows(Path(k.segment_csv))[0]}
        system = float(frames[int(k.frame)]["theta_sym_deg"])
        row[COLUMN] = f"{fill(system, k):.3f}"
    with sheet.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


# ---- 匯出 --------------------------------------------------------------

def test_the_pack_has_only_tilt_segments_three_images_each(tmp_path):
    root = _subjects(tmp_path / "data")
    out, result, key_path = _export(tmp_path, root)
    key = load_key(key_path)
    # 2 位 × 4 段側傾 × 3 張。端正坐有疊加版沒原圖，不收，所以不會擋匯出。
    assert result.items == 24 and len(key) == 24
    assert {k.condition for k in key.values()} == {"tilt-left", "tilt-right"}
    per_segment = {}
    for k in key.values():
        per_segment[k.segment_csv] = per_segment.get(k.segment_csv, 0) + 1
    assert set(per_segment.values()) == {3}


def test_the_sheet_asks_for_one_angle_and_the_readme_explains_the_sign(tmp_path):
    out, _, _ = _export(tmp_path, _subjects(tmp_path / "data"))
    header = _sheet(out).read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == ["rater_id", "blind_id", COLUMN, "order", "note"]
    readme = (out / "K1" / "README.txt").read_text(encoding="utf-8")
    assert "右肩比左肩高" in readme and "填正數" in readme and "填負數" in readme
    assert "Kinovea" in readme


def test_the_annotator_cannot_see_the_system_angle_in_anything_handed_over(tmp_path):
    out, _, key_path = _export(tmp_path, _subjects(tmp_path / "data"))
    names = [p.name for p in (out / "K1" / "images").iterdir()]
    # 原檔名帶著系統的 θ_CA 與判定（_ca+001.0_ok），交出去的檔名只剩順序與亂數。
    assert names and all("_ca" not in n and "_ok" not in n and "s_" not in n for n in names)
    assert not key_path.resolve().is_relative_to(out.resolve())


def test_an_overlay_only_tilt_segment_is_refused_not_silently_dropped(tmp_path):
    root = _subjects(tmp_path / "data", count=1)
    _segment(root, "s1", "tilt-left", 15.0, target=15, trial=3, raw=False)
    with pytest.raises(RatingPackError, match="沒有對應的原圖"):
        _export(tmp_path, root)


def test_a_pack_with_no_tilt_segments_is_an_error(tmp_path):
    root = tmp_path / "data"
    _upright(root, "s1")
    with pytest.raises(RatingPackError, match="盲評包是空的"):
        _export(tmp_path, root)


# ---- 匯入檢查 ----------------------------------------------------------

def test_every_problem_in_the_sheet_is_listed_at_once(tmp_path):
    root = _subjects(tmp_path / "data", count=1)
    out, _, key_path = _export(tmp_path, root)
    key = load_key(key_path)
    sheet = _sheet(out)
    rows = list(csv.DictReader(sheet.read_text(encoding="utf-8").splitlines()))
    rows[0][COLUMN] = "abc"                        # 不是數字
    rows[1][COLUMN] = "120"                        # 超出 ±90
    rows[2][COLUMN] = "5"
    rows[3]["blind_id"] = "zzzzzzzz"               # 不在 key 裡
    rows[3][COLUMN] = "5"
    rows[4]["rater_id"] = ""                       # 空的代號
    rows[4][COLUMN] = "5"
    rows.append(dict(rows[2]))                     # 同一張填兩次
    with sheet.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(KinoveaImportError) as caught:
        read_annotations([out], key)
    text = "\n".join(caught.value.problems)
    for expected in ("要是數字", "要在 ±90° 以內", "不在 key 裡", "rater_id 或 blind_id 是空的",
                     "填了兩次"):
        assert expected in text, expected
    assert len(caught.value.problems) == 5


def test_a_sheet_missing_the_angle_column_is_refused(tmp_path):
    out, _, key_path = _export(tmp_path, _subjects(tmp_path / "data", count=1))
    sheet = _sheet(out)
    lines = sheet.read_text(encoding="utf-8").splitlines()
    sheet.write_text("\n".join(line.replace(f",{COLUMN}", "") if i == 0 else line
                               for i, line in enumerate(lines)) + "\n", encoding="utf-8")
    with pytest.raises(KinoveaImportError, match="缺欄"):
        read_annotations([out], load_key(key_path))


def test_blank_rows_are_missing_not_errors(tmp_path):
    out, _, key_path = _export(tmp_path, _subjects(tmp_path / "data", count=1))
    key = load_key(key_path)
    first = next(iter(csv.DictReader(
        _sheet(out).read_text(encoding="utf-8").splitlines())))["blind_id"]
    _fill(out, key_path, lambda system, k: system + 1.0, blank=(first,))
    annotated = read_annotations([out], key)
    assert annotated.missing == {"K1": [first]}
    report = analyse(annotated, key)
    assert len(report.images) == 11 and "K1 還有 1 張沒填" in report_text(report)


# ---- 對回同一幀、統計 --------------------------------------------------

def test_the_angle_is_compared_with_the_raw_theta_sym_of_the_same_frame(tmp_path):
    root = _subjects(tmp_path / "data")
    out, _, key_path = _export(tmp_path, root)
    key = load_key(key_path)
    _fill(out, key_path, lambda system, k: system + 1.0)
    report = analyse(read_annotations([out], key), key)
    used = report.used
    assert len(used) == 24 and report.subjects == 2
    # 每一張的 3D 值就是那一段的原始 θ_sym（±0.5 的晃動），不是扣掉 3 的修正值。
    for image in used:
        assert abs(image.system - image.kinovea + 1.0) < 1e-9
    assert report.r == pytest.approx(1.0)
    assert report.bias == pytest.approx(-1.0)
    assert report.sd == pytest.approx(0.0, abs=1e-9)
    assert report.limits == pytest.approx((-1.0, -1.0))


def test_the_correlation_is_not_one_when_the_annotations_are_noisy(tmp_path):
    root = _subjects(tmp_path / "data")
    out, _, key_path = _export(tmp_path, root)
    key = load_key(key_path)
    noise = iter([4.0, -3.0, 2.0, -5.0, 6.0, -2.0] * 4)
    _fill(out, key_path, lambda system, k: system + next(noise))
    report = analyse(read_annotations([out], key), key)
    assert 0.0 < report.r < 1.0 and report.sd > 1.0


def test_cells_follow_side_and_target_with_the_signed_protractor(tmp_path):
    root = _subjects(tmp_path / "data")
    out, _, key_path = _export(tmp_path, root)
    key = load_key(key_path)
    _fill(out, key_path, lambda system, k: system + 1.0)
    cells = {c.label: c for c in analyse(read_annotations([out], key), key).cells}
    assert list(cells) == ["tilt-left 10°", "tilt-left 20°", "tilt-right 20°",
                           "tilt-right 30°"]
    left10 = cells["tilt-left 10°"]
    assert left10.images == 6 and left10.subjects == 2
    assert left10.system - left10.kinovea == pytest.approx(-1.0)
    # 量角器：開始值 9、10，結束值加 2，平均 (10 + 11) / 2 = 10.5 左傾為正；右傾為負。
    assert left10.protractor == pytest.approx(10.5)
    assert cells["tilt-right 20°"].protractor == pytest.approx(-20.5)


def test_images_that_cannot_be_compared_are_counted_with_their_reason(tmp_path):
    root = _subjects(tmp_path / "data", count=1)
    out, _, key_path = _export(tmp_path, root)
    key = load_key(key_path)
    victim = next(k for k in key.values() if k.condition == "tilt-left")
    csv_path = Path(victim.segment_csv)
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("frame,"))
    for i in range(header + 1, len(lines)):
        if lines[i].split(",")[0] == victim.frame:
            parts = lines[i].split(",")
            parts[2] = "0"                                  # usable = 0
            lines[i] = ",".join(parts)
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    other = next(k for k in key.values() if k.condition == "tilt-right")
    Path(other.segment_csv).unlink()                        # 整份 CSV 不見了
    # 有一份 CSV 不見了，_fill 讀不到它，所以直接填一個固定值，只看原因。
    sheet = _sheet(out)
    rows = list(csv.DictReader(sheet.read_text(encoding="utf-8").splitlines()))
    for row in rows:
        row[COLUMN] = "5"
    with sheet.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = analyse(read_annotations([out], key), key)
    reasons = {i.reason for i in report.images if i.reason}
    assert "系統那一幀被略過（沒有有效角度）" in reasons
    assert any(r.startswith("讀不到逐幀 CSV") for r in reasons)
    text = report_text(report)
    assert "沒納入" in text and "系統那一幀被略過" in text


def test_two_annotators_are_averaged_per_image(tmp_path):
    root = _subjects(tmp_path / "data", count=1)
    out, _, key_path = _export(tmp_path, root, annotators=("K1", "K2"))
    key = load_key(key_path)
    _fill(out, key_path, lambda system, k: system + 2.0, annotator="K1")
    _fill(out, key_path, lambda system, k: system + 4.0, annotator="K2")
    report = analyse(read_annotations([out], key), key)
    assert {i.annotators for i in report.images} == {2}
    assert report.bias == pytest.approx(-3.0)
    assert "2 位" in report_text(report)


def test_the_interval_needs_enough_subjects(tmp_path):
    few = tmp_path / "few"
    out, _, key_path = _export(tmp_path, _subjects(few / "data", count=2))
    key = load_key(key_path)
    noise = iter([1.0, -1.0] * 100)
    _fill(out, key_path, lambda system, k: system + next(noise))
    report = analyse(read_annotations([out], key), key, reps=200)
    assert report.interval is None and "不給信賴區間" in report_text(report)

    many_root = tmp_path / "many"
    many_out = many_root / "pack"
    result, _, _ = export_pack([_subjects(many_root / "data", count=6)], many_out, ["K1"],
                               seed=1, key_path=many_root / "key" / "key.csv")
    many_key = load_key(many_root / "key" / "key.csv")
    noise = iter([1.0, -1.0, 2.0, -2.0] * 100)
    _fill(many_out, many_root / "key" / "key.csv", lambda system, k: system + next(noise))
    report = analyse(read_annotations([many_out], many_key), many_key, reps=200)
    assert report.subjects == 6 and report.interval is not None
    assert report.interval.unit == "受試者" and report.interval.low is not None
    assert "95% 區間" in report_text(report)


def test_the_images_csv_lists_every_image_with_its_inclusion(tmp_path):
    root = _subjects(tmp_path / "data", count=1)
    out, _, key_path = _export(tmp_path, root)
    key = load_key(key_path)
    _fill(out, key_path, lambda system, k: system + 1.0)
    report = analyse(read_annotations([out], key), key)
    rows = list(csv.DictReader(images_csv(report).splitlines()))
    assert len(rows) == 12 and {r["included"] for r in rows} == {"1"}
    assert {r["condition"] for r in rows} == {"tilt-left", "tilt-right"}
    assert {r["target_angle_deg"] for r in rows} == {"10", "20", "30"}


def test_a_report_with_no_usable_image_does_not_crash(tmp_path):
    root = _subjects(tmp_path / "data", count=1)
    out, _, key_path = _export(tmp_path, root)
    key = load_key(key_path)
    shutil.rmtree(root)                                     # 逐幀 CSV 全不見
    sheet = _sheet(out)
    rows = list(csv.DictReader(sheet.read_text(encoding="utf-8").splitlines()))
    for row in rows:
        row[COLUMN] = "3"
    with sheet.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = analyse(read_annotations([out], key), key)
    assert report.used == [] and report.r is None
    assert "沒有可以對的圖" in report_text(report)
