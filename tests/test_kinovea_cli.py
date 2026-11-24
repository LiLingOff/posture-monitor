"""kinovea-pack 與 kinovea 兩個子指令：參數解析、白話訊息、不碰相機。"""
from types import SimpleNamespace

import pytest

from geometry.kinovea import COLUMN, load_key
from posture_loader import load_posture_module
from test_kinovea import _fill, _sheet, _subjects

posture = load_posture_module("posture_kinovea_cli")


def _pack_args(tmp_path, root, **overrides):
    values = dict(paths=[root], out=tmp_path / "pack", key_out=tmp_path / "key" / "key.csv",
                  annotators="K1", per_segment=3, seed=1, allow_overlay=False)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_the_parser_knows_both_subcommands_and_their_defaults():
    parser = posture.build_parser()
    pack = parser.parse_args(["kinovea-pack", "data/sessions", "--out", "x"])
    assert pack.mode == "kinovea-pack" and pack.annotators == "K1" and pack.per_segment == 3
    read = parser.parse_args(["kinovea", "--key", "k.csv"])
    assert read.mode == "kinovea" and read.sheets == [] and read.bootstrap is None


def test_the_pack_command_prints_where_everything_went(tmp_path, capsys):
    root = _subjects(tmp_path / "data")
    posture._run_kinovea_pack(_pack_args(tmp_path, root))
    out = capsys.readouterr().out
    assert "24 張側傾段的原圖（每段 3 張）" in out
    assert f"填 rating_sheet.csv 的 {COLUMN}" in out and "不要給標註者" in out
    assert (tmp_path / "key" / "key.csv").exists()


def test_the_pack_command_turns_pack_errors_into_a_plain_message(tmp_path):
    with pytest.raises(SystemExit, match="盲評包是空的"):
        posture._run_kinovea_pack(_pack_args(tmp_path, tmp_path / "nothing"))


def test_reading_back_with_no_sheet_says_so_instead_of_crashing(tmp_path, capsys):
    root = _subjects(tmp_path / "data", count=1)
    posture._run_kinovea_pack(_pack_args(tmp_path, root))
    capsys.readouterr()
    posture._run_kinovea(SimpleNamespace(key=tmp_path / "key" / "key.csv", sheets=[],
                                         out=None, bootstrap=None))
    assert "沒有標註檔" in capsys.readouterr().out


def test_reading_back_writes_the_report_and_the_image_table(tmp_path, capsys):
    root = _subjects(tmp_path / "data")
    posture._run_kinovea_pack(_pack_args(tmp_path, root))
    key_path = tmp_path / "key" / "key.csv"
    _fill(tmp_path / "pack", key_path, lambda system, k: system + 1.0)
    capsys.readouterr()
    result = tmp_path / "result"
    posture._run_kinovea(SimpleNamespace(key=key_path, sheets=[tmp_path / "pack"],
                                         out=result, bootstrap=100))
    out = capsys.readouterr().out
    assert "Pearson r" in out and "納入 24 張" in out
    assert (result / "kinovea.txt").exists() and (result / "kinovea_images.csv").exists()
    assert len(load_key(key_path)) == 24


def test_a_bad_sheet_is_reported_as_a_plain_message(tmp_path):
    root = _subjects(tmp_path / "data", count=1)
    posture._run_kinovea_pack(_pack_args(tmp_path, root))
    sheet = _sheet(tmp_path / "pack")
    lines = sheet.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace(",,", ",abc,", 1)
    sheet.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="要是數字"):
        posture._run_kinovea(SimpleNamespace(key=tmp_path / "key" / "key.csv",
                                             sheets=[tmp_path / "pack"], out=None,
                                             bootstrap=None))
