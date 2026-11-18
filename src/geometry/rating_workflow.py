"""`rating-pack` 與 `ratings` 兩個子指令的流程。回傳要印的文字，不碰終端機。

放在這裡而不是 `posture.py`，是為了讓 posture.py 只多兩個子指令的殼，
流程本身測得到。
"""
from __future__ import annotations

from pathlib import Path

from .rating_alarm import compare_methods, reliability, segment_results, summarise
from .rating_config import load_config
from .rating_import import find_rating_files, read_key, read_ratings
from .rating_pack import find_items, write_pack
from .rating_report import (NO_RATINGS, alarm_text, coverage_text,
                            reliability_text, segments_csv)
from .rating_stats import BOOTSTRAP_REPS


def export_pack(paths: list[Path], out_dir: Path, raters: list[str],
                config_path: Path | None = None, seed: int | None = None,
                per_segment: int = 0, allow_overlay: bool = False,
                key_path: Path | None = None) -> str:
    """找圖、寫盲評包，回傳摘要。出錯時丟 RatingPackError／RatingConfigError。"""
    columns = load_config(config_path).columns if config_path is not None else None
    items, skipped, overlay = find_items(paths, per_segment, allow_overlay)
    result = write_pack(items, out_dir, raters, columns, seed, key_path)
    kinds = {k: sum(1 for i in items if i.kind == k) for k in ("snapshot", "kendall")}
    segments = len({i.segment for i in items})
    subjects = len({i.subject for i in items})
    lines = [
        f"盲評包寫到 {out_dir}",
        f"  {result.items} 張（快照 {kinds['snapshot']}、Kendall 照片 {kinds['kendall']}），"
        f"{segments} 段、{subjects} 位受試者",
        f"  給評分者：{out_dir}/<評分者>/（各自一份，順序各自隨機）",
        f"  key 檔：{result.key_path}（在盲評包外面），不要給評分者",
    ]
    if columns is None:
        lines.append("  沒有給 --config，評分表只有 rater_id、blind_id、order、note，"
                     "評分欄要等量表定案後用設定檔重新匯出")
    for rater, count in result.adjacent_same_subject.items():
        if count:
            lines.append(f"  注意：{rater} 的順序裡還有 {count} 處同一位受試者相鄰"
                         f"（某位受試者的照片超過一半，排不開）")
    if result.reencoded_jpeg:
        lines.append(f"  {result.reencoded_jpeg} 張 JPEG 已依 EXIF 方向轉正後存成不帶中繼資料的 PNG")
    if overlay:
        lines.append("  用了疊加骨架的快照（--allow-overlay）：" + "、".join(overlay))
    for note in skipped:
        lines.append(f"  略過 {note}")
    return "\n".join(lines)


def analyse_ratings(key_path: Path, rating_paths: list[Path], config_path: Path,
                    sessions_root: Path | None = None, out_dir: Path | None = None,
                    reps: int = BOOTSTRAP_REPS) -> str:
    """讀評分、算信度與有效警報。沒有評分檔時回傳「沒有評分檔」的說明，不丟例外。"""
    files = find_rating_files(rating_paths)
    if not files:
        where = "、".join(str(p) for p in rating_paths) or "（沒有指定）"
        return (f"{NO_RATINGS}：{where} 底下沒有評分者交回的 CSV。"
                f"信度與有效警報這兩項跳過，其他分析（analyse、cohort）不受影響。")
    config = load_config(config_path)
    key = read_key(key_path)
    ratings = read_ratings(files, config, key)
    if not ratings.ratings:
        return (coverage_text(ratings, len(key)) + "\n"
                f"{NO_RATINGS}：評分表裡沒有任何填好的列，信度與有效警報跳過。")
    rel = reliability(ratings, key, config, reps)
    results = segment_results(ratings, key, config, sessions_root)
    summaries = summarise(results, config)
    comparison = compare_methods(results, config, summaries)
    report = "\n\n".join([
        coverage_text(ratings, len(key)),
        reliability_text(rel),
        alarm_text(results, summaries, comparison, config),
    ])
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ratings.txt").write_text(report, encoding="utf-8")
        (out_dir / "rating_segments.csv").write_text(segments_csv(results, config),
                                                     encoding="utf-8")
        report += f"\n\n已寫到 {out_dir / 'ratings.txt'} 與 {out_dir / 'rating_segments.csv'}"
    return report
