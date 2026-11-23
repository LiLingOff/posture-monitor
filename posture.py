"""端到端量測：相機 → 關鍵點 → 三角測量 → 坐姿角度。

放在 repo 根目錄而不是做成 `python -m src.geometry.cli`，是因為 `src/geometry`
用的是絕對匯入（`from calibration...`），需要 `src/` 在 sys.path 上；
而 `-m src.geometry.cli` 的 sys.path[0] 是 repo 根目錄。這裡先補上路徑再匯入，
與 `conftest.py` 給測試用的做法一致。

用法：
  python posture.py once
  python posture.py live --baseline data/baselines/<受試者>.json
  python posture.py analyse data/sessions/<記錄>.csv

解析度不給的話照標定檔要求，因為內參綁在解析度上，相機退回自己的預設值
（實機遇過 640x480）算出來的深度沒有意義。
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import re
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from calibration.device import (CameraReadError,  # noqa: E402
                                describe_camera_open_failure,
                                describe_resolution_mismatch,
                                find_camera_index, merged_capture_size,
                                open_camera, split_merged_frame)
from calibration.charuco import CharucoBoardSpec, detect_charuco  # noqa: E402
from calibration.stereo_calibration import StereoCalibrationResult  # noqa: E402
from geometry.baseline import (BaselineCollector,  # noqa: E402
                               PostureBaseline, baseline_quality_warnings)
from geometry.cohort import Separation, collect  # noqa: E402
from geometry.cohort_report import (session_rows, to_csv,  # noqa: E402
                                    to_text)
from geometry.judgement import MissCounter, Posture, PostureJudge  # noqa: E402
from geometry.measurement_log import MeasurementLog  # noqa: E402
from geometry.monitor import MonitorState  # noqa: E402
from geometry.pipeline import (PersonMatch,  # noqa: E402
                               estimate_theta_ca_precision_deg,
                               format_measurement, match_person_pair,
                               measure_posture, unusable_reason)
from geometry.recording import (CameraRetry, Recording,  # noqa: E402
                                Session, Snapshots, StuckWatcher,
                                forget_if_stale, write_frame)
from geometry.profile_reference import (compare as compare_profile,  # noqa: E402
                                        parse_plumb, parse_windows, pick_person,
                                        profile_angle_deg, report_lines,
                                        sample_from_person, stereo_segment,
                                        summarize_window)
from geometry.rating_config import RatingConfigError  # noqa: E402
from geometry.rating_import import RatingImportError  # noqa: E402
from geometry.rating_pack import RatingPackError  # noqa: E402
from geometry.rating_workflow import analyse_ratings, export_pack  # noqa: E402
from geometry.session_analysis import analyse_session  # noqa: E402
from geometry.session_report import format_report, segment_table  # noqa: E402
from geometry.smoothing import RollingAngle  # noqa: E402
from geometry.triangulation import triangulate_points  # noqa: E402
from geometry.trip_report import trip_csvs, trip_sections  # noqa: E402
from geometry.trips import read_groups  # noqa: E402
from geometry.live_report import (elapsed_text, live_line,  # noqa: E402
                                  monitor_line, study_summary_lines,
                                  summary_lines, transition_lines)
from geometry.scale_check import (check_scale,  # noqa: E402
                                 format_scale_check)
from geometry.study import (baseline_path, build_trip_options,  # noqa: E402
                            condition_brief, end_angle_footer, end_angle_for,
                            is_tilt, is_typing_segment, next_trial,
                            parse_conditions, parse_number_list,
                            parse_optional_conditions, review,
                            segment_metadata, session_paths, TYPING_TEXT,
                            UPRIGHT)
from geometry.preflight import (FrameProbe, git_items,  # noqa: E402
                                probe_items, verdict_lines)
from geometry.preflight import Item as PreflightItem  # noqa: E402
from geometry.preflight import all_clear as preflight_clear  # noqa: E402
from geometry.run_plan import (MAX_REDO_PER_SEGMENT, check_segment,  # noqa: E402
                               describe_offset, discard_segment,
                               estimated_minutes, load_plan, parse_offset,
                               trip_argv)
from geometry.terminal import cell, truncate  # noqa: E402
from pose.engine import (LightweightOpenPoseEngine,  # noqa: E402
                         LightweightOpenPoseModelPaths)
from view.live_view import LiveView  # noqa: E402
from view.window import NO_DISPLAY, open_display  # noqa: E402


def _build_engine(args) -> LightweightOpenPoseEngine:
    paths = LightweightOpenPoseModelPaths(
        checkpoint=args.checkpoint, engine_cache=args.engine_cache, repo_dir=args.repo_dir
    )
    return LightweightOpenPoseEngine(
        paths, precision=args.precision, device=args.device,
        input_height=args.input_height, subpixel=not args.no_subpixel,
    )


def _match(calib, left_detections, right_detections) -> PersonMatch:
    """挑出左右兩眼看到的同一個人。兩邊各取第一個是不對的，見 match_person_pair。"""
    if not left_detections or not right_detections:
        missing = "左" if not left_detections else "右"
        raise RuntimeError(
            f"{missing}眼沒有偵測到人。確認受試者在畫面內、光線足夠，"
            f"並檢查左右畫面是不是同一個場景"
        )
    try:
        return match_person_pair(calib, left_detections, right_detections)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def _open_selected_camera(args, calib=None):
    """開啟相機。--camera auto 時先自動挑一個讀得出畫面的節點。

    沒指定 --width/--height 時用標定檔的解析度當目標。這不是方便而已：
    內參綁在解析度上，不給的話相機會退回它自己的預設值（實機遇過 640x480），
    而那個畫面算出來的深度沒有意義。標定檔已經知道該用多少，不該叫人每次手打。
    """
    if calib is not None and not (args.width and args.height):
        args.width, args.height = merged_capture_size(
            calib.image_size, args.vertical_split
        )
        _step(f"沒指定解析度，照標定檔要求 {args.width}x{args.height}")
    if args.camera == "auto":
        args.camera, how = find_camera_index(args.width, args.height)
        _step(how)
    cap = open_camera(args.camera, args.width, args.height)
    if not cap.isOpened():
        raise RuntimeError(describe_camera_open_failure(args.camera))
    return cap


def _grab_pair(cap, args):
    ok, frame = cap.read()
    if not ok:
        raise CameraReadError("讀取相機影格失敗")
    return split_merged_frame(frame, args.vertical_split, args.swap_lr)


def _require_matching_resolution(calib, left_frame, args) -> None:
    """解析度不符就在取樣之前停下來。

    警告沒有用。2026-09-29 實機那次相機退回 640x480，程式照樣跑完 192 幀，
    每一幀都算出深度 104mm，而印出的原因指向左右配對錯誤，查錯了方向。
    """
    problem = describe_resolution_mismatch(
        calib.image_size,
        (left_frame.shape[1], left_frame.shape[0]),
        args.vertical_split,
    )
    if problem is not None:
        raise SystemExit(problem)


def _measure_once(engine, calib, cap, args):
    """回傳（量測, 配對, 原始左右畫面）。畫面留著給 --save-frames 用。"""
    frames = _grab_pair(cap, args)
    match = _match(calib, engine.infer(frames[0]), engine.infer(frames[1]))
    measurement = measure_posture(calib, match.left, match.right, args.min_confidence)
    return measurement, match, frames


def _save_frames(left_frame, right_frame, out_dir: Path, left_kp=None, right_kp=None) -> Path:
    """把左右兩眼的畫面存下來，偵測到的關鍵點疊上去。

    診斷訊息說得出「左右配對錯誤」，但說不出相機到底看到什麼。人在不在畫面裡、
    頭有沒有被切掉、畫面裡還有什麼被當成人，這些看一眼就知道，
    用座標猜要來回好幾輪。
    """
    for name, frame, kp in (("left", left_frame, left_kp), ("right", right_frame, right_kp)):
        write_frame(frame, out_dir / f"{name}.png", kp)
    return out_dir


def _describe_match(match: PersonMatch) -> str:
    """偵測到幾個人、挑中的那一對對得多齊。配錯人是深度離譜的頭號成因。"""
    line = (
        f"偵測到的人數     左眼 {match.left_count}   右眼 {match.right_count}"
        f"   挑中的一位距離 {match.distance_mm:.0f} mm"
        f"，垂直視差中位數 {match.median_vertical_disparity_px:.2f} px"
    )
    if match.rejected_farther:
        line += f"\n（排除了 {match.rejected_farther} 位更遠的人，背景有人經過時會用到這一條）"
    elif match.was_ambiguous:
        line += f"\n（有 {match.rejected_pairs} 種其他配法被排除；畫面裡不只一個偵測結果）"
    return line + "\n"


def _step(message: str) -> None:
    """載入權重與建TensorRT engine都要數秒到數分鐘，中間不出聲會像當掉。"""
    print(message, flush=True)


def _discard_frames(cap, count: int) -> None:
    """自動曝光要幾張影格才穩定，前幾張通常偏暗，關鍵點信心會低一截。"""
    for _ in range(count):
        cap.read()


def _run_once(args) -> None:
    calib = StereoCalibrationResult.load(args.calibration)
    _step(f"標定檔 {args.calibration}（基線 {calib.baseline_mm:.2f} mm，"
          f"單眼 {calib.image_size[0]}x{calib.image_size[1]}）")

    engine = _build_engine(args)
    cap = _open_selected_camera(args, calib)
    try:
        _step(f"等自動曝光穩定，丟掉前 {args.discard} 張")
        _discard_frames(cap, args.discard)

        left_frame, right_frame = _grab_pair(cap, args)
        _step(f"取得畫面，單眼 {left_frame.shape[1]}x{left_frame.shape[0]}")
        _require_matching_resolution(calib, left_frame, args)

        # 載入權重、搬上GPU，fp16還要建TensorRT engine，這段可能要等上幾分鐘，
        # 中間沒有任何輸出會看起來像當掉。
        _step(f"載入模型（{args.precision} / {args.device}）")
        engine.warmup(left_frame)
        _step("開始推論")

        left_detections = engine.infer(left_frame)
        right_detections = engine.infer(right_frame)
        if args.save_frames:
            # 配對失敗時也要存得下來，所以先存原始畫面，配對成功再補上關鍵點
            _save_frames(left_frame, right_frame, args.save_frames,
                         left_detections[0] if left_detections else None,
                         right_detections[0] if right_detections else None)
            _step(f"畫面已存到 {args.save_frames}")
        match = _match(calib, left_detections, right_detections)
        if args.save_frames:
            _save_frames(left_frame, right_frame, args.save_frames, match.left, match.right)
        measurement = measure_posture(calib, match.left, match.right, args.min_confidence)
    finally:
        cap.release()

    display = _open_display(args)
    try:
        # once 量一幀就結束，畫面是靜止的，所以停著等按鍵而不是重畫。
        # 這是架設相機時最有用的一種：看得出鏡頭到底對到什麼。
        _tell(display.hold(LiveView.build(
            frames=(left_frame, right_frame), match=match, measurement=measurement,
            corrected=(measurement.theta_ca_deg, measurement.theta_sym_deg),
            ca_window=RollingAngle(1), sym_window=RollingAngle(1),
            state=Posture.UNKNOWN, judging=False,
            ka_instant=measurement.theta_ka_deg,
            skip_reason=unusable_reason(measurement),
            rejected=0, frames_seen=1, keys=(("[任意鍵] 關閉", "[Any key] Close"),),
        )))
    finally:
        display.close()

    print()
    print(_describe_match(match))
    print(format_measurement(
        measurement, match.left, match.right, show_all_keypoints=args.all_keypoints
    ))


def _board_corners_3d(calib, board, left_frame, right_frame):
    """兩眼都看得到的角點，三角測量成 {角點編號: 3D 座標}。

    靠編號配對，不靠位置。ChArUco 的每個角點都有自己的編號，所以左右兩眼
    看到的是不是同一個點完全沒有疑義,這正是它比棋盤格適合這件事的地方，
    也是這條檢查能把幾何與關鍵點分開的原因。
    """
    found = []
    for frame in (left_frame, right_frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found.append(detect_charuco(gray, board))
    if found[0] is None or found[1] is None:
        return {}

    # 角點的形狀各版本不一（(N,1,2) 或 (N,2)），統一攤平再配。
    left_map, right_map = ({int(i): c for c, i in
                            zip(np.reshape(found[k][0], (-1, 2)),
                                np.reshape(found[k][1], -1))}
                           for k in (0, 1))
    shared = sorted(set(left_map) & set(right_map))
    if not shared:
        return {}

    points = triangulate_points(
        calib,
        np.asarray([left_map[i] for i in shared], dtype=np.float64),
        np.asarray([right_map[i] for i in shared], dtype=np.float64),
    )
    return {i: p for i, p in zip(shared, points) if np.isfinite(p).all()}


def _run_scale(args) -> None:
    """拿標定板驗證「算出來的 mm 是不是真的 mm」。

    不載入姿態模型：這條檢查刻意不經過它，量到的偏差才只可能來自標定。
    """
    if args.square_size_mm <= 0:
        raise SystemExit("--square-size-mm 要大於 0，它是比較的基準")
    if args.frames < 1:
        raise SystemExit("--frames 至少要 1")
    if args.marker_size_mm <= 0 or args.marker_size_mm >= args.square_size_mm:
        raise SystemExit(
            f"--marker-size-mm（{args.marker_size_mm:g}）要小於方格邊長"
            f"（{args.square_size_mm:g}）。這個選項的預設值是 18，"
            "換了板子就要連同 --dictionary 一起填成標定時那組")

    calib = StereoCalibrationResult.load(args.calibration)
    _step(f"標定檔 {args.calibration}（基線 {calib.baseline_mm:.2f} mm，"
          f"單眼 {calib.image_size[0]}x{calib.image_size[1]}）")
    spec = CharucoBoardSpec(
        squares_x=args.squares_x, squares_y=args.squares_y,
        square_size_mm=args.square_size_mm, marker_size_mm=args.marker_size_mm,
        dictionary_name=args.dictionary, legacy_pattern=args.legacy_pattern,
    )
    board = spec.build_board()
    _step(f"板子 {args.squares_x}x{args.squares_y}，方格 {args.square_size_mm:.2f} mm")

    cap = _open_selected_camera(args, calib)
    per_frame = []
    try:
        _step(f"等自動曝光穩定，丟掉前 {args.discard} 張")
        _discard_frames(cap, args.discard)
        for index in range(args.frames):
            left_frame, right_frame = _grab_pair(cap, args)
            if index == 0:
                _require_matching_resolution(calib, left_frame, args)
                if args.save_frames:
                    _save_frames(left_frame, right_frame, args.save_frames)
            corners = _board_corners_3d(calib, board, left_frame, right_frame)
            per_frame.append(corners)
            _print_line(f"第 {index + 1}/{args.frames} 幀，兩眼共同角點 {len(corners)}")
    finally:
        cap.release()
    _print_line("")

    empty = sum(1 for corners in per_frame if not corners)
    if empty:
        print(f"{empty}/{len(per_frame)} 幀一個共同角點都沒有。"
              f"板子要完整入鏡、兩眼都看得到，而且不要反光", flush=True)
    check = check_scale(per_frame, args.squares_x, args.squares_y,
                        args.square_size_mm, frames=len(per_frame))
    print()
    _print_lines(format_scale_check(check, args.tolerance))


def _prepare(args):
    """載入標定、開相機、暖機。回傳 (calib, engine, cap)。

    `study` 只做一次，之後每一段都重用。零間隔就是靠這一點：先前每跑一次
    `live` 都要重載模型，那段時間就是基準與量測之間的空檔，而 2026-09-29
    量到的 63% 誤報率就是從那裡來的。
    """
    calib = StereoCalibrationResult.load(args.calibration)
    engine = _build_engine(args)
    cap = _open_selected_camera(args, calib)

    _discard_frames(cap, args.discard)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("讀取相機影格失敗")
    left_frame = split_merged_frame(frame, args.vertical_split, args.swap_lr)[0]
    _require_matching_resolution(calib, left_frame, args)
    print(f"載入模型（{args.precision} / {args.device}）", flush=True)
    engine.warmup(left_frame)
    return calib, engine, cap


def _record(engine, calib, cap, args, baseline, log, snapshots,
            seconds: float | None = None, display=NO_DISPLAY,
            quiet: bool = False) -> Recording:
    """錄製一段。seconds 是 None 就跑到 Ctrl-C 為止。

    固定秒數是實測逼出來的：2026-09-29 有一次量測跑到受試者忘記維持姿勢，
    整段資料作廢。段落短就不會忘。

    quiet=True 時不印逐幀那一行與狀態變化（狀態變化照樣記進 transitions）。
    受試者在終端機上打字的那一段要用：原地更新的那一行會把打的字蓋掉。
    相機出問題的提示仍然會印，操作者要看得到。
    """
    judge = _make_judge(args, baseline)
    display.begin()
    started_at = time.perf_counter()
    ca_window = RollingAngle(args.window)
    sym_window = RollingAngle(args.window)
    drop_window = RollingAngle(args.window)
    ka_window = RollingAngle(args.window)
    ka_misses = MissCounter()
    rejected = 0
    frames_seen = 0
    consecutive_misses = 0
    retry = CameraRetry(args.camera)
    saved_a_rejected_frame = False
    transitions: list[str] = []
    lost: str | None = None
    session = Session()
    azimuths: list[float] = []
    stuck = StuckWatcher()

    try:
        # 結束旗標放在條件裡而不是迴圈裡 break：它同時涵蓋正常路徑與下面那個
        # continue，而且在 frames_seen += 1 之前，按鍵結束不會灌水到略過率的分母。
        while not display.stopped and (
            seconds is None or (time.perf_counter() - started_at) < seconds
        ):
            frames_seen += 1
            try:
                measurement, match, frames = _measure_once(engine, calib, cap, args)
                retry.ok()
            except RuntimeError as exc:
                lost = retry.failed(exc)
                if lost is not None:
                    break
                # 讀不到畫面與偵測不到人都是這一幀沒有資料，一樣要計入略過，
                # 否則結束時印的數字會與實際寫下的筆數對不起來。
                rejected += 1
                consecutive_misses += 1
                forget_if_stale(ca_window, sym_window, consecutive_misses,
                                drop_window, ka_window)
                _tell(stuck.saw(str(exc)))
                # waitKey 是視窗處理事件與重繪的地方。相機掉線時這條路要走
                # 兩秒半，不打點的話畫面會凍住而且按鍵沒反應。
                _tell(display.show(None))
                _print_line(f"{exc}（已略過 {rejected} 幀）")
                if log is not None:
                    log.write(None, reject_reason=str(exc))
                continue

            reason = unusable_reason(measurement)
            corrected = _corrected(baseline, measurement)
            drop = _shoulder_drop(baseline, measurement)
            ka = None if baseline is None else baseline.correct_ka(measurement.theta_ka_deg)
            if reason is None:
                consecutive_misses = 0
                session.add(corrected, ka)
                if measurement.camera_azimuth_deg is not None:
                    azimuths.append(measurement.camera_azimuth_deg)
                if corrected[0] is not None:
                    ca_window.add(corrected[0])
                if corrected[1] is not None:
                    sym_window.add(corrected[1])
                if drop is not None:
                    drop_window.add(drop)
                if ka is not None:
                    ka_window.add(ka)
            else:
                # 壞掉的幀混進平均比印出來更糟，所以先擋掉再計數。
                rejected += 1
                consecutive_misses += 1
                forget_if_stale(ca_window, sym_window, consecutive_misses,
                                drop_window, ka_window)
                if args.save_frames and not saved_a_rejected_frame:
                    _save_frames(*frames, args.save_frames, match.left, match.right)
                    saved_a_rejected_frame = True

            _tell(stuck.saw(reason))

            # θ_KA 常因髖被擋而沒有值，那一幀仍算數，要自己數缺值才清得掉舊平均。
            ka_misses.update(ka_window, reason is None and ka is not None)

            verdict = None
            if judge is not None:
                *_, verdict = judge.update(ca_window, sym_window, drop_window, ka_window)
                if verdict.changed:
                    line = f"{_elapsed(started_at)}  {verdict.posture.value}：{verdict.reason}"
                    transitions.append(line)
                    if not quiet:
                        # 狀態變化印成獨立的一行，因為它會被原地更新的那一行蓋掉。
                        _print_line("")
                        print("\r" + line, flush=True)

            remaining = (None if seconds is None
                         else seconds - (time.perf_counter() - started_at))
            if not quiet:
                _print_line(live_line(
                    ca_window, sym_window, measurement, corrected, rejected, reason,
                    verdict, remaining
                ))
            _tell(display.show(LiveView.build(
                frames=frames, match=match, measurement=measurement,
                corrected=corrected, ca_window=ca_window, sym_window=sym_window,
                state=Posture.UNKNOWN if verdict is None else verdict.posture,
                skip_reason=reason, rejected=rejected, frames_seen=frames_seen,
                remaining_s=remaining, baseline=baseline, keys=(("[Q] 離開", "[Q] Quit"),),
                drop_mm=drop_window.mean, drop_threshold_mm=_drop_threshold(baseline),
                ka_window=ka_window, ka_instant=ka,
            )))
            _write_frame_row(
                log, snapshots, measurement, match, frames, reason=reason,
                corrected=corrected, ca_window=ca_window, sym_window=sym_window,
                drop=drop, drop_mean=drop_window.mean, verdict=verdict,
                elapsed_s=time.perf_counter() - started_at,
                ka=ka, ka_mean=ka_window.mean,
            )
    except KeyboardInterrupt:
        pass

    if display.stopped:
        # 不記的話，總結只會看到一段莫名其妙的短記錄。
        transitions.append(f"{_elapsed(started_at)}  按鍵結束這一段")

    return Recording(
        session=session, ca_window=ca_window, sym_window=sym_window,
        rejected=rejected, frames=frames_seen, transitions=transitions,
        camera_lost=lost, log_path=None if log is None else log.path,
        azimuths=azimuths,
    )


def _write_frame_row(log, snapshots, measurement, match, frames, *,
                     reason, corrected, ca_window, sym_window,
                     drop, drop_mean, verdict, elapsed_s,
                     ka=None, ka_mean=None) -> None:
    """把這一幀寫進 CSV，時間到了順便存一張畫面。

    錄製與 monitor 兩個迴圈共用。先前各寫一份，於是每加一個 CSV 欄位就要記得
    改兩個地方，而漏掉的那一邊只會安靜地少一欄。
    """
    posture = "" if verdict is None else verdict.posture.value
    if log is not None:
        log.write(
            measurement, reject_reason=reason, corrected=corrected,
            ca_mean=ca_window.mean, sym_mean=sym_window.mean,
            ca_standard_error=ca_window.standard_error, posture=posture,
            shoulder_height_mm=measurement.shoulder_height_mm,
            shoulder_drop_mm=drop, shoulder_drop_mean_mm=drop_mean,
            theta_ka_corrected=ka, theta_ka_mean=ka_mean,
            keypoints_2d=(match.left.points, match.right.points),
        )
    snapshots.maybe_save(
        0 if log is None else log.frames_written, elapsed_s,
        frames[0], match.left, corrected[0], posture,
    )


def _report(recording: Recording, baseline, log) -> None:
    """一段錄完之後印出來的東西。

    相機掉線時也要走完這一段。已經量到的東西才是這次的產出，不該因為
    結束的方式不同就不印。
    """
    print()
    if recording.camera_lost is not None:
        print(recording.camera_lost)
    _print_lines(summary_lines(recording, baseline))
    _print_lines(transition_lines(recording.transitions))
    if log is not None:
        print(f"已記錄 {log.frames_written} 幀到 {log.path}")


def _run_live(args) -> None:
    calib, engine, cap = _prepare(args)
    display = _open_display(args)
    baseline = _load_baseline(args, calib)
    log = _open_log(args, baseline, args.log, args.condition, args.trial,
                    args.baseline)
    snapshots = _open_snapshots(args, args.snapshots)
    print(f"開始量測，平均視窗 {args.window} 幀。"
          + ("Ctrl-C 結束" if args.seconds is None else f"{args.seconds:.0f} 秒後自動結束"),
          flush=True)
    print("要看的是平均值，不是單幀。單幀誤差與判定門檻同量級", flush=True)

    try:
        recording = _record(engine, calib, cap, args, baseline, log, snapshots,
                            args.seconds, display)
        _report(recording, baseline, log)
    finally:
        if log is not None:
            log.close()
        display.close()
        cap.release()
    if recording.camera_lost is not None:
        raise SystemExit(1)


def _run_monitor(args) -> None:
    """一個視窗跑到底：按 c 當場歸零、按 q 離開。

    前作就是這樣用的。它順帶解決了一件本專案吃過大虧的事：在同一個視窗裡
    按 c，基準與量測之間不可能有間隔（原因見 `_prepare`）。`study` 用固定
    流程達成同一件事，這裡用互動達成。
    """
    calib, engine, cap = _prepare(args)
    display = _open_display(args)
    if not args.display:
        print("沒有視窗就收不到 c，這一場只能用 --baseline 預載的基準。"
              "要當場歸零的話加上 --display", flush=True)
    state = MonitorState(
        subject=args.subject, window=args.window, margin=args.margin,
        countdown_s=args.countdown, baseline_s=args.baseline_seconds,
        baseline=_load_baseline(args, calib), no_judge=args.no_judge,
    )
    snapshots = _open_snapshots(args, args.snapshots)
    print(f"視窗裡按 c 取基準（倒數 {args.countdown:.0f} 秒、取樣 "
          f"{args.baseline_seconds:.0f} 秒），按 q 離開", flush=True)

    log, run = None, 0
    started_at = time.perf_counter()
    retry = CameraRetry(args.camera)
    frames_seen = 0
    rejected = 0
    try:
        log, run = _rotate_monitor_log(args, state.baseline, log, run)
        while not state.stopped:
            frames_seen += 1
            try:
                measurement, match, frames = _measure_once(engine, calib, cap, args)
                retry.ok()
            except RuntimeError as exc:
                lost = retry.failed(exc)
                if lost is not None:
                    _tell(lost)
                    break
                rejected += 1
                _tell(display.show(None))
                _print_line(f"{exc}（已略過 {rejected} 幀）")
                if log is not None:
                    log.write(None, reject_reason=str(exc))
                _pump_keys(display, state)
                continue

            result = state.feed(measurement, time.perf_counter())
            if result.reason is not None:
                rejected += 1
            _tell(result.notice)
            if result.baseline_changed:
                # 換了零點，之後的列與之前的列不在同一個基準上，而 analyse 與
                # cohort 都假設一個檔案一個基準，所以換一個檔案。
                _save_monitor_baseline(args, state.baseline)
                log, run = _rotate_monitor_log(args, state.baseline, log, run)

            _print_line(monitor_line(state, result, rejected))
            _tell(display.show(LiveView.build(
                frames=frames, match=match, measurement=measurement,
                corrected=result.corrected, ca_window=state.ca_window,
                sym_window=state.sym_window,
                drop_mm=state.drop_window.mean,
                drop_threshold_mm=state.drop_threshold_mm,
                ka_window=state.ka_window, ka_instant=state.trunk_offset(measurement),
                state=(Posture.UNKNOWN if result.verdict is None
                       else result.verdict.posture),
                skip_reason=result.reason, rejected=rejected,
                frames_seen=frames_seen, baseline=state.baseline,
                mode=result.mode.value, phase_remaining_s=result.phase_remaining_s,
                keys=(("[C] 重新歸零", "[C] Zero"), ("[Q] 離開", "[Q] Quit")),
            )))
            _pump_keys(display, state)

            _write_frame_row(
                log, snapshots, measurement, match, frames,
                reason=result.reason, corrected=result.corrected,
                ca_window=state.ca_window, sym_window=state.sym_window,
                drop=state.shoulder_drop(measurement),
                drop_mean=state.drop_window.mean, verdict=result.verdict,
                elapsed_s=time.perf_counter() - started_at,
                ka=state.trunk_offset(measurement), ka_mean=state.ka_window.mean,
            )
    except KeyboardInterrupt:
        pass
    finally:
        if log is not None:
            log.close()
        display.close()
        cap.release()

    print()
    if frames_seen:
        print(f"共 {frames_seen} 幀，略過 {rejected} 幀"
              f"（{rejected / frames_seen * 100:.0f}%）")
    else:
        print("沒有量到任何一幀")


def _pump_keys(display, state: MonitorState) -> None:
    """把視窗收到的按鍵轉成狀態機的動作。

    分開寫是因為讀取失敗那條路也要處理按鍵：相機掉線時按 q 應該還能離開。
    """
    if display.take_baseline_request():
        state.request_baseline(time.perf_counter())
    if display.stopped:
        state.stop()


def _rotate_monitor_log(args, baseline, log, run: int):
    """換一個 CSV 檔。回傳 (新的 log, 第幾份)。"""
    if log is not None:
        log.close()
    if args.log is None:
        return None, run
    run += 1
    path = args.log.with_name(f"{args.log.stem}-{run}{args.log.suffix or '.csv'}")
    new_log = MeasurementLog(
        path, subject=args.subject, overwrite=args.overwrite,
        condition=args.condition, trial=run, baseline_file=None,
        baseline=baseline, display=args.display,
    )
    print(f"逐幀記錄到 {new_log.path}", flush=True)
    return new_log, run


def _save_monitor_baseline(args, baseline) -> None:
    """當場取的基準也要寫到磁碟，否則示範完就沒了。"""
    if baseline is None:
        return
    stamp = baseline.captured_at.replace(":", "").replace("-", "")
    path = args.baseline_dir / f"{args.subject}-{stamp}.json"
    baseline.save(path, overwrite=True)
    _tell(f"基準存到 {path}")


def _open_log(args, baseline, path, condition: str, trial: int | None,
              baseline_file, extra: dict | None = None):
    """開一份逐幀記錄。中繼資料一併寫進標頭。

    變動的四項（路徑、姿勢、第幾次、基準檔）一律由呼叫端傳，沒有預設值。
    先前那些預設值配上 `getattr(args, ...)` 看起來能共用，實際上 study 自己
    另外建了一份 MeasurementLog，於是同一件事有兩種寫法，而標頭少一個欄位
    只會讓 CSV 安靜地掉進「未標註」那一組。
    """
    if path is None:
        return None
    log = MeasurementLog(
        path, subject=args.subject, overwrite=args.overwrite,
        condition=condition, trial=trial,
        baseline_file=baseline_file, baseline=baseline, display=args.display,
        extra=extra,
    )
    print(f"逐幀記錄到 {log.path}（含被略過的幀）", flush=True)
    return log


def _open_display(args):
    """要視窗就開一個，並且說清楚它該擺在哪。

    視窗是給操作者看的。受試者盯著螢幕會轉頭，而解剖平面是由雙肩連線定義的，
    轉頭會把上半身一起帶過去：2026-09-29 誤報率 17% 那次就是這樣來的，而且
    當場沒有人發現，是事後從資料反推出來的。
    """
    display = open_display(args.display, args.display_width,
                           args.display_eyes, args.font, args.display_lang)
    if args.display:
        print("視窗已開。螢幕要放在受試者看不到的地方；"
              "視窗有焦點時按 q 結束這一段，終端機的 Ctrl-C 照常", flush=True)
    return display


def _open_snapshots(args, out_dir):
    keep_raw = getattr(args, "snapshot_raw", False)
    snapshots = Snapshots(out_dir, args.snapshot_every, keep_raw=keep_raw)
    if snapshots.enabled:
        print(f"每 {args.snapshot_every:.0f} 秒存一張畫面到 {out_dir}，"
              f"事後可以核對當時的姿勢", flush=True)
        if keep_raw:
            print(f"另存不疊加資訊的原圖到 {out_dir / Snapshots.RAW_DIR}，"
                  f"給外部評分者盲評用", flush=True)
    return snapshots


def _load_baseline(args, calib) -> PostureBaseline | None:
    if not args.baseline:
        print("沒有指定個人基準，印出的是原始角度。", flush=True)
        print("判定門檻套在原始角度上會因人而異，正式量測前先跑一次 "
              "python posture.py baseline", flush=True)
        return None
    baseline = PostureBaseline.load(args.baseline)
    print(baseline.describe(), flush=True)
    # 取基準當下看過一次就過去了，載入時要再講一次：這個偏移會進到之後每一次判定。
    expected = estimate_theta_ca_precision_deg(
        calib, baseline.distance_mm, baseline.azimuth_deg
    ) if baseline.distance_mm > 0 else None
    for warning in baseline_quality_warnings(baseline, expected):
        print(f"  需要注意：{warning}", flush=True)
    print("以下的角度都已扣除這個基準，也就是相對這個人端正坐姿的偏移量", flush=True)
    return baseline


def _shoulder_drop(baseline: PostureBaseline | None, measurement) -> float | None:
    """肩膀比端正坐姿低了多少。沒有基準、或基準沒記肩高，就不算。"""
    if baseline is None:
        return None
    return baseline.shoulder_drop_mm(measurement.shoulder_height_mm)


def _drop_threshold(baseline: PostureBaseline | None) -> float | None:
    return None if baseline is None else baseline.shoulder_drop_threshold_mm


def _corrected(baseline: PostureBaseline | None, measurement):
    """有基準就扣掉，沒有就原樣傳回，讓下游不必分兩種情況處理。"""
    if baseline is None:
        return measurement.theta_ca_deg, measurement.theta_sym_deg
    return baseline.correct(measurement.theta_ca_deg, measurement.theta_sym_deg)


def _countdown(seconds: int, message: str = "秒後開始…") -> None:
    """倒數，讓受試者坐定。

    這不是裝飾。2026-09-29 實機：同樣取樣 30 秒，倒數 3 秒那次的單幀散佈是
    ±11.1°、基準誤差 ±2.97°；倒數 20 秒那次是 ±4.5° 與 ±0.83°，而且零略過。
    剛坐下的十幾秒人還在調整，那一段會被平均進基準裡。
    """
    for remaining in range(seconds, 0, -1):
        print(f"\r{remaining} {message}", end="", flush=True)
        time.sleep(1.0)


def _collect_baseline(
    engine, calib, cap, args, seconds: float, display=NO_DISPLAY
) -> tuple[PostureBaseline, list[str]]:
    """請受試者保持不動，取這段時間的平均當作他的零點。

    相機與模型由呼叫端準備好。`study` 與 `monitor` 靠這一點讓取基準與量測
    之間沒有模型重載的空檔（原因見 `_prepare`）。
    """
    collector = BaselineCollector()
    display.begin()
    start = time.perf_counter()
    saved_a_rejected_frame = False
    retry = CameraRetry(args.camera)
    while (elapsed := time.perf_counter() - start) < seconds:
        try:
            measurement, match, frames = _measure_once(engine, calib, cap, args)
            retry.ok()
        except RuntimeError as exc:
            lost = retry.failed(exc)
            if lost is not None:
                # 取基準沒有「已經收到的資料」可以保，中斷比留下半份基準好。
                print()
                raise SystemExit(lost)
            _tell(display.show(None))
            _print_line(str(exc))
            continue
        reason = collector.add(measurement)
        # 存第一張被略過的畫面。全部被略過時，那正是唯一想看的東西。
        if reason and args.save_frames and not saved_a_rejected_frame:
            _save_frames(*frames, args.save_frames, match.left, match.right)
            saved_a_rejected_frame = True
        _print_line(
            f"剩下 {seconds - elapsed:4.1f} 秒   已收 {collector.count:3d} 幀"
            + (f"   略過 {collector.rejected}" if collector.rejected else "")
            + (f"   （{reason}）" if reason else "")
        )
        _tell(display.show(LiveView.build(
            frames=frames, match=match, measurement=measurement,
            corrected=(measurement.theta_ca_deg, measurement.theta_sym_deg),
            ca_window=RollingAngle(1), sym_window=RollingAngle(1),
            state=Posture.UNKNOWN, skip_reason=reason,
            rejected=collector.rejected, frames_seen=collector.count,
            remaining_s=seconds - elapsed, notice="取基準中，請保持不動",
        )))
    print()

    baseline = collector.finish(args.subject, time.perf_counter() - start)
    print()
    print(baseline.describe())
    # 警告一起回傳而不是掛在 baseline 上：PostureBaseline 是 frozen 的，
    # 而且這些警告描述的是「怎麼取的」，不是基準本身的內容。
    warnings = collector.quality_warnings(baseline)
    for warning in warnings:
        print(f"  需要注意：{warning}")
    return baseline, warnings


def _run_baseline(args) -> None:
    calib, engine, cap = _prepare(args)
    display = _open_display(args)
    try:
        print()
        print(f"請 {args.subject} 坐正、目視前方、雙肩放鬆，保持不動 {args.seconds:.0f} 秒。")
        _countdown(args.countdown)
        print("\r開始取樣，請保持不動        ", flush=True)
        baseline, _ = _collect_baseline(engine, calib, cap, args, args.seconds,
                                        display)
    finally:
        display.close()
        cap.release()

    baseline.save(args.out, overwrite=args.overwrite)
    print()
    print(f"已存到 {args.out}。之後這樣用：")
    print(f"  python posture.py live --baseline {args.out} --log data/sessions/xxx.csv ...")


def _show_typing_text() -> None:
    """把要打的英文印在終端機上，按終端機寬度在字與字之間換行。"""
    width = min(100, shutil.get_terminal_size(fallback=(100, 24)).columns - 1)
    print()
    print("請受試者看著下面這段英文，在這個視窗直接打字：")
    print("-" * width)
    print(textwrap.fill(TYPING_TEXT, width=width))
    print("-" * width, flush=True)


def _discard_typed_input() -> None:
    """丟掉輸入緩衝裡還沒被讀走的字。

    錄製時沒有人讀 stdin，受試者打的字（含中途按的 Enter）都堆在緩衝裡。
    stdin 不是終端機（管線、測試）時什麼都不做。
    """
    try:
        if not sys.stdin.isatty():
            return
    except (AttributeError, ValueError, OSError):
        return
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
        return
    except ImportError:
        pass
    except (OSError, ValueError):
        return
    try:
        import msvcrt  # Windows 開發機沒有 termios
        while msvcrt.kbhit():
            msvcrt.getwch()
    except (ImportError, OSError):
        pass


def _prompt(message: str, interactive: bool = True) -> None:
    """印出指導語並等使用者按 Enter。

    `--no-prompt` 時只印不等，讓沒有人在旁邊的情況也跑得完（例如自己一個人
    設定好之後走回座位，或是在測試裡）。
    """
    print()
    print(message, flush=True)
    if interactive:
        try:
            input("  坐定之後按 Enter 開始…")
        except (EOFError, OSError):
            # stdin 關掉、被導向、或在 nohup 底下都讀不到，那時候要直接往下走
            # 而不是當掉。OSError 也要接：管線與 pytest 給的是這一個。
            print()


def _run_study(args) -> None:
    """一個行程跑完整套流程：取基準，接著逐一量各種姿勢。

    流程本身就是資料品質的一部分。2026-09-29 那晚四次量測作廢三次，沒有一次
    是程式算錯：解析度不符、基準隔了九分鐘、量測中轉頭看螢幕、受試者不自覺
    前傾。把流程寫進程式，做錯就變難。

    最關鍵的一點是**基準與量測之間沒有模型重載的空檔**。先前那兩件事是兩次
    獨立的指令，中間要重新載入模型，受試者就會站起來活動一下。誤報率 63%
    就是從那個空檔來的。

    `--before-baseline` 的段落排在取基準之前（2026-11-18）：自然坐若排在
    「請坐正」之後，受試者已經意識到坐姿，量到的不是平常的坐法。這些段落
    不扣基準，只記原始角度，分析時與同一趟（同一個 trip_id）的端正坐配對。
    `--conditions` 是空的就不取基準（趟 5，表4）。
    """
    # 選項對不上的話在開相機之前就停。trip 在這裡建一次直接傳進去，
    # trip_id（study 開始的時間）才與拆出 _study_trip 之前相同。
    before, conditions = _study_plan(args)
    trip = _trip_options(args, before + conditions)
    calib, engine, cap = _prepare(args)
    display = NO_DISPLAY
    try:
        # 視窗開不起來（例如沒有圖形環境）時相機也要放掉，所以放在 try 裡。
        display = _open_display(args)
        recordings, baseline, cap = _study_trip(engine, calib, cap, args, display,
                                                trip=trip)
    finally:
        display.close()
        if cap is not None:
            cap.release()

    _study_summary(recordings, args, baseline)


def _study_trip(engine, calib, cap, args, display=NO_DISPLAY, check=None,
                trip=None, on_camera_lost=None):
    """跑一趟（同一次坐下、一份基準），回傳 (錄製, 基準, 相機)。

    相機與模型由呼叫端準備好，`study` 跑一趟、`run-plan` 連跑好幾趟都用它，
    模型只載入一次。相機掉線會換一個新的 cap，所以要回傳給呼叫端釋放；
    中途出錯時，換過的那個 cap 呼叫端拿不到，在這裡放掉。

    `check(recording, recordings, baseline, first_upright)` 是 `run-plan` 的
    當場檢查，回傳 "keep"、"redo"（作廢這段、同一段再量）或 "rebaseline"
    （作廢這段、重取基準再量）。作廢的檔案由 check 自己搬走。`study` 不給，
    行為與先前完全相同。

    `on_camera_lost(recording)` 也是 `run-plan` 用的：掉線的殘段交給它搬走，
    重量那段才沿用同一個編號（GPT 審查第 1 條）。`study` 不給，殘段照舊留著、
    標 `camera_lost=1`，重量那段接續編號。
    """
    before, conditions = _study_plan(args)
    if trip is None:
        trip = _trip_options(args, before + conditions)
    root = args.out_dir
    recordings = []
    reopens = 0
    baseline = None
    queue = _study_queue(before, conditions)
    first_upright_seen = False
    given_cap = cap
    try:
        while queue:
            item = queue.pop(0)
            condition, is_before, index, tilt_index = item
            if not is_before and baseline is None:
                baseline = _study_baseline(engine, calib, cap, args, root, display)
            recording = _study_one(
                engine, calib, cap, args, None if is_before else baseline,
                condition, root, display, before=is_before,
                trip=trip, segment_index=index, tilt_index=tilt_index)
            recordings.append(recording)
            if recording.camera_lost is None:
                if check is None:
                    continue
                first_upright = (not is_before and condition == UPRIGHT
                                 and not first_upright_seen)
                action = check(recording, recordings, baseline, first_upright)
                if action == "keep":
                    first_upright_seen = first_upright_seen or first_upright
                    continue
                recordings.pop()
                if action == "rebaseline":
                    baseline = None
                queue.insert(0, item)
                continue
            if on_camera_lost is not None:
                recordings.pop()
                on_camera_lost(recording)
            # 2026-10-06 實機：head-forward 那段掉線之後，下一段 head-back 拿著
            # 已經消失的 video0 繼續跑，一幀都沒讀到就結束了。掉線時節點編號
            # 每一次都會移位，原本的 cap 不可能再讀到東西。
            if reopens >= _STUDY_REOPEN_BUDGET:
                print(f"這一趟相機已經掉了 {reopens + 1} 次，剩下的段落不量了。"
                      "這是硬體的問題，先跑 dmesg | tail -40 看核心怎麼說", flush=True)
                break
            cap.release()
            cap = _reopen_camera(args, calib=calib)
            reopens += 1
            if cap is None:
                print("找不到相機，剩下的段落不量了。已經量完的照常可以分析", flush=True)
                break
            # 同一種姿勢重量一次。改成跳到下一種的話，這一種就得另外找一趟補，
            # 而那一趟的基準不是這一份。
            queue.insert(0, item)
    except BaseException:
        if cap is not None and cap is not given_cap:
            cap.release()
        raise
    return recordings, baseline, cap


def _study_queue(before: list[str], conditions: list[str]) -> list[tuple]:
    """依序要量的段落：(條件, 是否在基準之前, 第幾段, 第幾段側傾或 None)。

    位置在排程時就定下來，掉線重量時同一個項目插回最前面，標頭的
    目標角度與方位角標記才不會因為多量一次而錯位。
    """
    queue, tilts = [], 0
    for index, (condition, is_before) in enumerate(
            [(c, True) for c in before] + [(c, False) for c in conditions]):
        tilt_index = None
        if is_tilt(condition):
            tilt_index, tilts = tilts, tilts + 1
        queue.append((condition, is_before, index, tilt_index))
    return queue


def _study_plan(args) -> tuple[list[str], list[str]]:
    """拆出 (基準前的段落, 基準後的段落)。

    沒給 --before-baseline 時與先前完全一樣：--conditions 不能是空的。
    給了的話兩邊都可以空，但不能兩邊都空。
    """
    before_text = getattr(args, "before_baseline", None)
    if before_text is None:
        try:
            return [], parse_conditions(args.conditions)
        except ValueError as exc:
            raise SystemExit(f"--conditions {exc}")
    before = parse_optional_conditions(before_text)
    conditions = parse_optional_conditions(args.conditions)
    if not before and not conditions:
        raise SystemExit("--before-baseline 與 --conditions 都是空的，"
                         "至少要有一種姿勢，例如 --before-baseline natural")
    return before, conditions


def _trip_options(args, segments: list[str]):
    """這一趟的中繼資料，選項對不上的話在開相機之前就停。"""
    try:
        return build_trip_options(
            datetime.now().isoformat(timespec="seconds"), segments,
            desk_level=getattr(args, "desk_level", None),
            lift_cm=getattr(args, "lift_cm", None),
            desk_offset_cm=getattr(args, "desk_offset_cm", None),
            task=getattr(args, "task", None),
            target_angles=parse_number_list(
                getattr(args, "target_angle_deg", None), "--target-angle-deg"),
            measured_angles=parse_number_list(
                getattr(args, "measured_angle_deg", None), "--measured-angle-deg"),
            measured_end_angles=parse_number_list(
                getattr(args, "measured_end_angle_deg", None),
                "--measured-end-angle-deg"),
            azimuth_targets=parse_number_list(
                getattr(args, "azimuth_target_deg", None), "--azimuth-target-deg"),
        )
    except ValueError as exc:
        raise SystemExit(str(exc))


# 一趟 study 裡掉線之後重新找相機的次數上限。掉兩次以上就不是偶發，硬撐只會
# 讓受試者坐著一直等。
_STUDY_REOPEN_BUDGET = 2
_REOPEN_ATTEMPTS = 5
_REOPEN_PAUSE_S = 1.0


def _reopen_camera(args, attempts: int = _REOPEN_ATTEMPTS,
                   pause_s: float = _REOPEN_PAUSE_S, calib=None):
    """相機掉線之後重新找節點、重新開。找不到回傳 None。

    不能用原本的編號重開：實機每一次掉線，節點編號都整組移位（video0 消失、
    剩 video1 與 video2）。裝置重新列舉要一兩秒，所以給幾次機會。

    給了 `calib` 就連解析度一起驗：重新列舉後的相機可能退回驅動預設的 640x480，
    拿單眼 1280x720 的內參去算只會得到一整段看似正常、尺度全錯的資料
    （2026-09-29 的老問題，開機那次有擋，重開這一條路沒有）。
    """
    for _ in range(attempts):
        time.sleep(pause_s)
        try:
            index, how = find_camera_index(args.width, args.height)
        except RuntimeError:
            continue
        cap = open_camera(index, args.width, args.height)
        if not cap.isOpened():
            cap.release()
            continue
        _discard_frames(cap, args.discard)
        if calib is not None:
            try:
                left_frame = _grab_pair(cap, args)[0]
            except CameraReadError:
                cap.release()
                continue
            problem = describe_resolution_mismatch(
                calib.image_size,
                (left_frame.shape[1], left_frame.shape[0]),
                args.vertical_split,
            )
            if problem is not None:
                _step(f"重新接上的相機解析度不符，不採用。{problem}")
                cap.release()
                continue
        args.camera = index
        _step(f"重新接上相機。{how}")
        return cap
    return None


def _study_baseline(engine, calib, cap, args, root: Path, display=NO_DISPLAY):
    """取基準，品質不好就問要不要重取。

    取完就直接往下走是不行的。2026-09-29 有一份基準略過了 27% 的幀，當時
    沒有任何提示，而後面所有量測都帶著它。
    """
    target = baseline_path(root, args.subject)
    while True:
        _prompt(f"接下來要取 {args.subject} 的個人基準，取樣 {args.seconds:.0f} 秒。\n"
                + condition_brief("upright")
                + "\n  取完之後不要起身，會直接接著量測。",
                not args.no_prompt)
        _countdown(args.countdown)
        print("\r開始取樣，請保持不動        ", flush=True)
        baseline, warnings = _collect_baseline(engine, calib, cap, args,
                                               args.seconds, display)
        if not warnings or args.no_prompt:
            break
        print()
        if not _ask_yes("這份基準有上面的問題，要重取一次嗎？"):
            break

    baseline.save(target, overwrite=True)
    print(f"基準存到 {target}", flush=True)
    return baseline


def _ask_yes(question: str) -> bool:
    """問一個是非題。讀不到 stdin 時當作「否」。

    當作否是刻意的：這個函式問的是「要不要重做」，而沒有人在旁邊的時候
    停下來等一個不會來的答案，比繼續跑完更糟。
    """
    try:
        answer = input(f"{question}[Y/n] ").strip().lower()
    except (EOFError, OSError):
        return False
    return answer in ("", "y", "yes")


def _study_one(engine, calib, cap, args, baseline, condition: str, root: Path,
               display=NO_DISPLAY, before: bool = False, trip=None,
               segment_index: int = 0, tilt_index: int | None = None):
    """量一種姿勢。

    before=True 是取基準之前的段落：不扣基準、標頭不記基準檔，開始前也要倒數。
    剛坐下的十幾秒人還在調整，這條規則與取基準的倒數相同。
    """
    trial = next_trial(root, args.subject, condition)
    csv_path, shots_dir = session_paths(root, args.subject, condition, trial)
    head = f"接下來量「{condition}」第 {trial} 次，{args.seconds_each:.0f} 秒。"
    if before:
        head += "這一段在取基準之前，不扣基準。"
    _prompt(head + "\n" + condition_brief(condition), not args.no_prompt)

    extra = None
    if trip is not None:
        measured = None
        if tilt_index is not None and not trip.measured_angles and not args.no_prompt:
            measured = _ask_tilt_angle("量角器量到的實際角度是幾度？", args)
        extra = segment_metadata(trip, condition, segment_index, tilt_index, measured)
    typing = is_typing_segment(condition, getattr(trip, "task", None))
    if typing:
        _show_typing_text()
    if before:
        _countdown(args.countdown)
        print("\r開始量測        ", flush=True)

    log = _open_log(args, baseline, csv_path, condition, trial,
                    None if before else baseline_path(root, args.subject), extra)
    snapshots = _open_snapshots(args, shots_dir)
    try:
        if typing:
            # 放在開檔訊息之後，讓受試者打字的地方緊接在這句下面。
            print("請直接在下面打字，打錯不用改，打到哪算到哪：", flush=True)
            recording = _record(engine, calib, cap, args, baseline, log, snapshots,
                                args.seconds_each, display, quiet=True)
            # 受試者打的字還留在輸入緩衝裡，不清掉的話下一個問題（例如要不要
            # 重量）會把它當成回答。先換行，免得總結接在打到一半的那行後面。
            _discard_typed_input()
            print()
            print("時間到，可以停了。", flush=True)
        elif getattr(args, "hide_live", False):
            print("量測中，請保持姿勢，時間到會提示。", flush=True)
            recording = _record(engine, calib, cap, args, baseline, log, snapshots,
                                args.seconds_each, display, quiet=True)
            print("時間到。", flush=True)
        else:
            recording = _record(engine, calib, cap, args, baseline, log, snapshots,
                                args.seconds_each, display)
        _report(recording, baseline, log)
    finally:
        log.close()
    if recording.camera_lost is not None:
        # 標頭在開始時就寫完了，掉線是事後才知道的，所以補在檔尾。`read_rows` 收
        # 任何位置的 `# key=value`；分析端靠它不拿殘缺的 upright 當同趟對照。
        with open(log.path, "a", encoding="utf-8") as handle:
            handle.write("# camera_lost=1\n")
    elif tilt_index is not None and trip is not None:
        # 側傾量完之後再讀一次量角器。60 秒內上身可能慢慢滑掉，只記開始值的話，
        # 誤差裡分不出是 θ_sym 不準還是受試者沒撐住。掉線的那一段會被重量，不問。
        end = end_angle_for(trip, tilt_index)
        if end is None and not args.no_prompt:
            end = _ask_tilt_angle("量完了，量角器現在量到的角度是幾度？", args)
        if end is not None:
            with open(log.path, "a", encoding="utf-8") as handle:
                handle.write(end_angle_footer(end))
    # replace 而不是直接指派：回傳的 Recording 一建好就是完整的，不會有
    # 「要等 _study_one 跑完才算有效」的中間狀態。
    return replace(recording, condition=condition, trial=trial,
                   before_baseline=before)


def _ask_tilt_angle(question: str, args) -> float | None:
    """側傾段落問量角器。`run-plan` 要求一定要填（`args.require_angles`）。

    2026-11-19 S01 三次側傾六段全部沒有量角器數字，誤差算不出來；當時直接按
    Enter 就能留空。必填時空白不放行，真的量不到要明講輸入 `-`，再確認一次，
    這樣留空一定是有人決定的，不是手滑。讀不到 stdin 時照舊留空，
    沒有人在旁邊時卡住等輸入比留空更糟；但 run-plan 讀不到 stdin 時直接停，
    因為那代表操作者根本沒辦法回答。
    """
    if not getattr(args, "require_angles", False):
        return _ask_angle(question + "（直接按 Enter 留空）")
    while True:
        try:
            answer = input(f"  {question}（必填；真的量不到輸入 - ）").strip()
        except (EOFError, OSError):
            # run-plan 一定有人在旁邊。讀不到鍵盤卻默默留空，正是這個模式要擋的
            # 事（GPT 審查第 4 條），所以停下來講清楚。
            raise SystemExit(_NO_KEYBOARD)
        if answer in ("-", "－"):
            if _ask_yes("  確定這一段不記量角器？誤差會算不出來。"):
                return None
            continue
        if not answer:
            print("  量角器的數字一定要填，請量完再輸入", flush=True)
            continue
        try:
            return abs(float(answer))
        except ValueError:
            print(f"  {answer!r} 不是數字，請再輸入一次", flush=True)


def _ask_angle(question: str) -> float | None:
    """問操作者一個角度。讀不到 stdin、留空或不是數字都回傳 None（標頭留空）。"""
    try:
        answer = input(f"  {question}").strip()
    except (EOFError, OSError):
        return None
    if not answer:
        return None
    try:
        value = float(answer)
    except ValueError:
        print(f"  {answer!r} 不是數字，這一段的實測角度留空", flush=True)
        return None
    return abs(value)


def _study_summary(recordings, args, baseline=None) -> None:
    """全部量完之後的總結。排版在 live_report，這裡只負責印。"""
    _print_lines(study_summary_lines(
        review(recordings, baseline), args.subject,
        [r.log_path for r in recordings],
    ))


def _print_line(text: str) -> None:
    """原地更新一行，裁到終端機的實際寬度。

    這一行放不下時終端機會折行，而 CR 只退到最後一行的開頭，畫面就變成
    一串接不起來的殘句。要按顯示寬度裁，因為中文一個字佔兩欄，用字元數裁的話
    留下來的字數雖然對，佔用的欄數是兩倍，照樣會折行。被略過的那些幀
    印的是中文原因，正好是最長、最容易超出的一種。
    """
    width = shutil.get_terminal_size(fallback=(100, 24)).columns - 1
    print("\r" + cell(truncate(text, width), width), end="", flush=True)


def _elapsed(started_at: float) -> str:
    return elapsed_text(time.perf_counter() - started_at)


def _print_lines(lines) -> None:
    for line in lines:
        print(line)


def _make_judge(args, baseline: PostureBaseline | None) -> PostureJudge | None:
    """沒有個人基準就不判定。

    門檻套在原始角度上會因人而異。2026-09-24 實測，同一個人坐正的 θ_CA 是
    +10.04°，剛好壓在 10° 的門檻上；照著判會讓這個人正常坐著就持續報警。
    這是本專案相對前作的主要修正之一，不該因為忘記給 --baseline 就默默失效。
    """
    if args.no_judge:
        return None
    if baseline is None:
        print("沒有個人基準，不做超標判定。門檻套在原始角度上會因人而異", flush=True)
        return None
    judge = PostureJudge(margin_factor=args.margin)
    judge.watch_shoulder_drop(baseline.shoulder_drop_threshold_mm)
    judge.watch_trunk(baseline.trunk_threshold_deg)
    extra = ("" if judge.shoulder_drop is None else
             f"、肩部垂直位移 {judge.shoulder_drop.threshold_deg:.0f}mm")
    if judge.trunk is not None:
        extra += f"、θ_KA {judge.trunk.threshold_deg:.0f}°（設計值，前傾後仰都算）"
    print(f"判定門檻 θ_CA {judge.theta_ca.threshold_deg:.0f}°（前傾後仰都算）、"
          f"θ_sym {judge.theta_sym.threshold_deg:.0f}°（左右都算）{extra}，"
          f"遲滯寬度 {args.margin:.1f} 倍標準誤差", flush=True)
    return judge


def _tell(message: str | None) -> None:
    """把提示印成獨立的一行。逐幀那一行會被蓋掉，提示不該跟著消失。"""
    if message is None:
        return
    _print_line("")
    print("\r" + message, flush=True)


def _refuse_to_overwrite(args) -> None:
    """在開相機之前就檢查輸出檔。

    基準要請受試者坐著不動半分鐘、量測一次要跑二十分鐘，等到做完才發現檔名撞到，
    白費的是受試者的時間。
    """
    target, flag = None, None
    if args.mode == "baseline":
        target, flag = args.out, "--out"
    elif args.mode == "live":
        target, flag = args.log, "--log"
    elif args.mode == "monitor" and args.log is not None:
        # monitor 每次重新歸零就換一個檔，實際寫的是 <stem>-1、-2…，
        # 所以要檢查的是第一份而不是 --log 本身。
        target = args.log.with_name(f"{args.log.stem}-1{args.log.suffix or '.csv'}")
        flag = "--log"
    if target is not None and Path(target).exists() and not args.overwrite:
        raise SystemExit(
            f"{target} 已經存在。用這個名字，或確定要覆蓋的話加上 --overwrite：\n"
            f"  {flag} {_next_free_name(target)}"
        )


def _next_free_name(path: Path) -> Path:
    """在原檔名後面加序號，找出一個還沒被用掉的。

    只說「換個檔名」的話，受試者得坐在那裡等人想名字。這一步本來就是為了
    不浪費他的時間才放在開相機之前，那就該把名字也一起想好。
    """
    path = Path(path)
    for n in range(2, 100):
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
        if not candidate.exists():
            return candidate
    return path.with_name(f"{path.stem}-{int(time.time())}{path.suffix}")


def _run_cohort(args) -> None:
    """把一整批記錄彙整起來。不碰相機。"""
    cohort = collect(args.paths)
    if not cohort.sessions:
        raise SystemExit(
            "沒有讀到任何可用的 CSV。確認路徑，以及那些檔案不是只有標題列"
        )
    separation = Separation(
        baseline_condition=args.baseline_condition,
        other=args.condition,
        per_subject=cohort.differences(args.baseline_condition, args.condition),
    )
    # 逐段的數字只算一次，文字報告與 CSV 共用。每個 session 的標準誤差要掃
    # 過它所有的幀，重建三次就是三倍的工，而這個指令的用途正是吃一整個資料夾。
    rows = session_rows(cohort)
    groups = _subject_groups(args)
    report = to_text(cohort, separation, rows, groups)
    print(report)

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "cohort.txt").write_text(report, encoding="utf-8")
        (args.out / "cohort.csv").write_text(to_csv(rows), encoding="utf-8")
        print(f"已寫到 {args.out / 'cohort.txt'} 與 {args.out / 'cohort.csv'}")
        _write_trip_csvs(args.out, cohort.sessions, groups)


def _subject_groups(args) -> dict | None:
    """--subject-groups 給的身高層對照表。不給就是 None（表4 全部歸未分組）。"""
    path = getattr(args, "subject_groups", None)
    if path is None:
        return None
    try:
        return read_groups(path)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"--subject-groups 讀不進來：{exc}")


def _write_trip_csvs(out: Path, summaries, groups=None) -> None:
    """配對差值、側傾、表4 比值各一份 CSV，給迴歸與報告用。沒有資料的不寫。"""
    for name, text in trip_csvs(summaries, groups).items():
        (out / name).write_text(text, encoding="utf-8")
        print(f"已寫到 {out / name}")


def _run_analyse(args) -> None:
    """分析逐幀 CSV。不碰相機，所以在哪台機器上都跑得動。"""
    summaries = []
    for path in args.csv:
        if not path.exists():
            raise SystemExit(f"{path} 不存在")
        summaries.append(analyse_session(
            path, window=args.window, margin=args.margin, force_replay=args.replay
        ))
    print(format_report(summaries))
    if args.segments > 1:
        for summary in summaries:
            table = segment_table(summary, args.segments)
            if table:
                print()
                print(f"{summary.path.name}{table}")
    # 同一趟的段落一起給時才有配對差值、側傾與表4 比值；沒有就什麼都不印。
    for title, text in trip_sections(summaries):
        print()
        print(f"── {title} ──")
        print(text)
    if getattr(args, "out", None) is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        _write_trip_csvs(args.out, summaries)


def _run_rating_pack(args) -> None:
    """匯出給外部評分者的盲評包。不碰相機。"""
    raters = [r.strip() for r in args.raters.split(",") if r.strip()]
    try:
        print(export_pack(args.paths, args.out, raters, args.config, args.seed,
                          args.per_segment, args.allow_overlay, args.key_out))
    except (RatingPackError, RatingConfigError) as exc:
        raise SystemExit(str(exc))


def _run_ratings(args) -> None:
    """讀回評分，算評分者信度與有效警報。沒有評分檔時照實說，不當掉。"""
    try:
        print(analyse_ratings(args.key, args.ratings, args.config, args.sessions,
                              args.out, args.bootstrap))
    except (RatingImportError, RatingConfigError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))


_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def _profile_frames(source: Path, sample_fps: float):
    """產生 (秒, BGR 畫面) 與總長度。影片依 sample_fps 抽樣；資料夾依檔名排序，
    每張當作 1/sample_fps 秒。讀不出來的畫面略過，不當掉。
    """
    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
        if not files:
            raise SystemExit(f"{source} 裡沒有 jpg 或 png 圖片")

        def images():
            for i, path in enumerate(files):
                frame = cv2.imread(str(path))
                if frame is not None:
                    yield i / sample_fps, frame

        return images(), len(files) / sample_fps
    if not source.is_file():
        raise SystemExit(f"找不到 {source}")
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"開不了影片 {source}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    step = max(1, int(round(video_fps / sample_fps)))

    def video():
        index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    return
                if index % step == 0:
                    yield index / video_fps, frame
                index += 1
        finally:
            cap.release()

    return video(), (total / video_fps if total and total > 0 else None)


def _run_profile_ref(args) -> None:
    """手機在正側 90° 拍的影片或圖片，算每個時間窗的 θ_CA 參考值。不碰雙目相機。

    要載入姿態模型，所以在 Jetson 上跑。可選的 --stereo-csv 把雙目同一段的 3D θ_CA
    拿來相減。
    """
    try:
        up = parse_plumb(args.plumb)
    except ValueError as exc:
        raise SystemExit(str(exc))
    facing = {"right": 1.0, "left": -1.0}.get(args.facing)
    frames, duration = _profile_frames(args.source, args.fps)
    try:
        windows = parse_windows(args.windows, duration)
    except ValueError as exc:
        raise SystemExit(str(exc))
    stereo_paths = {}
    for item in args.stereo_csv:
        name, sep, path = item.partition("=")
        if not sep or name not in {w[0] for w in windows}:
            raise SystemExit(f"--stereo-csv 要寫成 時間窗名稱=CSV路徑，名稱要在 --windows 裡，收到 {item!r}")
        stereo_paths[name] = Path(path)

    engine = _build_engine(args)
    samples = []
    for t_s, frame in frames:
        person = pick_person(engine.infer(frame), args.side, args.min_confidence)
        samples.append(sample_from_person(t_s, person, args.side))
        if len(samples) % 50 == 0:
            print(f"  已處理 {len(samples)} 幀（{t_s:.0f} 秒）", flush=True)

    summaries = [summarize_window(name, samples, start, end, up, facing)
                 for name, start, end in windows]
    comparisons = [compare_profile(s, stereo_segment(stereo_paths[s.name]))
                   for s in summaries if s.name in stereo_paths]
    lines = report_lines(summaries, comparisons)
    print("\n".join(lines))

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "profile_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        by_window = {s.name: s for s in summaries}
        with open(args.out / "profile_frames.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["t_s", "window", "ear_x", "ear_y", "shoulder_x", "shoulder_y",
                             "nose_x", "nose_y", "angle_deg", "reason"])
            for s in samples:
                owner = next((n for n, a, b in windows if a <= s.t_s < b), "")
                sign = by_window[owner].facing if owner else None
                angle = (profile_angle_deg(s.ear, s.shoulder, sign, up)
                         if sign is not None else None)

                def cell_(point, i):
                    return "" if point is None else f"{point[i]:.2f}"

                writer.writerow([f"{s.t_s:.3f}", owner,
                                 cell_(s.ear, 0), cell_(s.ear, 1),
                                 cell_(s.shoulder, 0), cell_(s.shoulder, 1),
                                 cell_(s.nose, 0), cell_(s.nose, 1),
                                 "" if angle is None else f"{angle:.3f}", s.reason])
        print(f"\n逐幀與報告寫到 {args.out}")


# ---- run-plan：引導模式 ------------------------------------------------------

# run-plan 自己的選項，不抄進每一趟的 study 參數。其餘（相機、模型、視窗、
# 受試者、輸出資料夾）原樣抄過去，所以 --display、--camera 這些照常生效。
_PLAN_ONLY = frozenset({"mode", "plan", "dry_run"})

# run-plan 開始前自動預檢時試拍的幀數。
_PLAN_PREFLIGHT_FRAMES = 5


class _Tee:
    """stdout 同時寫進紀錄檔，遠端的人用 tail -f 就能旁觀組員的畫面。

    故意沒有 fileno：input() 發現 stdout 不是真的終端機，會把提示文字寫進
    sys.stdout（也就是寫進紀錄），遠端才看得到「現在在問什麼」。有 fileno 的話
    提示由 readline 直接寫到終端機，紀錄裡會缺。代價是方向鍵編輯行不能用，
    退格鍵照常。
    """

    def __init__(self, stream, handle):
        self._stream, self._handle = stream, handle

    def write(self, text):
        self._stream.write(text)
        self._handle.write(text)
        self._handle.flush()
        return len(text)

    def flush(self):
        self._stream.flush()
        self._handle.flush()

    def isatty(self):
        return False

    def fileno(self):
        raise io.UnsupportedOperation("fileno")

    def __getattr__(self, name):
        return getattr(self._stream, name)


@contextlib.contextmanager
def _tee_stdout(path: Path):
    """把這段期間的畫面也記到 path（附加，同一個代號重跑不會蓋掉）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        stamp = datetime.now().isoformat(timespec="seconds")
        handle.write(chr(10) + "===== " + stamp + " =====" + chr(10))
        real = sys.stdout
        sys.stdout = _Tee(real, handle)
        try:
            yield
        finally:
            sys.stdout = real


def _plan_preflight(engine, calib, cap, args) -> None:
    """量之前先把上機前檢查做完。沒過就讓操作者調整後重測，不讓人在壞的架設上量。"""
    base = git_items(*_git_state()) + [PreflightItem(
        True, "標定檔", f"{args.calibration}（基線 {calib.baseline_mm:.2f} mm）")]
    while True:
        items = base + _probe_items(engine, calib, cap, args,
                                    _PLAN_PREFLIGHT_FRAMES, warmup=False)
        _print_lines(verdict_lines(items))
        if preflight_clear(items):
            return
        try:
            answer = input("照上面調整，好了按 Enter 重新檢查；要離開輸入 q：")
        except (EOFError, OSError):
            raise SystemExit(_NO_KEYBOARD)
        if answer.strip().lower() == "q":
            raise SystemExit("已離開，沒有量任何資料。")

_NO_KEYBOARD = ("讀不到鍵盤輸入，桌面減手肘或量角器沒辦法填。"
                "請直接在實機的終端機（或 ssh 的互動視窗）重跑 run-plan，"
                "不要用管線或背景執行。已經量完的段落都存好了")

# 受試者代號只收英數、底線、連字號。代號會變成資料夾名稱，「..」或「.」沒有斜線
# 也會把資料寫到 data/sessions 的上一層或根目錄（GPT 審查第 2 條）。
_SUBJECT_PATTERN = re.compile(r"[A-Za-z0-9_-]+")


def _plan_trip_args(parser, args, plan, trip, offset):
    """把計畫的一趟組成 study 的 args。用同一個 parser，預設值只有一處。"""
    ns = parser.parse_args(["study"] + trip_argv(plan, trip, args.subject, offset))
    for key, value in vars(args).items():
        if key not in _PLAN_ONLY:
            setattr(ns, key, value)
    ns.require_angles = True
    # 受試者可能看得到螢幕（自然坐就在這台上打字），逐幀那一行有角度，
    # 看得到就會跟著調姿勢。每段量完的總結照印，那時候已經量完了。
    ns.hide_live = True
    ns.no_prompt = False
    ns.overwrite = False
    return ns


def _plan_header(plan, trip, n: int) -> list[str]:
    bar = "=" * 60
    segments = list(trip.before) + (["取基準"] if trip.conditions else []) + list(trip.conditions)
    lines = ["", bar, f"  第 {n}/{len(plan.trips)} 趟：{trip.title}",
             f"  約 {estimated_minutes(plan, trip):.0f} 分鐘（不含重量）",
             "  段落：" + " → ".join(segments), bar]
    if trip.setup:
        lines += ["", "【操作者先做】", trip.setup]
    if trip.subject_does:
        lines += ["", "【告訴受試者】", trip.subject_does]
    if trip.tilts:
        lines += ["", "【側傾】每一段開始前與量完都要用量角器量，程式會問，一定要輸入數字。"]
    return lines


def _ask_desk_offset() -> float | None:
    """問桌面減手肘。必填，問完再用白話覆述一次讓操作者確認正負號。

    2026-11-19 兩趟的 desk_offset_cm 都記成 10，看起來是沒有重量；
    覆述「桌面比手肘高 10 公分，對嗎？」比只看到一個數字更容易發現寫錯。
    """
    while True:
        try:
            text = input("桌面高減手肘高是幾公分？桌面比較高填正數，比較低填負數：")
        except (EOFError, OSError):
            raise SystemExit(_NO_KEYBOARD)
        value = parse_offset(text)
        if value is None:
            print("  要填數字，例如 5 或 -3。拿量尺量：桌面高度減坐著時手肘的高度", flush=True)
            continue
        if _ask_yes(f"  {describe_offset(value)}，對嗎？"):
            return value


def _plan_checker(args):
    """每段量完的當場檢查。回傳給 `_study_trip` 的 check。"""
    redo_counts: dict = {}

    def check(recording, recordings, baseline, first_upright):
        reviews = review(recordings, baseline)
        segment = reviews[-1]
        result = check_segment(segment, first_upright_after_baseline=first_upright)
        if result.ok and first_upright:
            _warn_turned_before(recordings[:-1], reviews[:-1])
        if result.ok:
            print("  當場檢查：通過", flush=True)
            return "keep"
        print("  當場檢查：沒過", flush=True)
        for problem in result.problems:
            print(f"   - {problem}", flush=True)
        # 重量時舊檔搬走，編號會沿用，所以 (條件, 編號) 認得出是同一段。
        key = (recording.condition, recording.trial)
        if redo_counts.get(key, 0) >= MAX_REDO_PER_SEGMENT:
            print("  這一段已經重量過，不再重量，記下來交給事後判讀", flush=True)
            return "keep"
        question = ("要重取基準並重量這一段嗎？（受試者不用起身）"
                    if result.retake_baseline else "要重量這一段嗎？")
        if not _ask_yes(question):
            print("  不重量，記下來交給事後判讀", flush=True)
            return "keep"
        redo_counts[key] = redo_counts.get(key, 0) + 1
        if recording.log_path is not None:
            moved = discard_segment(recording.log_path, args.out_dir, args.subject)
            print(f"  原本那份搬到 {moved}，不會被算進分析", flush=True)
        return "rebaseline" if result.retake_baseline else "redo"

    return check


def _warn_turned_before(recordings, reviews) -> None:
    """自然坐量的當下還沒有同趟端正坐可比，量完第一段端正坐才看得出它有沒有轉身。

    看出來了也不當場重量：自然坐要在「請坐正」之前量，已經念過坐正就不是
    平常的坐法了。只能提醒操作者，要補就讓受試者休息、起身後整趟重跑
    （GPT 審查第 3 條，只採提醒）。
    """
    for recording, segment in zip(recordings, reviews):
        if recording.before_baseline and segment.turned_too_far:
            print(f"  注意：{segment.condition} #{segment.trial} 的方位角和這段端正坐差 "
                  f"{segment.turned_deg:+.0f}°，分析時會被判轉身作廢。自然坐不能在"
                  "請坐正之後補量；要補的話，這一位全部量完、起身休息後，"
                  "再單獨重跑一次趟 1。", flush=True)


def _plan_discard_lost(args):
    """相機掉線的殘段搬到作廢區，重量那段沿用同一個編號。"""
    def discard(recording):
        if recording.log_path is not None and Path(recording.log_path).exists():
            moved = discard_segment(recording.log_path, args.out_dir, args.subject)
            print(f"  相機掉線，殘缺的那份搬到 {moved}，這一段重量", flush=True)
    return discard


def _plan_analyse(parser, args, paths) -> None:
    """一位受試者跑完就跑 analyse，輸出另存一份給 AI 判讀。"""
    paths = [p for p in paths if p is not None and Path(p).exists()]
    if not paths:
        print("沒有可分析的段落。", flush=True)
        return
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            _run_analyse(parser.parse_args(["analyse"] + [str(p) for p in paths]))
    except (Exception, SystemExit) as exc:  # 分析出錯不該讓已量好的資料像是白量
        print(buffer.getvalue())
        print(f"analyse 出錯：{exc}。資料都已存好，把這段畫面拍照傳給使用者。", flush=True)
        return
    text = buffer.getvalue()
    print(text)
    out = Path(args.out_dir) / args.subject / (
        f"analyse-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"分析結果另存到 {out}", flush=True)


def _ask_no(question: str) -> bool:
    """問一個是非題，預設「否」。讀不到 stdin 時當作否。"""
    try:
        answer = input(f"{question}[y/N] ").strip().lower()
    except (EOFError, OSError):
        return False
    return answer in ("y", "yes")


def _run_plan(args) -> None:
    """照計畫檔連跑一位受試者的所有趟次，每段量完當場檢查。

    一行指令做完全部：預檢、桌高、每一趟、分析，畫面同時記到 data/logs/。
    """
    if args.dry_run:
        _run_plan_body(args)
        return
    with _tee_stdout(Path(args.out_dir).parent / "logs" / f"{args.subject}.log"):
        _run_plan_body(args)


def _run_plan_body(args) -> None:
    try:
        plan = load_plan(args.plan)
    except ValueError as exc:
        raise SystemExit(str(exc))
    args.subject = (args.subject or "").strip()
    if not _SUBJECT_PATTERN.fullmatch(args.subject):
        raise SystemExit("--subject 要是一個代號，例如 S02，只能用英文、數字、底線、連字號")
    parser = build_parser()
    # 每一趟先組一次參數，選項不對在開相機之前就停。
    for trip in plan.trips:
        ns = _plan_trip_args(parser, args, plan, trip, 0.0)
        before, conditions = _study_plan(ns)
        _trip_options(ns, before + conditions)

    total = sum(estimated_minutes(plan, t) for t in plan.trips)
    print(f"計畫：{plan.name}　受試者：{args.subject}　共 {len(plan.trips)} 趟，"
          f"約 {total:.0f} 分鐘（不含重量與換人）", flush=True)
    for note in plan.notes:
        print(f"  ・{note}", flush=True)
    if args.dry_run:
        for n, trip in enumerate(plan.trips, start=1):
            print(f"\n第 {n} 趟 {trip.title}{'（可跳過）' if trip.optional else ''}")
            print("  等同指令：python posture.py study "
                  + " ".join(trip_argv(plan, trip, args.subject, None)))
        print("\n--dry-run 只檢查計畫檔，沒有開相機。", flush=True)
        return

    calib, engine, cap = _prepare(args)
    display = NO_DISPLAY
    logs = []
    done = []
    try:
        display = _open_display(args)
        _plan_preflight(engine, calib, cap, args)
        # 量桌面減手肘要受試者坐著，所以排在預檢之後。
        offset = None
        if any(t.desk_level is not None for t in plan.trips):
            offset = _ask_desk_offset()
        for n, trip in enumerate(plan.trips, start=1):
            ns = _plan_trip_args(parser, args, plan, trip, offset)
            _print_lines(_plan_header(plan, trip, n))
            if trip.optional:
                note = f"（{trip.optional_note}）" if trip.optional_note else ""
                if not _ask_no(f"這一趟可以跳過{note}。要做嗎？"):
                    print("跳過這一趟。", flush=True)
                    continue
            try:
                recordings, baseline, cap = _study_trip(
                    engine, calib, cap, ns, display, check=_plan_checker(ns),
                    on_camera_lost=_plan_discard_lost(ns))
            except Exception:
                # 換過的相機 _study_trip 已經放掉；這裡的 cap 是交進去的那個，
                # 由下面的 finally 放（重開過的話它早已放過，再放一次無妨）。
                print("\n程式報錯了。把整個畫面拍照傳給使用者，不要自己改程式。"
                      "已經量完的段落都存好了。", flush=True)
                raise
            # 掉線重開會換節點編號，下一趟要用新的。
            args.camera = ns.camera
            _study_summary(recordings, ns, baseline)
            logs += [r.log_path for r, seg in zip(recordings, review(recordings, baseline))
                     if seg.usable_data]
            done.append(trip.title)
            if cap is None:
                print("相機找不到了，今天先停在這裡。", flush=True)
                break
    finally:
        display.close()
        if cap is not None:
            cap.release()

    print("\n" + "=" * 60, flush=True)
    print(f"{args.subject} 量完了：{'、'.join(done) or '沒有完成任何一趟'}", flush=True)
    _plan_analyse(parser, args, logs)
    print("\n接下來：", flush=True)
    print(f"  下一位受試者：python posture.py run-plan {args.plan} --subject <下一個代號>")
    print("  今天全部量完就可以收工。AI 會自己用 ssh 拉資料判讀，不用傳任何檔案。")
    print("  有報錯或卡住，把畫面拍照傳給使用者。", flush=True)


# ---- preflight：上機前檢查 ---------------------------------------------------

def _git_state() -> tuple[str | None, str | None, bool | None]:
    """(目前 commit, GitHub 上的 commit, 有沒有未 commit 的修改)。拿不到就是 None。"""
    here = Path(__file__).resolve().parent

    def git(*argv, timeout=10):
        try:
            result = subprocess.run(["git", *argv], cwd=here, capture_output=True,
                                    text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    local = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    remote_line = git("ls-remote", "origin", "HEAD", timeout=15)
    remote = remote_line.split()[0] if remote_line else None
    return local, remote, None if status is None else bool(status)


def _probe_items(engine, calib, cap, args, frames: int, warmup: bool) -> list:
    """解析度加試拍幾幀，整理成預檢的項目。`preflight` 與 `run-plan` 共用。

    相機讀不到和「沒偵測到人」要分開講，該做的事不一樣（GPT 審查第 7 條）。
    """
    items, probes = [], []
    try:
        _discard_frames(cap, args.discard)
        left_frame = _grab_pair(cap, args)[0]
        problem = describe_resolution_mismatch(
            calib.image_size, (left_frame.shape[1], left_frame.shape[0]),
            args.vertical_split)
        items.append(PreflightItem(
            problem is None, "解析度",
            problem or f"單眼 {left_frame.shape[1]}x{left_frame.shape[0]}，與標定一致",
            "拔插相機 USB 再跑一次"))
        if problem is None:
            if warmup:
                _step(f"載入模型（{args.precision} / {args.device}）")
                engine.warmup(left_frame)
            _step(f"試拍 {frames} 幀")
            for _ in range(frames):
                try:
                    measurement = _measure_once(engine, calib, cap, args)[0]
                except CameraReadError:
                    raise
                except RuntimeError as exc:  # 配不到人
                    probes.append(FrameProbe(error=str(exc)))
                    continue
                probes.append(FrameProbe(
                    shared_count=measurement.shared_count,
                    angle_disparity_px=measurement.angle_max_vertical_disparity_px,
                    depth_mm=measurement.reference_depth_mm,
                    unusable=unusable_reason(measurement)))
            items += probe_items(probes)
    except CameraReadError as exc:
        items.append(PreflightItem(False, "相機", f"讀不到畫面：{exc}",
                                   "拔插相機 USB；還是不行就跑 dmesg | tail -40"))
    return items


def _run_preflight(args) -> None:
    """一個指令做完自測手冊「開始前」那幾項。沒過的話結束碼是 1。"""
    items = git_items(*_git_state())
    calib = StereoCalibrationResult.load(args.calibration)
    items.append(PreflightItem(True, "標定檔",
                               f"{args.calibration}（基線 {calib.baseline_mm:.2f} mm）"))
    engine = _build_engine(args)
    try:
        cap = _open_selected_camera(args, calib)
    except RuntimeError as exc:
        items.append(PreflightItem(False, "相機", str(exc),
                                   "確認 USB 有接好；還是不行就跑 dmesg | tail -40"))
        _print_lines(verdict_lines(items))
        raise SystemExit(1)
    try:
        items += _probe_items(engine, calib, cap, args, args.frames, warmup=True)
    finally:
        cap.release()
    _print_lines(verdict_lines(items))
    if not preflight_clear(items):
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    """建整套 CLI。

    與 `main` 分開是為了測得到。這一百九十行先前整個藏在 `main` 裡，而每一個
    子指令收哪些選項是會出錯的事,2026-10-27 之前 scale 收下了 --display、
    --precision 這些它根本不看的選項，使用者加了只會以為沒生效。
    """
    parser = argparse.ArgumentParser(description="端到端坐姿量測")
    sub = parser.add_subparsers(dest="mode", required=True)

    ch = sub.add_parser("cohort", help="把一整批逐幀記錄彙整成報告要的表，不需要相機")
    ch.add_argument("paths", type=Path, nargs="+",
                    help="資料夾或 CSV。資料夾會遞迴找所有 .csv")
    ch.add_argument("--baseline-condition", default="upright",
                    help="當作基準的姿勢名稱，差距是相對它算的")
    ch.add_argument("--condition", default="forward",
                    help="要與基準比較的姿勢名稱")
    ch.add_argument("--out", type=Path, default=None,
                    help="把報告與 CSV 寫到這個資料夾。不給就只印出來。"
                         "要放進報告的表格從 CSV 匯入")
    ch.add_argument("--subject-groups", type=Path, default=None,
                    help="受試者分組表（CSV，欄位 subject,group），表4 依身高層"
                         "分組報告時用。不給就全部歸未分組")

    # analyse 不開相機，所以那些硬體參數對它沒有意義，單獨建 parser。
    ap = sub.add_parser("analyse", help="分析 live 留下的逐幀 CSV，不需要相機")
    ap.add_argument("csv", type=Path, nargs="+",
                    help="一份或多份逐幀記錄。給兩份以上會多印一段對照")
    ap.add_argument("--window", type=int, default=30,
                    help="重播判定時用的移動平均視窗，要與當時的 live 一致")
    ap.add_argument("--margin", type=float, default=1.0,
                    help="重播判定時的遲滯寬度。調這個可以看誤報率有多敏感")
    ap.add_argument("--replay", action="store_true",
                    help="即使 CSV 有 posture 欄也重新判一次，用來換參數比較")
    ap.add_argument("--segments", type=int, default=0,
                    help="把整段切成幾塊各自印統計，用來看漂移。0 是不印")
    ap.add_argument("--out", type=Path, default=None,
                    help="把配對差值、側傾、表4 比值的 CSV 寫到這個資料夾。"
                         "不給就只印出來")

    # 外部評分者盲評（說明書表3）。兩個都不開相機。
    rp = sub.add_parser("rating-pack", help="匯出給外部評分者的盲評包，不需要相機")
    rp.add_argument("paths", type=Path, nargs="+",
                    help="受試者資料夾（data/sessions/<subject>）或整個 data/sessions")
    rp.add_argument("--out", type=Path, required=True,
                    help="盲評包的資料夾，要不存在或是空的。底下每位評分者一個資料夾，"
                         "整包交出去也不帶 key")
    rp.add_argument("--key-out", type=Path, default=None,
                    help="key 檔的位置，不能在 --out 裡面。預設是 --out 旁邊的 "
                         "<資料夾名>-key/key.csv")
    rp.add_argument("--raters", required=True,
                    help="評分者代號，逗號分隔，例如 R1,R2。每人一份，順序各自隨機")
    rp.add_argument("--config", type=Path, default=None,
                    help="評分設定檔（JSON）。給了評分表才有評分欄，見文件 10")
    rp.add_argument("--per-segment", type=int, default=0,
                    help="每段最多取幾張快照（平均挑）。0 是全部")
    rp.add_argument("--seed", type=int, default=None,
                    help="亂數種子。不給就用系統亂數，每次匯出都不同")
    rp.add_argument("--allow-overlay", action="store_true",
                    help="沒有原圖（量測時沒加 --snapshot-raw）時改用疊加骨架的快照。"
                         "評分者會看到系統的骨架，盲評的前提打折")

    rt = sub.add_parser("ratings", help="讀回評分，算評分者信度與有效警報，不需要相機")
    rt.add_argument("--key", type=Path, required=True,
                    help="rating-pack 產生的 key 檔（預設在盲評包旁邊的 <資料夾名>-key/key.csv）")
    rt.add_argument("--ratings", type=Path, nargs="*", default=[],
                    help="評分者交回的 CSV 或放它們的資料夾。沒有的話印「沒有評分檔」")
    rt.add_argument("--config", type=Path, required=True,
                    help="匯出時用的同一份評分設定檔")
    rt.add_argument("--sessions", type=Path, default=None,
                    help="逐幀 CSV 的根目錄（data/sessions）。不給就用 key 檔記的路徑")
    rt.add_argument("--out", type=Path, default=None,
                    help="把報告與逐段 CSV 寫到這個資料夾")
    rt.add_argument("--bootstrap", type=int, default=2000,
                    help="信賴區間的 bootstrap 次數")

    # 正側面參考：手機的影片或圖片，要載入姿態模型但不開雙目相機。
    pr = sub.add_parser("profile-ref",
                        help="手機正側 90° 的影片或圖片，算 θ_CA 的獨立參考值，不需要雙目相機")
    pr.add_argument("source", type=Path,
                    help="影片檔，或放圖片（jpg、png）的資料夾，圖片依檔名排序")
    pr.add_argument("--side", choices=["right", "left"], default="right",
                    help="近側。要與雙目那邊用的一側一致")
    pr.add_argument("--windows", default=None,
                    help="時間窗，名稱=起:訖（秒），逗號分隔，例如 upright-8=5:35,"
                         "head-forward-2=50:80。不給就是整段一個時間窗 all")
    pr.add_argument("--stereo-csv", nargs="*", default=[], metavar="NAME=CSV",
                    help="時間窗名稱=雙目那一段的逐幀 CSV，給了就印 3D 減參考")
    pr.add_argument("--plumb", default=None,
                    help="畫面裡鉛垂線的兩個點，x1,y1,x2,y2（上面的點在前，像素）。"
                         "不給就假設手機沒有翻滾")
    pr.add_argument("--facing", choices=["right", "left"], default=None,
                    help="受試者在畫面裡面向哪一邊。不給就由鼻子相對耳朵判斷")
    pr.add_argument("--fps", type=float, default=5.0,
                    help="每秒取樣幾幀。影片會依它抽樣；圖片資料夾每張當作 1/fps 秒")
    pr.add_argument("--out", type=Path, default=None,
                    help="把報告與逐幀 CSV 寫到這個資料夾")
    pr.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    pr.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    pr.add_argument("--checkpoint", type=Path,
                    default=Path("data/pose_models/checkpoint_iter_370000.pth"))
    pr.add_argument("--engine-cache", type=Path,
                    default=Path("data/pose_models/lightweight_openpose_fp16.pth"))
    pr.add_argument("--repo-dir", type=Path,
                    default=Path("third_party/lightweight-human-pose-estimation.pytorch"))
    pr.add_argument("--min-confidence", type=float, default=0.0,
                    help="低於這個信心度的耳或肩不用")
    pr.add_argument("--input-height", type=int, default=256,
                    help="網路輸入高度。要與雙目量測時一致，才不會把模型設定的差異當成誤差")
    pr.add_argument("--no-subpixel", action="store_true")

    modes = (
        ("once", "量測一次並印出完整診斷"),
        ("live", "持續量測，印移動平均"),
        ("baseline", "請受試者坐正保持不動，取個人基準（θ_offset）"),
        ("study", "一個行程跑完整套實驗流程：取基準，接著逐一量各種姿勢"),
        ("monitor", "一個視窗跑到底：按 c 當場取基準歸零，按 q 離開"),
        ("scale", "拿標定板驗證三角測量的尺度，不需要姿態模型"),
        ("run-plan", "引導模式：照計畫檔連跑一位受試者的所有趟次，每段量完當場檢查"),
        ("preflight", "上機前檢查：程式版本、相機、解析度、關鍵點、垂直視差、距離"),
    )
    for name, help_text in modes:
        p = sub.add_parser(name, help=help_text)
        # 相機：每一種模式都要，scale 也是。
        p.add_argument("--calibration", type=Path,
                       default=Path("data/calibration_output/stereo.npz"))
        p.add_argument("--camera", default="auto",
                       help="相機 index，或 auto 自動挑。節點編號會因為重新插拔或"
                            "重開機而移位，auto 會逐一試到讀得出畫面為止")
        p.add_argument("--width", type=int, default=None,
                       help="合併畫面的寬度。不給的話照標定檔要求，"
                            "因為內參綁在解析度上，不符的話深度沒有意義")
        p.add_argument("--height", type=int, default=None,
                       help="合併畫面的高度。同上")
        p.add_argument("--vertical-split", action="store_true")
        p.add_argument("--swap-lr", action="store_true")
        p.add_argument("--discard", type=int, default=5,
                       help="開始量測前先丟掉幾張，讓自動曝光穩定")
        p.add_argument("--save-frames", type=Path, default=None,
                       help="把左右兩眼的畫面存成 png，偵測到的關鍵點疊上去。"
                            "診斷訊息說不出相機到底看到什麼，這個看得出來。"
                            "baseline 與 live 存的是第一張被略過的畫面")

        if name == "scale":
            # 板子的規格要與標定時下的那組一致，不然量出來的「偏差」只是
            # 兩個不同的板子。
            p.add_argument("--squares-x", type=int, default=10,
                           help="ChArUco 板橫向方格數")
            p.add_argument("--squares-y", type=int, default=7,
                           help="ChArUco 板縱向方格數")
            p.add_argument("--square-size-mm", type=float, default=25.0,
                           help="板子方格的實際邊長。拿尺量一次,這個數字"
                                "就是比較的基準，填錯等於驗了個寂寞")
            p.add_argument("--marker-size-mm", type=float, default=18.0)
            p.add_argument("--dictionary", type=str, default="DICT_5X5_100")
            p.add_argument("--legacy-pattern", action="store_true")
            p.add_argument("--frames", type=int, default=10,
                           help="取幾幀。單幀的角點偵測本身有雜訊，而要看的是"
                                "系統性的偏差，多取幾幀比較穩")
            p.add_argument("--tolerance", type=float, default=0.02,
                           help="差距在這個比例以內就算過")
            # scale 到此為止。它不載入姿態模型、不開視窗，那幾十個選項列在
            # --help 裡只會讓人以為 --display 有用。
            continue

        # 姿態模型。
        p.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
        p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
        p.add_argument("--checkpoint", type=Path,
                       default=Path("data/pose_models/checkpoint_iter_370000.pth"))
        p.add_argument("--engine-cache", type=Path,
                       default=Path("data/pose_models/lightweight_openpose_fp16.pth"))
        p.add_argument("--repo-dir", type=Path,
                       default=Path("third_party/lightweight-human-pose-estimation.pytorch"))
        p.add_argument("--min-confidence", type=float, default=0.0,
                       help="低於這個信心度的關鍵點不參與三角測量")
        p.add_argument("--input-height", type=int, default=256,
                       help="網路輸入高度。調高會直接降低熱圖的量化誤差（深度精度的"
                            "主要瓶頸），代價是推論變慢；384或512值得一試")
        p.add_argument("--no-subpixel", action="store_true",
                       help="關掉熱圖峰值的次像素精修，用來量化它的影響")

        if name == "preflight":
            p.add_argument("--frames", type=int, default=10,
                           help="試拍幾幀。受試者要坐在平常的位置")
            # preflight 不開視窗、不錄資料，下面那些選項對它沒有意義。
            continue

        # 視窗。
        p.add_argument("--display", action="store_true",
                       help="開一個視窗，畫面疊上骨架與狀態面板。給操作者架設"
                            "與示範用，螢幕要放在受試者視野之外。視窗有焦點時"
                            "按 q 結束這一段，終端機的 Ctrl-C 照常")
        p.add_argument("--display-width", type=int, default=1280,
                       help="視窗寬度。低於 640 會拉回 640，再小面板就看不清")
        p.add_argument("--display-eyes", choices=["both", "left"], default="left",
                       help="left 只畫左眼，受試者佔滿整個寬度；both 左右眼並排，"
                            "配對錯了看得出來，但兩張幾乎一樣，各只剩一半寬")
        p.add_argument("--display-lang", choices=["en", "zh"], default="en",
                       help="面板的語言。en 不需要中日韓字型；zh 要有字型，"
                            "沒有的話照樣退回英文")
        p.add_argument("--font", type=Path, default=None,
                       help="--display-lang zh 時用的中日韓字型。不給的話自動找，"
                            "找不到就改用英文，量測照常")
        p.add_argument("--window", type=int, default=30,
                       help="live 模式的平均視窗幀數。單幀誤差與判定門檻同量級，"
                            "平均N幀把偵測雜訊降到1/√N；30幀約5秒")

        if name == "once":
            p.add_argument("--all-keypoints", action="store_true",
                           help="印出全部18點，不只角度用到的那幾個")
        if name in ("live", "baseline", "study", "monitor"):
            p.add_argument("--subject", default="受試者",
                           help="受試者代號，寫進基準檔與 CSV")
        if name == "run-plan":
            p.add_argument("plan", type=Path, nargs="?",
                           default=Path("plans/selftest.json"),
                           help="計畫檔（JSON）。不給就用 plans/selftest.json")
            p.add_argument("--subject", required=True,
                           help="受試者代號，例如 S02。一人一碼，不要重複用")
            p.add_argument("--out-dir", type=Path, default=Path("data/sessions"),
                           help="資料寫到 <out-dir>/<subject>/ 底下；作廢的段落搬到"
                                " <out-dir>-discarded/<subject>/")
            p.add_argument("--dry-run", action="store_true",
                           help="只檢查計畫檔並印出每一趟等同的 study 指令，不開相機")
        if name in ("live", "monitor"):
            p.add_argument("--baseline", type=Path, default=None,
                           help="個人基準檔，由 baseline 子指令產生。"
                                "指定之後印出的是相對這個人端正坐姿的偏移量")
            p.add_argument("--log", type=Path, default=None,
                           help="把每一幀寫成 CSV，含被略過的幀。"
                                "事後分析與 Kinovea 對標都需要這份原始資料")
            p.add_argument("--snapshots", type=Path, default=None,
                           help="每隔一段時間存一張左眼畫面到這個資料夾。"
                                "姿勢事後查證不了，沒有畫面的話量到的數字"
                                "既不能當誤報率、也不能當正確偵測的證據")
            p.add_argument("--condition", default="",
                           help="姿勢條件的名稱，例如 upright 或 forward。"
                                "寫進 CSV 標頭，彙整時靠它分組")
        if name == "live":
            p.add_argument("--seconds", type=float, default=None,
                           help="量測長度。不給就跑到 Ctrl-C 為止。"
                                "固定秒數的段落比較不會發生受試者忘記維持姿勢")
            p.add_argument("--trial", type=int, default=None,
                           help="同一個人同一種姿勢的第幾次，寫進 CSV 標頭")
        if name == "monitor":
            p.add_argument("--baseline-seconds", type=float, default=20.0,
                           help="按 c 之後取樣多久。一幀的零點沒有用，"
                                "單幀散佈與判定門檻是同一個量級")
            p.add_argument("--countdown", type=float, default=5.0,
                           help="按 c 之後先倒數幾秒讓受試者坐定。"
                                "這段時間的幀不進基準")
            p.add_argument("--baseline-dir", type=Path,
                           default=Path("data/baselines"),
                           help="當場取的基準存到這裡，檔名帶時間")
        if name in ("live", "study", "monitor"):
            p.add_argument("--margin", type=float, default=1.0,
                           help="遲滯寬度，單位是標準誤差的倍數。調高會減少誤報但"
                                "反應變慢、不動作的區間變寬")
            p.add_argument("--no-judge", action="store_true",
                           help="只印角度不做超標判定，用來收集原始資料")
            p.add_argument("--snapshot-every", type=float, default=10.0,
                           help="存畫面的間隔秒數。study 一定會存，live 要搭配 --snapshots")
            p.add_argument("--snapshot-raw", action="store_true",
                           help="快照另存一份不疊骨架與點名的原圖（快照資料夾的 raw/），"
                                "給 rating-pack 做外部評分者的盲評包。預設不存")
        if name in ("live", "baseline", "study", "monitor"):
            p.add_argument("--overwrite", action="store_true",
                           help="允許覆蓋既有的基準檔或記錄檔")
        if name == "baseline":
            p.add_argument("--out", type=Path, default=None,
                           help="預設是 data/baselines/<subject>.json")
            p.add_argument("--seconds", type=float, default=30.0,
                           help="取樣長度。實機約 5fps，30 秒約 150 幀。"
                                "不夠準的話程式會算出該取樣幾秒")
            p.add_argument("--countdown", type=int, default=3,
                           help="開始前的倒數秒數。剛坐下的十幾秒人還在調整，"
                                "那段的散佈明顯比後面大，會被平均進基準裡")
        if name == "study":
            p.add_argument("--conditions", default="upright,forward",
                           help="要量的姿勢，逗號分隔。名稱可以自己取，"
                                "程式不限定種類，只把它寫進 CSV 與檔名。"
                                "有給 --before-baseline 時可以是空字串，"
                                "那一趟就不取基準（表4 那一趟）")
            p.add_argument("--before-baseline", default=None,
                           help="取基準之前先量的姿勢，逗號分隔，例如 natural。"
                                "這些段落不扣基準、只記原始角度，分析時靠 "
                                "trip_id 與同一趟的 upright 配對。不給就與先前一樣")
            p.add_argument("--desk-level", default=None,
                           choices=["base", "desk-up", "seat-up"],
                           help="桌面相對手肘高度的做法：base 原桌椅、desk-up "
                                "桌腳墊高、seat-up 座面墊高加腳踏。寫進 CSV 標頭")
            p.add_argument("--lift-cm", type=float, default=None,
                           help="實際墊高幾公分。base 自動記 0")
            p.add_argument("--desk-offset-cm", type=float, default=None,
                           help="桌面高減手肘高（公分，保留正負號），操作者手量")
            p.add_argument("--task", default=None,
                           help="自然坐的任務名稱，只寫進 natural 段落。"
                                "不給時記 copy-text；要抄的文字由操作者準備")
            p.add_argument("--target-angle-deg", default=None,
                           help="側傾的目標角度，逗號分隔，依序對應這一趟的 "
                                "tilt-left、tilt-right 段落")
            p.add_argument("--measured-angle-deg", default=None,
                           help="側傾的量角器實測值，逗號分隔，依序對應側傾段落。"
                                "不給的話每段開始前會問")
            p.add_argument("--measured-end-angle-deg", default=None,
                           help="側傾量完那一段之後，量角器再讀一次的值，逗號分隔，"
                                "依序對應側傾段落。不給的話每段量完會問，"
                                "留空就只有開始值。看一段 60 秒內上身滑掉多少")
            p.add_argument("--azimuth-target-deg", default=None,
                           help="表4 那一趟的目標方位角，一個值套用到每段，"
                                "或逗號分隔依序每段一個。只是標記，"
                                "分析用實際算出的方位角")
            p.add_argument("--seconds", type=float, default=30.0,
                           help="取基準的取樣長度")
            p.add_argument("--seconds-each", type=float, default=90.0,
                           help="每一種姿勢量多久。固定秒數比 Ctrl-C 好，"
                                "段落短受試者才不會忘記維持姿勢")
            p.add_argument("--countdown", type=int, default=15,
                           help="每一段開始前的倒數秒數，讓受試者坐定。"
                                "實測 3 秒與 20 秒的基準誤差差了 3.5 倍")
            p.add_argument("--out-dir", type=Path, default=Path("data/sessions"),
                           help="資料寫到 <out-dir>/<subject>/ 底下，檔名自動產生")
            p.add_argument("--no-prompt", action="store_true",
                           help="不等按鍵，每一段印完指導語就直接開始。"
                                "自己一個人量、或在測試裡跑的時候用")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    # analyse 與 cohort 只讀 CSV，標定檔與相機的檢查對它們都不適用。
    offline = {"analyse": _run_analyse, "cohort": _run_cohort,
               "rating-pack": _run_rating_pack, "ratings": _run_ratings,
               "profile-ref": _run_profile_ref}
    if args.mode in offline:
        offline[args.mode](args)
        return
    if args.mode == "run-plan" and args.dry_run:
        _run_plan(args)
        return
    if not args.calibration.is_file():
        raise SystemExit(
            f"找不到標定檔 {args.calibration}。先執行：\n"
            f"  python -m src.calibration.cli stereo --charuco ..."
        )
    # 每個子指令的選項不同，補上預設值讓三條路徑共用同一個 args
    args.all_keypoints = getattr(args, "all_keypoints", False)
    if args.camera != "auto":
        try:
            args.camera = int(args.camera)
        except ValueError:
            raise SystemExit(f"--camera 要填數字或 auto，收到 {args.camera!r}")
    if args.mode == "baseline" and args.out is None:
        # 檔名預設跟著受試者代號走。連續替幾個人取基準時，
        # 固定的預設檔名會讓後一個人蓋掉前一個人的基準。
        args.out = Path("data/baselines") / f"{args.subject}.json"
    _refuse_to_overwrite(args)

    {"once": _run_once, "live": _run_live, "baseline": _run_baseline,
     "study": _run_study, "monitor": _run_monitor,
     "scale": _run_scale, "run-plan": _run_plan,
     "preflight": _run_preflight}[args.mode](args)


if __name__ == "__main__":
    main()
