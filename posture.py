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
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from calibration.capture import (CameraReadError, _open_camera,  # noqa: E402
                                 describe_camera_loss,
                                 describe_camera_open_failure,
                                 describe_resolution_mismatch, find_camera_index,
                                 merged_capture_size, split_merged_frame)
from calibration.stereo_calibration import StereoCalibrationResult  # noqa: E402
from geometry.pipeline import (PersonMatch,  # noqa: E402
                               estimate_theta_ca_precision_deg,
                               format_measurement, match_person_pair,
                               measure_posture, rejection_advice, unusable_reason)
from geometry.baseline import (BaselineCollector,  # noqa: E402
                               PostureBaseline, baseline_quality_warnings)
from geometry.judgement import PostureJudge, windows_are_stale  # noqa: E402
from geometry.baseline import group_rejection_reason  # noqa: E402
from geometry.measurement_log import MeasurementLog  # noqa: E402
from geometry.cohort import Separation, collect  # noqa: E402
from geometry.cohort_report import to_csv, to_markdown  # noqa: E402
from geometry.session_analysis import analyse_session  # noqa: E402
from geometry.session_report import format_report, segment_table  # noqa: E402
from geometry.smoothing import RollingAngle  # noqa: E402
from geometry.uncertainty import standard_error  # noqa: E402
from geometry.terminal import cell, truncate  # noqa: E402
from pose.engine import (LightweightOpenPoseEngine,  # noqa: E402
                         LightweightOpenPoseModelPaths)
from pose.topology import COCO18_KEYPOINT_NAMES  # noqa: E402


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
    cap = _open_camera(args.camera, args.width, args.height)
    if not cap.isOpened():
        raise RuntimeError(describe_camera_open_failure(args.camera))
    return cap


# 相機掉線之後每次讀取都會立刻失敗，不擋的話迴圈會全速空轉。實機那次在
# 110 秒內寫了 162485 筆空記錄。每次失敗之間停一下，累積到這個次數就停止量測。
_READ_RETRY_PAUSE_S = 0.1
_MAX_CONSECUTIVE_READ_FAILURES = 25
# 略過率超過這個比例就提醒重量。乾淨的量測是 2~5%；2026-09-29 那份 23% 的資料
# 共同關鍵點從 12.2 掉到 9.0，留下來的幀代表不了整段姿勢。
_REJECTION_LIMIT = 0.15


def _grab_pair(cap, args):
    ok, frame = cap.read()
    if not ok:
        raise CameraReadError("讀取相機影格失敗")
    return split_merged_frame(frame, args.vertical_split, args.swap_lr)


# 同一個原因連續略過這麼多幀就提示一次。實機約 5fps，15 幀是三秒，
# 短到還來得及調整，長到不會被零星的失誤觸發。
_STUCK_AFTER_FRAMES = 15


class _StuckWatcher:
    """一直因為同一件事被略過時提示一次該怎麼辦。

    逐幀那一行只說「哪裡不對」，而且會被下一幀蓋掉，所以連續略過三十秒看到的
    是同一句話重複閃動，沒有人知道該動什麼。原因分組時把數字換掉，因為
    「垂直視差 8.9px」與「8.4px」是同一件事。

    提示每一段只印一次。反覆印會把它變成跟原本那一行一樣的背景雜訊。
    """

    def __init__(self, after: int = _STUCK_AFTER_FRAMES):
        self._after = after
        self._key: str | None = None
        self._run = 0
        self._told = False

    def saw(self, reason: str | None) -> str | None:
        """收下這一幀的略過原因（沒被略過就傳 None），回傳該印的提示。"""
        if reason is None:
            self._key, self._run, self._told = None, 0, False
            return None
        key = group_rejection_reason(reason)
        if key != self._key:
            self._key, self._run, self._told = key, 1, False
        else:
            self._run += 1
        # 門檻只在這裡判斷一次。分到兩個分支去判斷的話，一段的第一幀會被
        # 漏掉，提示就永遠晚一幀，而 after=1 這種設定會完全不合預期。
        if self._told or self._run < self._after:
            return None
        self._told = True
        advice = rejection_advice(reason)
        head = f"連續 {self._run} 幀都是同一個原因：{reason}"
        return head + ("\n  " + advice if advice else "")


def _camera_lost(args, failures: int) -> str | None:
    """連續讀取失敗到這個程度就當成相機不見了，回傳該印的話。

    偶爾掉一幀是正常的，相機不見了則重試多少次都一樣。分不開的話只有兩種
    壞法：太早放棄，或像實機那次一樣空轉到把有效資料埋掉。
    """
    if failures < _MAX_CONSECUTIVE_READ_FAILURES:
        return None
    return describe_camera_loss(
        args.camera, failures, failures * _READ_RETRY_PAUSE_S
    )


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


def snapshot_name(frame: int, elapsed_s: float, theta_ca_deg: float | None,
                  posture: str | None) -> str:
    """快照的檔名。要能不開 CSV 就看出這一張是什麼時候、量到多少、判成什麼。

    檔名排序要與時間順序一致，所以幀號補零；角度帶正負號，因為 θ_CA 的符號
    就是前傾與後仰的差別。角度算不出來時寫 na，不寫 0，0 是合法的角度。
    """
    angle = "na" if theta_ca_deg is None else f"{theta_ca_deg:+06.1f}"
    verdict = {"超標": "over", "正常": "ok", "未知": "unknown"}.get(posture or "", "none")
    return f"{frame:06d}_t{elapsed_s:05.1f}s_ca{angle}_{verdict}.png"


class _Snapshots:
    """每隔一段時間存一張左眼畫面。

    姿勢事後查證不了，所以要留下畫面。2026-09-29 連續兩次量測的結論都被推翻：
    一次是方位角在過程中漂了三十幾度，一次是受試者不自覺前傾而事後才想起來。
    兩次都不是程式的問題，而是「請坐正兩分鐘」這件事沒有留下任何獨立記錄，
    所以量到的數字既不能當誤報率、也不能當正確偵測的證據。

    存左眼就夠。要確認的是姿勢而不是立體配對，而且一半的張數換一半的磁碟。
    """

    def __init__(self, out_dir: Path | None, every_s: float):
        self._dir = out_dir
        self._every = every_s
        self._next_at = 0.0

    @property
    def enabled(self) -> bool:
        return self._dir is not None and self._every > 0

    def maybe_save(self, frame_index, elapsed_s, left_frame, left_kp,
                   theta_ca_deg, posture) -> str | None:
        """時間到了就存一張，回傳檔名。"""
        if not self.enabled or elapsed_s < self._next_at:
            return None
        self._next_at = elapsed_s + self._every
        name = snapshot_name(frame_index, elapsed_s, theta_ca_deg, posture)
        _write_frame(left_frame, self._dir / name, left_kp)
        return name


def _write_frame(frame, path: Path, keypoints=None) -> None:
    """把一張畫面寫成 png，偵測到的關鍵點疊上去。"""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = frame.copy()
    if keypoints is not None:
        for i, (x, y) in enumerate(keypoints.points):
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            cv2.circle(canvas, (int(x), int(y)), 4, (0, 255, 0), -1)
            cv2.putText(canvas, COCO18_KEYPOINT_NAMES[i], (int(x) + 6, int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.imwrite(str(path), canvas)


def _save_frames(left_frame, right_frame, out_dir: Path, left_kp=None, right_kp=None) -> Path:
    """把左右兩眼的畫面存下來，偵測到的關鍵點疊上去。

    診斷訊息說得出「左右配對錯誤」，但說不出相機到底看到什麼。人在不在畫面裡、
    頭有沒有被切掉、畫面裡還有什麼被當成人，這些看一眼就知道，
    用座標猜要來回好幾輪。
    """
    for name, frame, kp in (("left", left_frame, left_kp), ("right", right_frame, right_kp)):
        _write_frame(frame, out_dir / f"{name}.png", kp)
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

    print()
    print(_describe_match(match))
    print(format_measurement(
        measurement, match.left, match.right, show_all_keypoints=args.all_keypoints
    ))


@dataclass
class Recording:
    """錄製一段之後留下來的東西。

    回傳而不是直接印出來，因為 `study` 要把好幾段收集起來最後一起比較，
    而 `live` 只要印一段。判斷「這一段能不能用」的邏輯也只該寫一次。
    """

    session: "_Session"
    ca_window: RollingAngle
    sym_window: RollingAngle
    rejected: int
    frames: int
    transitions: list
    camera_lost: str | None = None
    log_path: Path | None = None
    condition: str = ""      # study 用，live 留空
    trial: int | None = None

    @property
    def rejection_rate(self) -> float:
        return 0.0 if self.frames == 0 else self.rejected / self.frames

    @property
    def usable(self) -> int:
        return len(self.session.ca)


def _prepare(args):
    """載入標定、開相機、暖機。回傳 (calib, engine, cap)。

    `study` 只做一次，之後每一段都重用。零間隔就是靠這一點：先前每跑一次
    `live` 都要重載模型，那段時間就是基準與量測之間的空檔，而那個空檔
    正是 2026-09-29 量到 63% 誤報率的原因。
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
            seconds: float | None = None) -> Recording:
    """錄製一段。seconds 是 None 就跑到 Ctrl-C 為止。

    固定秒數是實測逼出來的：2026-09-29 有一次量測跑到受試者忘記維持姿勢，
    整段資料作廢。段落短就不會忘。
    """
    judge = _make_judge(args, baseline)
    started_at = time.perf_counter()
    ca_window = RollingAngle(args.window)
    sym_window = RollingAngle(args.window)
    rejected = 0
    frames_seen = 0
    consecutive_misses = 0
    read_failures = 0
    saved_a_rejected_frame = False
    transitions: list[str] = []
    lost: str | None = None
    session = _Session()
    stuck = _StuckWatcher()

    try:
        while seconds is None or (time.perf_counter() - started_at) < seconds:
            frames_seen += 1
            try:
                measurement, match, frames = _measure_once(engine, calib, cap, args)
                read_failures = 0
            except RuntimeError as exc:
                if isinstance(exc, CameraReadError):
                    read_failures += 1
                    lost = _camera_lost(args, read_failures)
                    if lost is not None:
                        break
                    time.sleep(_READ_RETRY_PAUSE_S)
                # 讀不到畫面與偵測不到人都是這一幀沒有資料，一樣要計入略過，
                # 否則結束時印的數字會與實際寫下的筆數對不起來。
                rejected += 1
                consecutive_misses += 1
                _forget_if_stale(ca_window, sym_window, consecutive_misses)
                _tell(stuck.saw(str(exc)))
                _print_line(f"{exc}（已略過 {rejected} 幀）")
                if log is not None:
                    log.write(None, reject_reason=str(exc))
                continue

            reason = unusable_reason(measurement)
            corrected = _corrected(baseline, measurement)
            if reason is None:
                consecutive_misses = 0
                session.add(corrected)
                if corrected[0] is not None:
                    ca_window.add(corrected[0])
                if corrected[1] is not None:
                    sym_window.add(corrected[1])
            else:
                # 壞掉的幀混進平均比印出來更糟，所以先擋掉再計數。
                rejected += 1
                consecutive_misses += 1
                _forget_if_stale(ca_window, sym_window, consecutive_misses)
                if args.save_frames and not saved_a_rejected_frame:
                    _save_frames(*frames, args.save_frames, match.left, match.right)
                    saved_a_rejected_frame = True

            _tell(stuck.saw(reason))

            verdict = None
            if judge is not None:
                _, _, verdict = judge.update(ca_window, sym_window)
                if verdict.changed:
                    line = f"{_elapsed(started_at)}  {verdict.posture.value}：{verdict.reason}"
                    transitions.append(line)
                    # 狀態變化印成獨立的一行，因為它會被原地更新的那一行蓋掉。
                    _print_line("")
                    print("\r" + line, flush=True)

            _print_line(_live_line(
                ca_window, sym_window, measurement, corrected, rejected, reason,
                verdict, None if seconds is None else seconds - (time.perf_counter() - started_at)
            ))
            if log is not None:
                log.write(
                    measurement, reject_reason=reason, corrected=corrected,
                    ca_mean=ca_window.mean, sym_mean=sym_window.mean,
                    ca_standard_error=ca_window.standard_error,
                    posture="" if verdict is None else verdict.posture.value,
                )
            snapshots.maybe_save(
                0 if log is None else log.frames_written,
                time.perf_counter() - started_at, frames[0], match.left,
                corrected[0], None if verdict is None else verdict.posture.value,
            )
    except KeyboardInterrupt:
        pass

    return Recording(
        session=session, ca_window=ca_window, sym_window=sym_window,
        rejected=rejected, frames=frames_seen, transitions=transitions,
        camera_lost=lost, log_path=None if log is None else log.path,
    )


def _report(recording: Recording, baseline, log) -> None:
    """一段錄完之後印出來的東西。

    相機掉線時也要走完這一段。已經量到的東西才是這次的產出，不該因為
    結束的方式不同就不印。
    """
    print()
    if recording.camera_lost is not None:
        print(recording.camera_lost)
    _print_summary(recording.session, recording.ca_window, recording.sym_window,
                   recording.rejected, baseline)
    _print_transitions(recording.transitions)
    if log is not None:
        print(f"已記錄 {log.frames_written} 幀到 {log.path}")


def _run_live(args) -> None:
    calib, engine, cap = _prepare(args)
    baseline = _load_baseline(args, calib)
    log = _open_log(args, baseline)
    snapshots = _open_snapshots(args)
    print(f"開始量測，平均視窗 {args.window} 幀。"
          + ("Ctrl-C 結束" if args.seconds is None else f"{args.seconds:.0f} 秒後自動結束"),
          flush=True)
    print("要看的是平均值，不是單幀。單幀誤差與判定門檻同量級", flush=True)

    try:
        recording = _record(engine, calib, cap, args, baseline, log, snapshots,
                            args.seconds)
        _report(recording, baseline, log)
    finally:
        if log is not None:
            log.close()
        cap.release()
    if recording.camera_lost is not None:
        raise SystemExit(1)


def _open_log(args, baseline, condition: str = "", trial: int | None = None,
              path: Path | None = None):
    """開一份逐幀記錄。中繼資料一併寫進標頭。"""
    target = path if path is not None else args.log
    if target is None:
        return None
    log = MeasurementLog(
        target, subject=args.subject, overwrite=args.overwrite,
        condition=condition or getattr(args, "condition", "") or "",
        trial=trial if trial is not None else getattr(args, "trial", None),
        baseline_file=getattr(args, "baseline", None), baseline=baseline,
    )
    print(f"逐幀記錄到 {log.path}（含被略過的幀）", flush=True)
    return log


def _open_snapshots(args, out_dir: Path | None = None):
    target = out_dir if out_dir is not None else args.snapshots
    snapshots = _Snapshots(target, args.snapshot_every)
    if snapshots.enabled:
        print(f"每 {args.snapshot_every:.0f} 秒存一張畫面到 {target}，"
              f"事後可以核對當時的姿勢", flush=True)
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
    engine, calib, cap, args, seconds: float
) -> tuple[PostureBaseline, list[str]]:
    """請受試者保持不動，取這段時間的平均當作他的零點。

    相機與模型由呼叫端準備好。`study` 靠這一點讓取基準與量測之間沒有模型
    重載的空檔，而那個空檔正是 63% 誤報率的來源。
    """
    collector = BaselineCollector()
    start = time.perf_counter()
    saved_a_rejected_frame = False
    read_failures = 0
    while (elapsed := time.perf_counter() - start) < seconds:
        try:
            measurement, match, frames = _measure_once(engine, calib, cap, args)
            read_failures = 0
        except RuntimeError as exc:
            if isinstance(exc, CameraReadError):
                read_failures += 1
                lost = _camera_lost(args, read_failures)
                if lost is not None:
                    print()
                    raise SystemExit(lost)
                # 相機掉線時每次讀取都立刻失敗，不停一下的話這個迴圈會
                # 用滿剩下的取樣時間空轉。
                time.sleep(_READ_RETRY_PAUSE_S)
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
    try:
        print()
        print(f"請 {args.subject} 坐正、目視前方、雙肩放鬆，保持不動 {args.seconds:.0f} 秒。")
        _countdown(args.countdown)
        print("\r開始取樣，請保持不動        ", flush=True)
        baseline, _ = _collect_baseline(engine, calib, cap, args, args.seconds)
    finally:
        cap.release()

    baseline.save(args.out, overwrite=args.overwrite)
    print()
    print(f"已存到 {args.out}。之後這樣用：")
    print(f"  python posture.py live --baseline {args.out} --log data/sessions/xxx.csv ...")


def session_paths(root: Path, subject: str, condition: str, trial: int):
    """一次量測要寫出去的三個路徑。

    檔名自己說得出是誰、哪種姿勢、第幾次，因為彙整時靠中繼資料而不是檔名，
    但人在檔案總管裡找東西還是靠檔名。
    """
    folder = Path(root) / subject
    stem = f"{condition}-{trial}"
    return folder / f"{stem}.csv", folder / f"{stem}-shots", folder / "baseline.json"


def _next_trial(root: Path, subject: str, condition: str) -> int:
    """這個人這種姿勢已經量過幾次了。

    自動接續而不是每次都從 1 開始，因為撞名是 2026-09-29 真的發生過的事，
    而當時受試者正坐著等。
    """
    folder = Path(root) / subject
    if not folder.is_dir():
        return 1
    used = set()
    for path in folder.glob(f"{condition}-*.csv"):
        tail = path.stem[len(condition) + 1:]
        if tail.isdigit():
            used.add(int(tail))
    return max(used) + 1 if used else 1


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


def _condition_brief(condition: str) -> str:
    """每種姿勢的指導語。

    只有坐正與前傾寫死，其他名稱照樣跑得動，只是指導語要口頭給。姿勢種類
    還沒定案，程式不該先把清單釘死。
    """
    known = {
        "upright": "請坐正：背部貼著椅背、雙腳平放地面、下巴微收、"
                   "視線看向前方牆上的固定點。**全程不要看螢幕。**",
        "forward": "請刻意前傾：上半身往前，但不要轉身，視線仍然看向同一個固定點。",
    }
    return known.get(condition, f"請維持「{condition}」的姿勢，視線看向前方的固定點。")


def _run_study(args) -> None:
    """一個行程跑完整套流程：取基準，接著逐一量各種姿勢。

    流程本身就是資料品質的一部分。2026-09-29 那晚四次量測作廢三次，沒有一次
    是程式算錯：解析度不符、基準隔了九分鐘、量測中轉頭看螢幕、受試者不自覺
    前傾。把流程寫進程式，做錯就變難。

    最關鍵的一點是**基準與量測之間沒有模型重載的空檔**。先前那兩件事是兩次
    獨立的指令，中間要重新載入模型，受試者就會站起來活動一下。誤報率 63%
    就是從那個空檔來的。
    """
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    if not conditions:
        raise SystemExit("--conditions 至少要有一種姿勢，例如 upright,forward")

    calib, engine, cap = _prepare(args)
    root = args.out_dir
    recordings = []
    try:
        baseline = _study_baseline(engine, calib, cap, args, root)
        for condition in conditions:
            recordings.append(
                _study_one(engine, calib, cap, args, baseline, condition, root)
            )
    finally:
        cap.release()

    _study_summary(recordings, args)


def _study_baseline(engine, calib, cap, args, root: Path):
    """取基準，品質不好就問要不要重取。

    取完就直接往下走是不行的。2026-09-29 有一份基準略過了 27% 的幀，當時
    沒有任何提示，而後面所有量測都帶著它。
    """
    _, _, baseline_path = session_paths(root, args.subject, "x", 1)
    while True:
        _prompt(f"接下來要取 {args.subject} 的個人基準，取樣 {args.seconds:.0f} 秒。\n"
                + _condition_brief("upright")
                + "\n  取完之後**不要起身**，會直接接著量測。",
                not args.no_prompt)
        _countdown(args.countdown)
        print("\r開始取樣，請保持不動        ", flush=True)
        baseline, warnings = _collect_baseline(engine, calib, cap, args, args.seconds)
        if not warnings or args.no_prompt:
            break
        print()
        if not _ask_yes("這份基準有上面的問題，要重取一次嗎？"):
            break

    baseline.save(baseline_path, overwrite=True)
    print(f"基準存到 {baseline_path}", flush=True)
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


def _study_one(engine, calib, cap, args, baseline, condition: str, root: Path):
    """量一種姿勢。"""
    trial = _next_trial(root, args.subject, condition)
    csv_path, shots_dir, baseline_path = session_paths(
        root, args.subject, condition, trial
    )
    _prompt(f"接下來量「{condition}」第 {trial} 次，{args.seconds_each:.0f} 秒。\n"
            + _condition_brief(condition), not args.no_prompt)

    log = MeasurementLog(
        csv_path, subject=args.subject, overwrite=args.overwrite,
        condition=condition, trial=trial,
        baseline_file=baseline_path, baseline=baseline,
    )
    snapshots = _Snapshots(shots_dir, args.snapshot_every)
    print(f"記錄到 {csv_path}，畫面存到 {shots_dir}", flush=True)
    try:
        recording = _record(engine, calib, cap, args, baseline, log, snapshots,
                            args.seconds_each)
        _report(recording, baseline, log)
    finally:
        log.close()
    recording.condition = condition
    recording.trial = trial
    return recording


def _study_summary(recordings, args) -> None:
    """全部量完之後的總結。

    每一段自己的摘要已經印過了，這裡看的是段與段之間：哪一段的資料不能用、
    以及兩種姿勢分不分得開。
    """
    print()
    print("=" * 60)
    print(f"受試者 {args.subject}，共 {len(recordings)} 段")
    for recording in recordings:
        mean = recording.session.mean_ca
        shown = "—" if mean is None else f"{mean:+.2f}°"
        flag = "" if recording.rejection_rate <= _REJECTION_LIMIT else "  ← 略過率偏高"
        print(f"  {recording.condition:12s} #{recording.trial}  "
              f"θ_CA {shown:>9s}  "
              f"可用 {recording.usable}/{recording.frames}"
              f"（略過 {recording.rejection_rate * 100:.0f}%）{flag}")

    bad = [r for r in recordings if r.rejection_rate > _REJECTION_LIMIT]
    if bad:
        print()
        print(f"略過率超過 {_REJECTION_LIMIT * 100:.0f}% 的段落，資料的代表性有限，"
              f"建議重量：" + "、".join(f"{r.condition} #{r.trial}" for r in bad))

    print()
    print("接下來：先翻一遍存下來的畫面，確認每一張都是預期的姿勢，再跑")
    print("  python posture.py analyse "
          + " ".join(str(r.log_path) for r in recordings if r.log_path))


def _print_line(text: str) -> None:
    """原地更新一行，裁到終端機的實際寬度。

    這一行放不下時終端機會折行，而 CR 只退到最後一行的開頭，畫面就變成
    一串接不起來的殘句。要按顯示寬度裁，因為中文一個字佔兩欄，用字元數裁的話
    留下來的字數雖然對，佔用的欄數是兩倍，照樣會折行。被略過的那些幀
    印的是中文原因，正好是最長、最容易超出的一種。
    """
    width = shutil.get_terminal_size(fallback=(100, 24)).columns - 1
    print("\r" + cell(truncate(text, width), width), end="", flush=True)


def _angle_text(window: RollingAngle, instant: float | None) -> str:
    """平均值擺前面，單幀值放在括號裡，要看的是平均。"""
    mean = window.mean
    error = window.standard_error
    if mean is None:
        return "   —  "
    shown = f"{mean:+5.1f}" + (f"±{error:.1f}" if error is not None else "     ")
    return f"{shown}(單幀{instant:+5.1f})" if instant is not None else f"{shown}(單幀  — )"


def _where(measurement) -> str:
    """距離與方位角。方位角要雙肩才量得到，所以它常常是 None。

    一邊肩膀沒偵測到時，深度與 θ_CA 仍然算得出來，那一幀不會被擋掉，
    但方位角是 None。直接格式化會在那一幀當掉。
    """
    distance = measurement.reference_depth_mm
    azimuth = measurement.camera_azimuth_deg
    return (
        ("  ——mm" if distance is None else f"  {distance:4.0f}mm")
        + (" ——°" if azimuth is None else f" {azimuth:2.0f}°")
    )


def _live_line(
    ca_window: RollingAngle, sym_window: RollingAngle, measurement,
    corrected, rejected: int, reason, verdict=None, remaining_s: float | None = None,
) -> str:
    # 固定秒數的段落要顯示剩下多久。受試者維持姿勢時最想知道的就是這個，
    # 不知道還要多久就容易提早鬆掉。
    left = "" if remaining_s is None else f"  剩 {max(0.0, remaining_s):3.0f}s"
    if reason is not None:
        # 調整架設位置時正是略過最多的時候，這幾個數字不能跟著消失
        return f"略過：{reason}{_where(measurement)}  已略過 {rejected} 幀{left}"
    # 括號裡放的是扣掉基準之後的單幀值。放原始角度的話它跟前面的平均差了一個
    # 基準的量，看起來像兩個不相干的數字。
    return (
        f"θ_CA {_angle_text(ca_window, corrected[0])}"
        f"  θ_sym {_angle_text(sym_window, corrected[1])}"
        f"{_where(measurement)}"
        f"  {ca_window.count:2d}/{ca_window.window}幀"
        + (f"  略過{rejected}" if rejected else "")
        + ("" if verdict is None else f"  {verdict.posture.value}")
        + left
    )


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
    print(f"判定門檻 θ_CA {judge.theta_ca.threshold_deg:.0f}°（只看前傾）、"
          f"θ_sym {judge.theta_sym.threshold_deg:.0f}°（左右都算），"
          f"遲滯寬度 {args.margin:.1f} 倍標準誤差", flush=True)
    return judge


def _forget_if_stale(
    ca_window: RollingAngle, sym_window: RollingAngle, consecutive_misses: int
) -> None:
    """偵測連續失敗久了就把移動平均清掉，讓狀態回到未知。

    不清的話，受試者離開座位之後裝置會對著空椅子繼續回報上一個狀態。
    """
    if windows_are_stale(consecutive_misses, ca_window.window):
        ca_window.clear()
        sym_window.clear()


class _Session:
    """整段量測收下的角度。

    移動視窗只留最近 N 個，結尾的摘要需要的是全部。CSV 也有，但要看一眼結果
    就得再開一個程式，那不合理。
    """

    def __init__(self):
        self.ca: list[float] = []
        self.sym: list[float] = []

    def add(self, corrected) -> None:
        if corrected[0] is not None:
            self.ca.append(corrected[0])
        if corrected[1] is not None:
            self.sym.append(corrected[1])

    @property
    def mean_ca(self) -> float | None:
        """整段的 θ_CA 平均。沒有資料時回傳 None 而不是 0，0 是合法的角度。"""
        return float(np.mean(self.ca)) if self.ca else None

    @property
    def mean_sym(self) -> float | None:
        return float(np.mean(self.sym)) if self.sym else None


def _tell(message: str | None) -> None:
    """把提示印成獨立的一行。逐幀那一行會被蓋掉，提示不該跟著消失。"""
    if message is None:
        return
    _print_line("")
    print("\r" + message, flush=True)


def _elapsed(started_at: float) -> str:
    seconds = time.perf_counter() - started_at
    return f"{int(seconds) // 60:2d}:{int(seconds) % 60:02d}"


def _print_transitions(transitions: list[str]) -> None:
    """把狀態變化重印一次。

    逐幀的那一行會被下一幀蓋掉，狀態變化夾在裡面很容易錯過，而這幾行
    正是要記錄下來的東西。
    """
    if not transitions:
        print("整段沒有狀態變化")
        return
    print(f"狀態變化 {len(transitions)} 次：")
    for line in transitions:
        print(f"  {line}")


def _print_summary(
    session: "_Session", ca_window: RollingAngle, sym_window: RollingAngle,
    rejected: int, baseline: PostureBaseline | None = None,
) -> None:
    """結束時把整段的統計印出來，這才是可以記錄下來的數字。

    印的是**整段**，不是移動視窗。視窗只有 30 幀（約 6 秒），拿它當結尾的摘要
    等於把一百秒的量測講成最後六秒的樣子，而標題寫的是整段。移動視窗另外印
    一行，因為判定看的是它，兩個數字差很多本身就是資訊：那代表姿勢在變。
    """
    print("整段（相對個人基準的偏移量）：" if baseline else "整段（原始角度，未扣除個人基準）：")
    for name, values in (("θ_CA ", session.ca), ("θ_sym", session.sym)):
        if len(values) < 2:
            print(f"{name}  沒有足夠的量測")
            continue
        array = np.asarray(values)
        error = standard_error(array)
        shown = "" if error is None else f" ± {error:.1f}°"
        print(f"{name}  {array.mean():+.2f}°{shown}"
              f"（{len(values)} 幀，單幀標準差 ±{array.std():.1f}°）")
    for name, window in (("θ_CA ", ca_window), ("θ_sym", sym_window)):
        if window.mean is None:
            continue
        error = f" ± {window.standard_error:.1f}°" if window.standard_error is not None else ""
        print(f"  結束前 {window.count} 幀  {name} {window.mean:+.2f}°{error}")
    if rejected:
        total = rejected + len(session.ca)
        print(f"略過 {rejected} / {total} 幀（{rejected / total * 100:.0f}%）偵測失誤")


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
    markdown = to_markdown(cohort, separation)
    print(markdown)

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "cohort.md").write_text(markdown, encoding="utf-8")
        (args.out / "cohort.csv").write_text(to_csv(cohort), encoding="utf-8")
        print(f"已寫到 {args.out / 'cohort.md'} 與 {args.out / 'cohort.csv'}")


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


def main() -> None:
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
                    help="把 Markdown 與 CSV 寫到這個資料夾。不給就只印出來")

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

    modes = (
        ("once", "量測一次並印出完整診斷"),
        ("live", "持續量測，印移動平均"),
        ("baseline", "請受試者坐正保持不動，取個人基準（θ_offset）"),
        ("study", "一個行程跑完整套實驗流程：取基準，接著逐一量各種姿勢"),
    )
    for name, help_text in modes:
        p = sub.add_parser(name, help=help_text)
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
        p.add_argument("--discard", type=int, default=5,
                       help="開始量測前先丟掉幾張，讓自動曝光穩定")
        p.add_argument("--input-height", type=int, default=256,
                       help="網路輸入高度。調高會直接降低熱圖的量化誤差（深度精度的"
                            "主要瓶頸），代價是推論變慢；384或512值得一試")
        p.add_argument("--no-subpixel", action="store_true",
                       help="關掉熱圖峰值的次像素精修，用來量化它的影響")
        p.add_argument("--save-frames", type=Path, default=None,
                       help="把左右兩眼的畫面存成 png，偵測到的關鍵點疊上去。"
                            "診斷訊息說不出相機到底看到什麼，這個看得出來。"
                            "baseline 與 live 存的是第一張被略過的畫面")
        p.add_argument("--window", type=int, default=30,
                       help="live 模式的平均視窗幀數。單幀誤差與判定門檻同量級，"
                            "平均N幀把偵測雜訊降到1/√N；30幀約5秒")
        if name == "once":
            p.add_argument("--all-keypoints", action="store_true",
                           help="印出全部18點，不只角度用到的那幾個")
        if name in ("live", "baseline", "study"):
            p.add_argument("--subject", default="受試者",
                           help="受試者代號，寫進基準檔與 CSV")
        if name == "live":
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
            p.add_argument("--seconds", type=float, default=None,
                           help="量測長度。不給就跑到 Ctrl-C 為止。"
                                "固定秒數的段落比較不會發生受試者忘記維持姿勢")
            p.add_argument("--condition", default="",
                           help="姿勢條件的名稱，例如 upright 或 forward。"
                                "寫進 CSV 標頭，彙整時靠它分組")
            p.add_argument("--trial", type=int, default=None,
                           help="同一個人同一種姿勢的第幾次，寫進 CSV 標頭")
        if name in ("live", "study"):
            p.add_argument("--margin", type=float, default=1.0,
                           help="遲滯寬度，單位是標準誤差的倍數。調高會減少誤報但"
                                "反應變慢、不動作的區間變寬")
            p.add_argument("--no-judge", action="store_true",
                           help="只印角度不做超標判定，用來收集原始資料")
            p.add_argument("--snapshot-every", type=float, default=10.0,
                           help="存畫面的間隔秒數。study 一定會存，live 要搭配 --snapshots")
        if name in ("live", "baseline", "study"):
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
                                "程式不限定種類，只把它寫進 CSV 與檔名")
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

    args = parser.parse_args()
    # analyse 與 cohort 只讀 CSV，標定檔與相機的檢查對它們都不適用。
    if args.mode in ("analyse", "cohort"):
        {"analyse": _run_analyse, "cohort": _run_cohort}[args.mode](args)
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
     "study": _run_study, "analyse": _run_analyse}[args.mode](args)


if __name__ == "__main__":
    main()
