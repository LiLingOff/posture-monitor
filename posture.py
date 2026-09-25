"""端到端量測：相機 → 關鍵點 → 三角測量 → 坐姿角度。

放在 repo 根目錄而不是做成 `python -m src.geometry.cli`，是因為 `src/geometry`
用的是絕對匯入（`from calibration...`），需要 `src/` 在 sys.path 上；
而 `-m src.geometry.cli` 的 sys.path[0] 是 repo 根目錄。這裡先補上路徑再匯入，
與 `conftest.py` 給測試用的做法一致。

用法：
  python posture.py once   --width 2560 --height 720
  python posture.py live   --width 2560 --height 720
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from calibration.capture import (_open_camera,  # noqa: E402
                                 describe_camera_open_failure, find_camera_index,
                                 split_merged_frame)
from calibration.stereo_calibration import StereoCalibrationResult  # noqa: E402
from geometry.pipeline import (PersonMatch,  # noqa: E402
                               estimate_theta_ca_precision_deg,
                               format_measurement, match_person_pair,
                               measure_posture, unusable_reason)
from geometry.baseline import (BaselineCollector,  # noqa: E402
                               PostureBaseline, baseline_quality_warnings)
from geometry.measurement_log import MeasurementLog  # noqa: E402
from geometry.smoothing import RollingAngle  # noqa: E402
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


def _open_selected_camera(args):
    """開啟相機。--camera auto 時先自動挑一個讀得出畫面的節點。"""
    if args.camera == "auto":
        args.camera, how = find_camera_index(args.width, args.height)
        _step(how)
    cap = _open_camera(args.camera, args.width, args.height)
    if not cap.isOpened():
        raise RuntimeError(describe_camera_open_failure(args.camera))
    return cap


def _grab_pair(cap, args):
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("讀取相機影格失敗")
    return split_merged_frame(frame, args.vertical_split, args.swap_lr)


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
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame, kp in (("left", left_frame, left_kp), ("right", right_frame, right_kp)):
        canvas = frame.copy()
        if kp is not None:
            for i, (x, y) in enumerate(kp.points):
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                cv2.circle(canvas, (int(x), int(y)), 4, (0, 255, 0), -1)
                cv2.putText(canvas, COCO18_KEYPOINT_NAMES[i], (int(x) + 6, int(y) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        cv2.imwrite(str(out_dir / f"{name}.png"), canvas)
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
    cap = _open_selected_camera(args)
    try:
        _step(f"等自動曝光穩定，丟掉前 {args.discard} 張")
        _discard_frames(cap, args.discard)

        left_frame, right_frame = _grab_pair(cap, args)
        _step(f"取得畫面，單眼 {left_frame.shape[1]}x{left_frame.shape[0]}")

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


def _run_live(args) -> None:
    calib = StereoCalibrationResult.load(args.calibration)
    engine = _build_engine(args)
    cap = _open_selected_camera(args)

    _discard_frames(cap, args.discard)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("讀取相機影格失敗")
    print(f"載入模型（{args.precision} / {args.device}）", flush=True)
    engine.warmup(split_merged_frame(frame, args.vertical_split, args.swap_lr)[0])

    baseline = _load_baseline(args, calib)
    log = (MeasurementLog(args.log, subject=args.subject, overwrite=args.overwrite)
           if args.log else None)
    if log is not None:
        print(f"逐幀記錄到 {log.path}（含被略過的幀）", flush=True)
    print(f"開始量測，平均視窗 {args.window} 幀。Ctrl-C 結束", flush=True)
    print("要看的是平均值，不是單幀。單幀誤差與判定門檻同量級", flush=True)

    ca_window = RollingAngle(args.window)
    sym_window = RollingAngle(args.window)
    rejected = 0
    saved_a_rejected_frame = False
    try:
        while True:
            try:
                measurement, match, frames = _measure_once(engine, calib, cap, args)
            except RuntimeError as exc:
                _print_line(str(exc))
                if log is not None:
                    log.write(None, reject_reason=str(exc))
                continue

            reason = unusable_reason(measurement)
            corrected = _corrected(baseline, measurement)
            if reason is None:
                if corrected[0] is not None:
                    ca_window.add(corrected[0])
                if corrected[1] is not None:
                    sym_window.add(corrected[1])
            else:
                # 壞掉的幀混進平均比印出來更糟，所以先擋掉再計數。
                rejected += 1
                if args.save_frames and not saved_a_rejected_frame:
                    _save_frames(*frames, args.save_frames, match.left, match.right)
                    saved_a_rejected_frame = True

            _print_line(_live_line(
                ca_window, sym_window, measurement, corrected, rejected, reason
            ))
            if log is not None:
                log.write(
                    measurement, reject_reason=reason, corrected=corrected,
                    ca_mean=ca_window.mean, sym_mean=sym_window.mean,
                    ca_standard_error=ca_window.standard_error,
                )
    except KeyboardInterrupt:
        print()
        _print_summary(ca_window, sym_window, rejected, baseline)
        if log is not None:
            print(f"已記錄 {log.frames_written} 幀到 {log.path}")
    finally:
        if log is not None:
            log.close()
        cap.release()


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


def _run_baseline(args) -> None:
    """請受試者坐正保持不動，取這段時間的平均當作他的零點。"""
    calib = StereoCalibrationResult.load(args.calibration)
    engine = _build_engine(args)
    cap = _open_selected_camera(args)

    collector = BaselineCollector()
    try:
        _discard_frames(cap, args.discard)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("讀取相機影格失敗")
        _step(f"載入模型（{args.precision} / {args.device}）")
        engine.warmup(split_merged_frame(frame, args.vertical_split, args.swap_lr)[0])

        print()
        print(f"請 {args.subject} 坐正、目視前方、雙肩放鬆，保持不動 {args.seconds:.0f} 秒。")
        for remaining in range(args.countdown, 0, -1):
            print(f"\r{remaining} 秒後開始…", end="", flush=True)
            time.sleep(1.0)
        print("\r開始取樣，請保持不動        ", flush=True)

        start = time.perf_counter()
        saved_a_rejected_frame = False
        while (elapsed := time.perf_counter() - start) < args.seconds:
            try:
                measurement, match, frames = _measure_once(engine, calib, cap, args)
            except RuntimeError as exc:
                _print_line(str(exc))
                continue
            reason = collector.add(measurement)
            # 存第一張被略過的畫面。全部被略過時，那正是唯一想看的東西。
            if reason and args.save_frames and not saved_a_rejected_frame:
                _save_frames(*frames, args.save_frames, match.left, match.right)
                saved_a_rejected_frame = True
            _print_line(
                f"剩下 {args.seconds - elapsed:4.1f} 秒   已收 {collector.count:3d} 幀"
                + (f"   略過 {collector.rejected}" if collector.rejected else "")
                + (f"   （{reason}）" if reason else "")
            )
        print()
    finally:
        cap.release()

    baseline = collector.finish(args.subject, time.perf_counter() - start)
    print()
    print(baseline.describe())
    for warning in collector.quality_warnings(baseline):
        print(f"  需要注意：{warning}")

    baseline.save(args.out, overwrite=args.overwrite)
    print()
    print(f"已存到 {args.out}。之後這樣用：")
    print(f"  python posture.py live --baseline {args.out} --log data/sessions/xxx.csv ...")


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
    corrected, rejected: int, reason,
) -> str:
    if reason is not None:
        # 調整架設位置時正是略過最多的時候，這幾個數字不能跟著消失
        return f"略過：{reason}{_where(measurement)}  已略過 {rejected} 幀"
    # 括號裡放的是扣掉基準之後的單幀值。放原始角度的話它跟前面的平均差了一個
    # 基準的量，看起來像兩個不相干的數字。
    return (
        f"θ_CA {_angle_text(ca_window, corrected[0])}"
        f"  θ_sym {_angle_text(sym_window, corrected[1])}"
        f"{_where(measurement)}"
        f"  {ca_window.count:2d}/{ca_window.window}幀"
        + (f"  略過{rejected}" if rejected else "")
    )


def _print_summary(
    ca_window: RollingAngle, sym_window: RollingAngle, rejected: int,
    baseline: PostureBaseline | None = None,
) -> None:
    """結束時把整段的統計印出來，這才是可以記錄下來的數字。"""
    print("相對個人基準的偏移量：" if baseline else "原始角度（未扣除個人基準）：")
    for name, window in (("θ_CA ", ca_window), ("θ_sym", sym_window)):
        if window.mean is None:
            print(f"{name}  沒有可用的量測")
            continue
        spread = f"，單幀標準差 ±{window.std:.1f}°" if window.std is not None else ""
        error = f" ± {window.standard_error:.1f}°" if window.standard_error is not None else ""
        print(f"{name}  {window.mean:+.2f}°{error}（{window.count} 幀{spread}）")
    if rejected:
        print(f"略過 {rejected} 幀偵測失誤")


def _refuse_to_overwrite(args) -> None:
    """在開相機之前就檢查輸出檔。

    基準要請受試者坐著不動半分鐘、量測一次要跑二十分鐘，等到做完才發現檔名撞到，
    白費的是受試者的時間。
    """
    target = None
    if args.mode == "baseline":
        target = args.out
    elif args.mode == "live":
        target = args.log
    if target is not None and Path(target).exists() and not args.overwrite:
        raise SystemExit(
            f"{target} 已經存在。換個檔名，或確定要覆蓋的話加上 --overwrite"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="端到端坐姿量測")
    sub = parser.add_subparsers(dest="mode", required=True)

    modes = (
        ("once", "量測一次並印出完整診斷"),
        ("live", "持續量測，印移動平均"),
        ("baseline", "請受試者坐正保持不動，取個人基準（θ_offset）"),
    )
    for name, help_text in modes:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--calibration", type=Path,
                       default=Path("data/calibration_output/stereo.npz"))
        p.add_argument("--camera", default="auto",
                       help="相機 index，或 auto 自動挑。節點編號會因為重新插拔或"
                            "重開機而移位，auto 會逐一試到讀得出畫面為止")
        p.add_argument("--width", type=int, default=None,
                       help="必須與標定時的解析度一致")
        p.add_argument("--height", type=int, default=None)
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
        if name in ("live", "baseline"):
            p.add_argument("--subject", default="受試者",
                           help="受試者代號，寫進基準檔與 CSV")
        if name == "live":
            p.add_argument("--baseline", type=Path, default=None,
                           help="個人基準檔，由 baseline 子指令產生。"
                                "指定之後印出的是相對這個人端正坐姿的偏移量")
            p.add_argument("--log", type=Path, default=None,
                           help="把每一幀寫成 CSV，含被略過的幀。"
                                "事後分析與 Kinovea 對標都需要這份原始資料")
        if name in ("live", "baseline"):
            p.add_argument("--overwrite", action="store_true",
                           help="允許覆蓋既有的基準檔或記錄檔")
        if name == "baseline":
            p.add_argument("--out", type=Path, default=None,
                           help="預設是 data/baselines/<subject>.json")
            p.add_argument("--seconds", type=float, default=30.0,
                           help="取樣長度。實機約 6.4fps，30 秒約 190 幀，基準誤差約 ±2°")
            p.add_argument("--countdown", type=int, default=3,
                           help="開始前的倒數秒數，讓受試者坐定")

    args = parser.parse_args()
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

    {"once": _run_once, "live": _run_live, "baseline": _run_baseline}[args.mode](args)


if __name__ == "__main__":
    main()
