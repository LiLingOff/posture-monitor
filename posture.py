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

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from calibration.capture import (_open_camera,  # noqa: E402
                                 describe_camera_open_failure, split_merged_frame)
from calibration.stereo_calibration import StereoCalibrationResult  # noqa: E402
from geometry.pipeline import (format_measurement,  # noqa: E402
                               measure_posture, unusable_reason)
from geometry.baseline import (BaselineCollector,  # noqa: E402
                               PostureBaseline)
from geometry.measurement_log import MeasurementLog  # noqa: E402
from geometry.smoothing import RollingAngle  # noqa: E402
from pose.engine import (LightweightOpenPoseEngine,  # noqa: E402
                         LightweightOpenPoseModelPaths)


def _build_engine(args) -> LightweightOpenPoseEngine:
    paths = LightweightOpenPoseModelPaths(
        checkpoint=args.checkpoint, engine_cache=args.engine_cache, repo_dir=args.repo_dir
    )
    return LightweightOpenPoseEngine(
        paths, precision=args.precision, device=args.device,
        input_height=args.input_height, subpixel=not args.no_subpixel,
    )


def _first_person(detections, side: str):
    if not detections:
        raise RuntimeError(
            f"{side}眼沒有偵測到人。確認受試者在畫面內、光線足夠，"
            f"並檢查左右畫面是不是同一個場景"
        )
    return detections[0]


def _grab_pair(cap, args):
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("讀取相機影格失敗")
    return split_merged_frame(frame, args.vertical_split, args.swap_lr)


def _measure_once(engine, calib, cap, args):
    left_frame, right_frame = _grab_pair(cap, args)
    left = _first_person(engine.infer(left_frame), "左")
    right = _first_person(engine.infer(right_frame), "右")
    return measure_posture(calib, left, right, args.min_confidence), left, right


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
    cap = _open_camera(args.camera, args.width, args.height)
    if not cap.isOpened():
        raise RuntimeError(describe_camera_open_failure(args.camera))
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

        left = _first_person(engine.infer(left_frame), "左")
        right = _first_person(engine.infer(right_frame), "右")
        measurement = measure_posture(calib, left, right, args.min_confidence)
    finally:
        cap.release()

    print()
    print(format_measurement(measurement, left, right, show_all_keypoints=args.all_keypoints))


def _run_live(args) -> None:
    calib = StereoCalibrationResult.load(args.calibration)
    engine = _build_engine(args)
    cap = _open_camera(args.camera, args.width, args.height)
    if not cap.isOpened():
        raise RuntimeError(describe_camera_open_failure(args.camera))

    _discard_frames(cap, args.discard)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("讀取相機影格失敗")
    print(f"載入模型（{args.precision} / {args.device}）", flush=True)
    engine.warmup(split_merged_frame(frame, args.vertical_split, args.swap_lr)[0])

    baseline = _load_baseline(args)
    log = MeasurementLog(args.log, subject=args.subject) if args.log else None
    if log is not None:
        print(f"逐幀記錄到 {log.path}（含被略過的幀）", flush=True)
    print(f"開始量測，平均視窗 {args.window} 幀。Ctrl-C 結束", flush=True)
    print("要看的是平均值，不是單幀——單幀誤差與判定門檻同量級", flush=True)

    ca_window = RollingAngle(args.window)
    sym_window = RollingAngle(args.window)
    rejected = 0
    try:
        while True:
            try:
                measurement, _, _ = _measure_once(engine, calib, cap, args)
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

            _print_line(_live_line(ca_window, sym_window, measurement, rejected, reason))
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


def _load_baseline(args) -> PostureBaseline | None:
    if not args.baseline:
        print("沒有指定個人基準，印出的是原始角度。", flush=True)
        print("判定門檻套在原始角度上會因人而異，正式量測前先跑一次 "
              "python posture.py baseline", flush=True)
        return None
    baseline = PostureBaseline.load(args.baseline)
    print(baseline.describe(), flush=True)
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
    cap = _open_camera(args.camera, args.width, args.height)
    if not cap.isOpened():
        raise RuntimeError(describe_camera_open_failure(args.camera))

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
        while (elapsed := time.perf_counter() - start) < args.seconds:
            try:
                measurement, _, _ = _measure_once(engine, calib, cap, args)
            except RuntimeError as exc:
                _print_line(str(exc))
                continue
            reason = collector.add(measurement)
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

    baseline.save(args.out)
    print()
    print(f"已存到 {args.out}。之後這樣用：")
    print(f"  python posture.py live --baseline {args.out} --log data/sessions/xxx.csv ...")


def _print_line(text: str) -> None:
    """原地更新一行，並裁到終端機寬度。

    加了欄位之後這一行超出終端機寬度，換行之後 \\r 只退到該行開頭，
    畫面就變成一串接不起來的殘句。
    """
    width = shutil.get_terminal_size(fallback=(100, 24)).columns - 1
    print(f"\r{text[:width]:<{width}}", end="", flush=True)


def _angle_text(window: RollingAngle, instant: float | None) -> str:
    """平均值擺前面，單幀值放在括號裡——要看的是平均。"""
    mean = window.mean
    error = window.standard_error
    if mean is None:
        return "   —  "
    shown = f"{mean:+5.1f}" + (f"±{error:.1f}" if error is not None else "     ")
    return f"{shown}(單幀{instant:+5.1f})" if instant is not None else f"{shown}(單幀  — )"


def _live_line(
    ca_window: RollingAngle, sym_window: RollingAngle, measurement, rejected: int, reason
) -> str:
    if reason is not None:
        return f"略過這一幀：{reason}（已略過 {rejected} 幀）"
    distance = measurement.reference_depth_mm
    azimuth = measurement.camera_azimuth_deg
    return (
        f"θ_CA {_angle_text(ca_window, measurement.theta_ca_deg)}"
        f"  θ_sym {_angle_text(sym_window, measurement.theta_sym_deg)}"
        f"  {distance:4.0f}mm {azimuth:2.0f}°"
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
        p.add_argument("--camera", type=int, default=0)
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
        p.add_argument("--window", type=int, default=30,
                       help="live 模式的平均視窗幀數。單幀誤差與判定門檻同量級，"
                            "平均N幀把雜訊降到1/√N；30幀約5秒")
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
        if name == "baseline":
            p.add_argument("--out", type=Path, default=Path("data/baselines/baseline.json"))
            p.add_argument("--seconds", type=float, default=10.0,
                           help="取樣長度。實機約 6.4fps，10 秒約 60 幀")
            p.add_argument("--countdown", type=int, default=3,
                           help="開始前的倒數秒數，讓受試者坐定")

    args = parser.parse_args()
    if not args.calibration.is_file():
        raise SystemExit(
            f"找不到標定檔 {args.calibration}。先執行：\n"
            f"  python -m src.calibration.cli stereo --charuco ..."
        )
    # live 沒有這個選項，補一個預設值讓兩條路徑共用同一個 args
    args.all_keypoints = getattr(args, "all_keypoints", False)

    {"once": _run_once, "live": _run_live, "baseline": _run_baseline}[args.mode](args)


if __name__ == "__main__":
    main()
