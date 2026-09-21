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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from calibration.capture import _open_camera, split_merged_frame  # noqa: E402
from calibration.stereo_calibration import StereoCalibrationResult  # noqa: E402
from geometry.pipeline import format_measurement, measure_posture  # noqa: E402
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
        raise RuntimeError(f"無法開啟相機 index={args.camera}")
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
        raise RuntimeError(f"無法開啟相機 index={args.camera}")

    _discard_frames(cap, args.discard)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("讀取相機影格失敗")
    print(f"載入模型（{args.precision} / {args.device}）", flush=True)
    engine.warmup(split_merged_frame(frame, args.vertical_split, args.swap_lr)[0])
    print("開始量測，Ctrl-C 結束", flush=True)
    try:
        while True:
            try:
                measurement, _, _ = _measure_once(engine, calib, cap, args)
            except RuntimeError as exc:
                print(f"\r{exc}", end="", flush=True)
                continue
            ca = measurement.theta_ca_deg
            sym = measurement.theta_sym_deg
            distance = measurement.reference_depth_mm
            precision = measurement.theta_ca_precision_deg
            # 誤差也一起印，調整座位時可以直接看著這個數字找位置
            print(
                f"\rθ_CA {'  —  ' if ca is None else f'{ca:+6.1f}°'}"
                f"   θ_sym {'  —  ' if sym is None else f'{sym:+6.1f}°'}"
                f"   距離 {'—' if distance is None else f'{distance:4.0f}mm'}"
                f"   誤差 {'—' if precision is None else f'±{precision:4.1f}°'}"
                f"   共同點 {measurement.shared_count:2d}   ",
                end="", flush=True,
            )
    except KeyboardInterrupt:
        print()
    finally:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="端到端坐姿量測")
    sub = parser.add_subparsers(dest="mode", required=True)

    for name, help_text in (("once", "量測一次並印出完整診斷"), ("live", "持續量測，只印角度")):
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
        if name == "once":
            p.add_argument("--all-keypoints", action="store_true",
                           help="印出全部18點，不只角度用到的那幾個")

    args = parser.parse_args()
    if not args.calibration.is_file():
        raise SystemExit(
            f"找不到標定檔 {args.calibration}。先執行：\n"
            f"  python -m src.calibration.cli stereo --charuco ..."
        )
    # live 沒有這個選項，補一個預設值讓兩條路徑共用同一個 args
    args.all_keypoints = getattr(args, "all_keypoints", False)

    if args.mode == "once":
        _run_once(args)
    else:
        _run_live(args)


if __name__ == "__main__":
    main()
