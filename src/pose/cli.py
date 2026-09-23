from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..calibration.capture import (_open_camera, describe_camera_open_failure,
                                   split_merged_frame)
from .benchmark import compare_precision_rmse, measure_latency, measure_sequential_multi_camera
from .engine import LightweightOpenPoseModelPaths


def _grab_frames(
    camera_index: int, count: int, width: int | None = None, height: int | None = None
) -> list[np.ndarray]:
    cap = _open_camera(camera_index, width, height)
    if not cap.isOpened():
        raise RuntimeError(describe_camera_open_failure(camera_index))
    frames = []
    try:
        while len(frames) < count:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("讀取相機影格失敗")
            frames.append(frame)
    finally:
        cap.release()
    return frames


def _run_benchmark(args: argparse.Namespace) -> None:
    from .engine import LightweightOpenPoseEngine

    paths = LightweightOpenPoseModelPaths(args.checkpoint, args.engine_cache, args.repo_dir)
    engine = LightweightOpenPoseEngine(paths, precision=args.precision, device=args.device)

    front_frames = _grab_frames(args.front_camera, args.warmup + args.frames, args.width, args.height)
    single = measure_latency(engine, front_frames, warmup=args.warmup)
    print(f"單相機（正面）precision={args.precision}：平均{single.mean_ms:.2f}ms/frame，{single.fps:.1f}fps")

    stereo_frames = _grab_frames(args.stereo_camera, args.warmup + args.frames, args.width, args.height)
    frame_sets = []
    for front, stereo in zip(front_frames, stereo_frames):
        if args.single_device:
            left, right = split_merged_frame(stereo, args.vertical_split, args.swap_lr)
            frame_sets.append([front, left, right])
        else:
            frame_sets.append([front, stereo])

    multi = measure_sequential_multi_camera(engine, frame_sets, warmup=args.warmup)
    print(f"多相機循序推論 precision={args.precision}：平均{multi.mean_ms:.2f}ms/frame，{multi.fps:.1f}fps")
    print("（文件基準：三相機循序推論約333ms/frame，優化前）")


def _run_compare_precision(args: argparse.Namespace) -> None:
    from .engine import LightweightOpenPoseEngine

    paths = LightweightOpenPoseModelPaths(args.checkpoint, args.engine_cache, args.repo_dir)
    fp32_engine = LightweightOpenPoseEngine(paths, precision="fp32", device=args.device)
    fp16_engine = LightweightOpenPoseEngine(paths, precision="fp16")

    frames = _grab_frames(args.camera, args.samples, args.width, args.height)
    fp32_results, fp16_results = [], []
    skipped = 0
    for frame in frames:
        detected_fp32 = fp32_engine.infer(frame)
        detected_fp16 = fp16_engine.infer(frame)
        # 沒有偵測到人時infer回傳空list，直接取[0]會產生IndexError
        if not detected_fp32 or not detected_fp16:
            skipped += 1
            continue
        fp32_results.append(detected_fp32[0])
        fp16_results.append(detected_fp16[0])

    if skipped:
        print(f"有{skipped}/{len(frames)}張影格沒有偵測到人，已略過")
    if not fp32_results:
        raise RuntimeError("所有取樣影格都沒有偵測到人，無法比對——請確認受試者位於畫面內、光線是否足夠")

    mean_rmse, passed = compare_precision_rmse(fp32_results, fp16_results, args.rmse_threshold_px)
    chosen = "fp16" if passed else "fp32"
    print(f"平均RMSE={mean_rmse:.4f}px，建議使用precision={chosen}")


def main() -> None:
    parser = argparse.ArgumentParser(description="人體關鍵點偵測效能測試與精度比對")
    sub = parser.add_subparsers(dest="mode", required=True)

    bench_p = sub.add_parser("benchmark", help="單相機/多相機延遲基準測試")
    bench_p.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    bench_p.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
        help="cuda 或 cpu。Jetson 的 PyTorch 還沒弄好時，先用 cpu 搭配 --precision fp32 驗證流程——這個模型本來就是為 CPU 設計的")
    bench_p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/pose_models/checkpoint_iter_370000.pth"),
    )
    bench_p.add_argument("--engine-cache", type=Path, default=Path("data/pose_models/lightweight_openpose_fp16.pth"))
    bench_p.add_argument("--repo-dir", type=Path, default=Path("third_party/lightweight-human-pose-estimation.pytorch"),
        help="clone下來的lightweight-human-pose-estimation.pytorch目錄（該repo沒有setup.py，不能pip安裝）")
    bench_p.add_argument("--front-camera", type=int, default=0)
    bench_p.add_argument("--stereo-camera", type=int, default=1)
    bench_p.add_argument("--single-device", action="store_true",
        help="雙目為單一裝置、左右眼合併在同一畫面")
    bench_p.add_argument("--vertical-split", action="store_true",
        help="合併畫面為上下拼接而非左右並排，需與拍攝時一致")
    bench_p.add_argument("--swap-lr", action="store_true",
        help="左右眼顛倒時加上這個對調，需與拍攝時一致")
    bench_p.add_argument("--frames", type=int, default=60)
    bench_p.add_argument("--warmup", type=int, default=5)
    bench_p.add_argument("--width", type=int, default=None,
        help="相機解析度寬度；雙目並排模式通常只在特定寬解析度下才有")
    bench_p.add_argument("--height", type=int, default=None, help="相機解析度高度")

    compare_p = sub.add_parser("compare-precision", help="比對FP32/FP16關鍵點RMSE")
    compare_p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/pose_models/checkpoint_iter_370000.pth"),
    )
    compare_p.add_argument("--engine-cache", type=Path, default=Path("data/pose_models/lightweight_openpose_fp16.pth"))
    compare_p.add_argument("--repo-dir", type=Path, default=Path("third_party/lightweight-human-pose-estimation.pytorch"),
        help="clone下來的lightweight-human-pose-estimation.pytorch目錄（該repo沒有setup.py，不能pip安裝）")
    compare_p.add_argument("--camera", type=int, default=0)
    compare_p.add_argument("--samples", type=int, default=30)
    compare_p.add_argument("--rmse-threshold-px", type=float, default=3.0)
    compare_p.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
        help="fp32 那一側要用的裝置；fp16 一定是 cuda")
    compare_p.add_argument("--width", type=int, default=None,
        help="相機解析度寬度；雙目並排模式通常只在特定寬解析度下才有")
    compare_p.add_argument("--height", type=int, default=None, help="相機解析度高度")

    args = parser.parse_args()

    if args.mode == "benchmark":
        _run_benchmark(args)
    else:
        _run_compare_precision(args)


if __name__ == "__main__":
    main()
