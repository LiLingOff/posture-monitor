from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..calibration.capture import _open_camera
from .benchmark import compare_precision_rmse, measure_latency, measure_sequential_multi_camera
from .engine import TrtPoseModelPaths


def _grab_frames(camera_index: int, count: int) -> list[np.ndarray]:
    cap = _open_camera(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟相機 index={camera_index}")
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


def _split_stereo(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = frame.shape[1]
    return frame[:, : w // 2], frame[:, w // 2 :]


def _run_benchmark(args: argparse.Namespace) -> None:
    from .engine import TrtPoseEngine

    paths = TrtPoseModelPaths(args.checkpoint, args.engine_cache, args.topology)
    engine = TrtPoseEngine(paths, precision=args.precision)

    front_frames = _grab_frames(args.front_camera, args.warmup + args.frames)
    single = measure_latency(engine, front_frames, warmup=args.warmup)
    print(f"單相機（正面）precision={args.precision}：平均{single.mean_ms:.2f}ms/frame，{single.fps:.1f}fps")

    stereo_frames = _grab_frames(args.stereo_camera, args.warmup + args.frames)
    frame_sets = []
    for front, stereo in zip(front_frames, stereo_frames):
        if args.single_device:
            left, right = _split_stereo(stereo)
            frame_sets.append([front, left, right])
        else:
            frame_sets.append([front, stereo])

    multi = measure_sequential_multi_camera(engine, frame_sets, warmup=args.warmup)
    print(f"多相機循序推論 precision={args.precision}：平均{multi.mean_ms:.2f}ms/frame，{multi.fps:.1f}fps")
    print("（文件基準：三相機循序推論約333ms/frame，優化前）")


def _run_compare_precision(args: argparse.Namespace) -> None:
    from .engine import TrtPoseEngine

    paths = TrtPoseModelPaths(args.checkpoint, args.engine_cache, args.topology)
    fp32_engine = TrtPoseEngine(paths, precision="fp32")
    fp16_engine = TrtPoseEngine(paths, precision="fp16")

    frames = _grab_frames(args.camera, args.samples)
    fp32_results, fp16_results = [], []
    skipped = 0
    for frame in frames:
        detected_fp32 = fp32_engine.infer(frame)
        detected_fp16 = fp16_engine.infer(frame)
        # 沒偵測到人時infer回傳空list，直接取[0]會IndexError
        if not detected_fp32 or not detected_fp16:
            skipped += 1
            continue
        fp32_results.append(detected_fp32[0])
        fp16_results.append(detected_fp16[0])

    if skipped:
        print(f"有{skipped}/{len(frames)}張影格沒偵測到人，已略過")
    if not fp32_results:
        raise RuntimeError("所有取樣影格都沒偵測到人，無法比對——確認受試者有在畫面內、光線是否足夠")

    mean_rmse, passed = compare_precision_rmse(fp32_results, fp16_results, args.rmse_threshold_px)
    chosen = "fp16" if passed else "fp32"
    print(f"平均RMSE={mean_rmse:.4f}px，建議使用precision={chosen}")


def main() -> None:
    parser = argparse.ArgumentParser(description="人體關鍵點偵測效能測試與精度比對")
    sub = parser.add_subparsers(dest="mode", required=True)

    bench_p = sub.add_parser("benchmark", help="單相機/多相機延遲基準測試")
    bench_p.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    bench_p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/pose_models/resnet18_baseline_att_224x224_A_epoch_249.pth"),
    )
    bench_p.add_argument("--engine-cache", type=Path, default=Path("data/pose_models/resnet18_fp16.pth"))
    bench_p.add_argument("--topology", type=Path, default=Path("data/pose_models/human_pose.json"))
    bench_p.add_argument("--front-camera", type=int, default=0)
    bench_p.add_argument("--stereo-camera", type=int, default=1)
    bench_p.add_argument("--single-device", action="store_true")
    bench_p.add_argument("--frames", type=int, default=60)
    bench_p.add_argument("--warmup", type=int, default=5)

    compare_p = sub.add_parser("compare-precision", help="比對FP32/FP16關鍵點RMSE")
    compare_p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/pose_models/resnet18_baseline_att_224x224_A_epoch_249.pth"),
    )
    compare_p.add_argument("--engine-cache", type=Path, default=Path("data/pose_models/resnet18_fp16.pth"))
    compare_p.add_argument("--topology", type=Path, default=Path("data/pose_models/human_pose.json"))
    compare_p.add_argument("--camera", type=int, default=0)
    compare_p.add_argument("--samples", type=int, default=30)
    compare_p.add_argument("--rmse-threshold-px", type=float, default=3.0)

    args = parser.parse_args()

    if args.mode == "benchmark":
        _run_benchmark(args)
    else:
        _run_compare_precision(args)


if __name__ == "__main__":
    main()
