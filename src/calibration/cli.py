from __future__ import annotations

import argparse
from pathlib import Path

from .chessboard import ChessboardSpec
from .mono_calibration import calibrate_mono
from .stereo_calibration import calibrate_stereo


def main() -> None:
    parser = argparse.ArgumentParser(description="相機標定計算")
    sub = parser.add_subparsers(dest="mode", required=True)

    mono_p = sub.add_parser("mono", help="正面相機張氏標定")
    mono_p.add_argument(
        "--images", type=Path, default=Path("data/calibration_images/front")
    )
    mono_p.add_argument(
        "--out", type=Path, default=Path("data/calibration_output/front.npz")
    )
    mono_p.add_argument("--cols", type=int, default=9)
    mono_p.add_argument("--rows", type=int, default=6)
    mono_p.add_argument("--square-size-mm", type=float, default=25.0)
    mono_p.add_argument("--target-error-px", type=float, default=0.3)

    stereo_p = sub.add_parser("stereo", help="前側雙目標定")
    stereo_p.add_argument(
        "--left-images", type=Path, default=Path("data/calibration_images/stereo_left")
    )
    stereo_p.add_argument(
        "--right-images",
        type=Path,
        default=Path("data/calibration_images/stereo_right"),
    )
    stereo_p.add_argument(
        "--out", type=Path, default=Path("data/calibration_output/stereo.npz")
    )
    stereo_p.add_argument("--cols", type=int, default=9)
    stereo_p.add_argument("--rows", type=int, default=6)
    stereo_p.add_argument("--square-size-mm", type=float, default=25.0)
    stereo_p.add_argument("--target-error-px", type=float, default=0.5)

    args = parser.parse_args()
    spec = ChessboardSpec(
        cols=args.cols, rows=args.rows, square_size_mm=args.square_size_mm
    )

    if args.mode == "mono":
        result = calibrate_mono(args.images, spec, args.target_error_px)
        result.save(args.out)
        print(f"單眼標定完成：RMS重投影誤差 = {result.rms_reprojection_error:.4f}px")
        print(f"使用影像數：{len(result.used_images)}，已存至 {args.out}")
    else:
        result = calibrate_stereo(
            args.left_images, args.right_images, spec, args.target_error_px
        )
        result.save(args.out)
        print(f"雙目標定完成：RMS重投影誤差 = {result.rms_error:.4f}px")
        print(f"基線長度 = {result.baseline_mm:.2f}mm，已存至 {args.out}")


if __name__ == "__main__":
    main()
