from __future__ import annotations

import argparse
from pathlib import Path

from .charuco import CharucoBoardSpec
from .chessboard import ChessboardSpec
from .mono_calibration import calibrate_mono, calibrate_mono_charuco
from .stereo_calibration import calibrate_stereo, calibrate_stereo_charuco


def _add_chessboard_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cols", type=int, default=9)
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--square-size-mm", type=float, default=25.0)


def _add_charuco_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--charuco", action="store_true", help="影像是用ChArUco板拍的")
    p.add_argument("--squares-x", type=int, default=10)
    p.add_argument("--squares-y", type=int, default=8)
    p.add_argument("--marker-size-mm", type=float, default=18.0)
    p.add_argument("--dictionary", type=str, default="DICT_5X5_100")


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
    _add_chessboard_args(mono_p)
    _add_charuco_args(mono_p)
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
    _add_chessboard_args(stereo_p)
    _add_charuco_args(stereo_p)
    stereo_p.add_argument("--target-error-px", type=float, default=0.5)

    args = parser.parse_args()

    if args.mode == "mono":
        if args.charuco:
            board_spec = CharucoBoardSpec(
                squares_x=args.squares_x,
                squares_y=args.squares_y,
                square_size_mm=args.square_size_mm,
                marker_size_mm=args.marker_size_mm,
                dictionary_name=args.dictionary,
            )
            result = calibrate_mono_charuco(args.images, board_spec, args.target_error_px)
        else:
            spec = ChessboardSpec(
                cols=args.cols, rows=args.rows, square_size_mm=args.square_size_mm
            )
            result = calibrate_mono(args.images, spec, args.target_error_px)
        result.save(args.out)
        print(f"單眼標定完成：RMS重投影誤差 = {result.rms_reprojection_error:.4f}px")
        print(f"使用影像數：{len(result.used_images)}，已存至 {args.out}")
    else:
        if args.charuco:
            board_spec = CharucoBoardSpec(
                squares_x=args.squares_x,
                squares_y=args.squares_y,
                square_size_mm=args.square_size_mm,
                marker_size_mm=args.marker_size_mm,
                dictionary_name=args.dictionary,
            )
            result = calibrate_stereo_charuco(
                args.left_images, args.right_images, board_spec, args.target_error_px
            )
        else:
            spec = ChessboardSpec(
                cols=args.cols, rows=args.rows, square_size_mm=args.square_size_mm
            )
            result = calibrate_stereo(
                args.left_images, args.right_images, spec, args.target_error_px
            )
        result.save(args.out)
        print(f"雙目標定完成：RMS重投影誤差 = {result.rms_error:.4f}px")
        print(f"基線長度 = {result.baseline_mm:.2f}mm，已存至 {args.out}")


if __name__ == "__main__":
    main()
