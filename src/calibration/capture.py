from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from .chessboard import ChessboardSpec, find_corners


def _open_camera(index: int) -> cv2.VideoCapture:
    if sys.platform.startswith("linux"):
        return cv2.VideoCapture(index, cv2.CAP_V4L2)
    return cv2.VideoCapture(index)


def _hold_until_keypress(*frames_by_window: tuple[str, np.ndarray]) -> None:
    for window_name, frame in frames_by_window:
        done = frame.copy()
        cv2.putText(
            done, "拍攝完成", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
        )
        cv2.imshow(window_name, done)
    cv2.waitKey(0)


def _draw_feedback(
    frame: np.ndarray,
    spec: ChessboardSpec,
    corners: np.ndarray | None,
    saved_count: int,
    target_count: int,
) -> None:
    if corners is not None:
        cv2.drawChessboardCorners(frame, (spec.cols, spec.rows), corners, True)
    status = f"已存 {saved_count}/{target_count}  [SPACE]拍攝  [ESC]結束"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def capture_mono(
    camera_index: int, out_dir: Path, spec: ChessboardSpec, target_count: int
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = _open_camera(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟相機 index={camera_index}")

    saved = len(list(out_dir.glob("*.png")))
    cancelled = False
    try:
        while saved < target_count:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("讀取相機影格失敗")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = find_corners(gray, spec)
            display = frame.copy()
            _draw_feedback(display, spec, corners, saved, target_count)
            cv2.imshow("mono capture", display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cancelled = True
                break
            if key == 32 and corners is not None:
                saved += 1
                cv2.imwrite(str(out_dir / f"frame_{saved:04d}.png"), frame)
                print(f"已存 {saved}/{target_count}")

        print(f"拍攝結束，共存 {saved} 張於 {out_dir}")
        if not cancelled:
            _hold_until_keypress(("mono capture", display))
    finally:
        cap.release()
        cv2.destroyAllWindows()


def capture_stereo(
    left_index: int,
    right_index: int,
    left_out: Path,
    right_out: Path,
    spec: ChessboardSpec,
    target_count: int,
) -> None:
    left_out.mkdir(parents=True, exist_ok=True)
    right_out.mkdir(parents=True, exist_ok=True)
    cap_l = _open_camera(left_index)
    cap_r = _open_camera(right_index)
    if not cap_l.isOpened() or not cap_r.isOpened():
        raise RuntimeError(f"無法開啟雙目相機 index=({left_index}, {right_index})")

    saved = len(list(left_out.glob("*.png")))
    cancelled = False
    try:
        while saved < target_count:
            ok_l, frame_l = cap_l.read()
            ok_r, frame_r = cap_r.read()
            if not (ok_l and ok_r):
                raise RuntimeError("讀取雙目相機影格失敗")

            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
            corners_l = find_corners(gray_l, spec)
            corners_r = find_corners(gray_r, spec)

            disp_l, disp_r = frame_l.copy(), frame_r.copy()
            _draw_feedback(disp_l, spec, corners_l, saved, target_count)
            _draw_feedback(disp_r, spec, corners_r, saved, target_count)
            cv2.imshow("stereo left", disp_l)
            cv2.imshow("stereo right", disp_r)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cancelled = True
                break
            if key == 32 and corners_l is not None and corners_r is not None:
                saved += 1
                name = f"frame_{saved:04d}.png"
                cv2.imwrite(str(left_out / name), frame_l)
                cv2.imwrite(str(right_out / name), frame_r)
                print(f"已存 {saved}/{target_count}")

        print(f"拍攝結束，共存 {saved} 組於 {left_out} / {right_out}")
        if not cancelled:
            _hold_until_keypress(("stereo left", disp_l), ("stereo right", disp_r))
    finally:
        cap_l.release()
        cap_r.release()
        cv2.destroyAllWindows()


def capture_stereo_single_device(
    camera_index: int,
    left_out: Path,
    right_out: Path,
    spec: ChessboardSpec,
    target_count: int,
    vertical_split: bool = False,
    swap_lr: bool = False,
) -> None:
    """給左右眼合併輸出成同一張畫面的雙目相機用（單一USB裝置、單一video node）。

    水平並排（預設）：畫面左半=左眼、右半=右眼，對半寬度切開。
    vertical_split=True 則改成上下切開。
    """
    left_out.mkdir(parents=True, exist_ok=True)
    right_out.mkdir(parents=True, exist_ok=True)
    cap = _open_camera(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟相機 index={camera_index}")

    def split(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if vertical_split:
            h = frame.shape[0]
            a, b = frame[: h // 2], frame[h // 2 :]
        else:
            w = frame.shape[1]
            a, b = frame[:, : w // 2], frame[:, w // 2 :]
        return (b, a) if swap_lr else (a, b)

    saved = len(list(left_out.glob("*.png")))
    cancelled = False
    try:
        while saved < target_count:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("讀取相機影格失敗")
            frame_l, frame_r = split(frame)

            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
            corners_l = find_corners(gray_l, spec)
            corners_r = find_corners(gray_r, spec)

            disp_l, disp_r = frame_l.copy(), frame_r.copy()
            _draw_feedback(disp_l, spec, corners_l, saved, target_count)
            _draw_feedback(disp_r, spec, corners_r, saved, target_count)
            cv2.imshow("stereo left", disp_l)
            cv2.imshow("stereo right", disp_r)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cancelled = True
                break
            if key == 32 and corners_l is not None and corners_r is not None:
                saved += 1
                name = f"frame_{saved:04d}.png"
                cv2.imwrite(str(left_out / name), frame_l)
                cv2.imwrite(str(right_out / name), frame_r)
                print(f"已存 {saved}/{target_count}")

        print(f"拍攝結束，共存 {saved} 組於 {left_out} / {right_out}")
        if not cancelled:
            _hold_until_keypress(("stereo left", disp_l), ("stereo right", disp_r))
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="標定影像互動式擷取")
    sub = parser.add_subparsers(dest="mode", required=True)

    mono_p = sub.add_parser("mono", help="擷取單眼標定影像")
    mono_p.add_argument("--camera", type=int, default=0)
    mono_p.add_argument(
        "--out", type=Path, default=Path("data/calibration_images/front")
    )
    mono_p.add_argument("--cols", type=int, default=9)
    mono_p.add_argument("--rows", type=int, default=6)
    mono_p.add_argument("--square-size-mm", type=float, default=25.0)
    mono_p.add_argument("--target-count", type=int, default=40)

    stereo_p = sub.add_parser("stereo", help="擷取雙目標定影像")
    stereo_p.add_argument(
        "--single-device",
        action="store_true",
        help="雙目模組是單一USB裝置、左右眼合併在同一畫面（用--left-camera指定該裝置index）",
    )
    stereo_p.add_argument("--left-camera", type=int, default=1)
    stereo_p.add_argument("--right-camera", type=int, default=2)
    stereo_p.add_argument(
        "--vertical-split",
        action="store_true",
        help="合併畫面變上下切（預設左右切），僅搭配--single-device使用",
    )
    stereo_p.add_argument(
        "--swap-lr", action="store_true", help="左右眼相反時加這個對調"
    )
    stereo_p.add_argument(
        "--left-out", type=Path, default=Path("data/calibration_images/stereo_left")
    )
    stereo_p.add_argument(
        "--right-out", type=Path, default=Path("data/calibration_images/stereo_right")
    )
    stereo_p.add_argument("--cols", type=int, default=9)
    stereo_p.add_argument("--rows", type=int, default=6)
    stereo_p.add_argument("--square-size-mm", type=float, default=25.0)
    stereo_p.add_argument("--target-count", type=int, default=20)

    args = parser.parse_args()

    if args.mode == "mono":
        spec = ChessboardSpec(
            cols=args.cols, rows=args.rows, square_size_mm=args.square_size_mm
        )
        capture_mono(args.camera, args.out, spec, args.target_count)
    else:
        spec = ChessboardSpec(
            cols=args.cols, rows=args.rows, square_size_mm=args.square_size_mm
        )
        if args.single_device:
            capture_stereo_single_device(
                args.left_camera,
                args.left_out,
                args.right_out,
                spec,
                args.target_count,
                vertical_split=args.vertical_split,
                swap_lr=args.swap_lr,
            )
        else:
            capture_stereo(
                args.left_camera,
                args.right_camera,
                args.left_out,
                args.right_out,
                spec,
                args.target_count,
            )


if __name__ == "__main__":
    main()
