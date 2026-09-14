from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from .charuco import CharucoBoardSpec, detect_charuco
from .chessboard import ChessboardSpec, find_corners

_MIN_SHARED_CHARUCO_CORNERS = 6


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


def _draw_charuco_feedback(
    frame: np.ndarray,
    detected: tuple[np.ndarray, np.ndarray] | None,
    saved_count: int,
    target_count: int,
) -> None:
    if detected is not None:
        corners, ids = detected
        cv2.aruco.drawDetectedCornersCharuco(frame, corners, ids)
        count_text = f"偵測到{len(ids)}點"
    else:
        count_text = "未偵測到角點"
    status = f"{count_text}  已存{saved_count}/{target_count}  [SPACE]拍攝  [ESC]結束"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def capture_mono_charuco(
    camera_index: int,
    out_dir: Path,
    board_spec: CharucoBoardSpec,
    target_count: int,
    min_corners: int = _MIN_SHARED_CHARUCO_CORNERS,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    board = board_spec.build_board()
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
            detected = detect_charuco(gray, board, min_corners=1)
            display = frame.copy()
            _draw_charuco_feedback(display, detected, saved, target_count)
            cv2.imshow("mono capture (charuco)", display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cancelled = True
                break
            if key == 32 and detected is not None and len(detected[1]) >= min_corners:
                saved += 1
                cv2.imwrite(str(out_dir / f"frame_{saved:04d}.png"), frame)
                print(f"已存 {saved}/{target_count}（{len(detected[1])}個角點）")

        print(f"拍攝結束，共存 {saved} 張於 {out_dir}")
        if not cancelled:
            _hold_until_keypress(("mono capture (charuco)", display))
    finally:
        cap.release()
        cv2.destroyAllWindows()


def _shared_corner_count(
    det_l: tuple[np.ndarray, np.ndarray] | None, det_r: tuple[np.ndarray, np.ndarray] | None
) -> int:
    if det_l is None or det_r is None:
        return 0
    ids_l = set(det_l[1].flatten().tolist())
    ids_r = set(det_r[1].flatten().tolist())
    return len(ids_l & ids_r)


def _draw_stereo_charuco_feedback(
    disp_l: np.ndarray,
    disp_r: np.ndarray,
    det_l: tuple[np.ndarray, np.ndarray] | None,
    det_r: tuple[np.ndarray, np.ndarray] | None,
    shared: int,
    min_shared_corners: int,
    saved: int,
    target_count: int,
) -> None:
    _draw_charuco_feedback(disp_l, det_l, saved, target_count)
    _draw_charuco_feedback(disp_r, det_r, saved, target_count)
    shared_text = f"共同角點：{shared}（需>={min_shared_corners}）"
    color = (0, 200, 0) if shared >= min_shared_corners else (0, 140, 255)
    cv2.putText(disp_l, shared_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def capture_stereo_charuco(
    left_index: int,
    right_index: int,
    left_out: Path,
    right_out: Path,
    board_spec: CharucoBoardSpec,
    target_count: int,
    min_shared_corners: int = _MIN_SHARED_CHARUCO_CORNERS,
) -> None:
    """雙目兩顆鏡頭各自視野重疊區域小、拍不到完整board時用這個——
    不需要整塊board同時入鏡，只要左右畫面有足夠共同角點就能存。
    """
    left_out.mkdir(parents=True, exist_ok=True)
    right_out.mkdir(parents=True, exist_ok=True)
    board = board_spec.build_board()
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
            det_l = detect_charuco(gray_l, board, min_corners=1)
            det_r = detect_charuco(gray_r, board, min_corners=1)
            shared = _shared_corner_count(det_l, det_r)

            disp_l, disp_r = frame_l.copy(), frame_r.copy()
            _draw_stereo_charuco_feedback(disp_l, disp_r, det_l, det_r, shared, min_shared_corners, saved, target_count)
            cv2.imshow("stereo left", disp_l)
            cv2.imshow("stereo right", disp_r)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cancelled = True
                break
            if key == 32 and shared >= min_shared_corners:
                saved += 1
                name = f"frame_{saved:04d}.png"
                cv2.imwrite(str(left_out / name), frame_l)
                cv2.imwrite(str(right_out / name), frame_r)
                print(f"已存 {saved}/{target_count}（共同角點{shared}個）")

        print(f"拍攝結束，共存 {saved} 組於 {left_out} / {right_out}")
        if not cancelled:
            _hold_until_keypress(("stereo left", disp_l), ("stereo right", disp_r))
    finally:
        cap_l.release()
        cap_r.release()
        cv2.destroyAllWindows()


def capture_stereo_charuco_single_device(
    camera_index: int,
    left_out: Path,
    right_out: Path,
    board_spec: CharucoBoardSpec,
    target_count: int,
    vertical_split: bool = False,
    swap_lr: bool = False,
    min_shared_corners: int = _MIN_SHARED_CHARUCO_CORNERS,
) -> None:
    left_out.mkdir(parents=True, exist_ok=True)
    right_out.mkdir(parents=True, exist_ok=True)
    board = board_spec.build_board()
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
            det_l = detect_charuco(gray_l, board, min_corners=1)
            det_r = detect_charuco(gray_r, board, min_corners=1)
            shared = _shared_corner_count(det_l, det_r)

            disp_l, disp_r = frame_l.copy(), frame_r.copy()
            _draw_stereo_charuco_feedback(disp_l, disp_r, det_l, det_r, shared, min_shared_corners, saved, target_count)
            cv2.imshow("stereo left", disp_l)
            cv2.imshow("stereo right", disp_r)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cancelled = True
                break
            if key == 32 and shared >= min_shared_corners:
                saved += 1
                name = f"frame_{saved:04d}.png"
                cv2.imwrite(str(left_out / name), frame_l)
                cv2.imwrite(str(right_out / name), frame_r)
                print(f"已存 {saved}/{target_count}（共同角點{shared}個）")

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
    mono_p.add_argument(
        "--charuco", action="store_true", help="用ChArUco板取代一般棋盤格（容許畫面只看到板子一部分）"
    )
    mono_p.add_argument("--squares-x", type=int, default=10, help="ChArUco板橫向方格數")
    mono_p.add_argument("--squares-y", type=int, default=8, help="ChArUco板縱向方格數")
    mono_p.add_argument("--marker-size-mm", type=float, default=18.0, help="ChArUco標記邊長")
    mono_p.add_argument("--dictionary", type=str, default="DICT_5X5_100")
    mono_p.add_argument(
        "--legacy-pattern", action="store_true", help="現成板子（如AndyMark）偵測不到就加這個"
    )
    mono_p.add_argument("--min-corners", type=int, default=_MIN_SHARED_CHARUCO_CORNERS)

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
    stereo_p.add_argument(
        "--charuco",
        action="store_true",
        help="用ChArUco板取代一般棋盤格——雙目兩顆鏡頭視野重疊區域小、拍不到完整棋盤格時用這個",
    )
    stereo_p.add_argument("--squares-x", type=int, default=10, help="ChArUco板橫向方格數")
    stereo_p.add_argument("--squares-y", type=int, default=8, help="ChArUco板縱向方格數")
    stereo_p.add_argument("--marker-size-mm", type=float, default=18.0, help="ChArUco標記邊長")
    stereo_p.add_argument("--dictionary", type=str, default="DICT_5X5_100")
    stereo_p.add_argument(
        "--legacy-pattern", action="store_true", help="現成板子（如AndyMark）偵測不到就加這個"
    )
    stereo_p.add_argument(
        "--min-shared-corners",
        type=int,
        default=_MIN_SHARED_CHARUCO_CORNERS,
        help="左右畫面至少要有幾個共同角點才允許存檔",
    )

    board_p = sub.add_parser("board", help="產生ChArUco板圖檔供列印")
    board_p.add_argument("--out", type=Path, default=Path("data/charuco_board.png"))
    board_p.add_argument("--squares-x", type=int, default=10)
    board_p.add_argument("--squares-y", type=int, default=8)
    board_p.add_argument("--square-size-mm", type=float, default=25.0)
    board_p.add_argument("--marker-size-mm", type=float, default=18.0)
    board_p.add_argument("--dictionary", type=str, default="DICT_5X5_100")
    board_p.add_argument(
        "--legacy-pattern", action="store_true", help="現成板子（如AndyMark）偵測不到就加這個"
    )
    board_p.add_argument("--pixels-per-square", type=int, default=80)

    args = parser.parse_args()

    if args.mode == "board":
        from .charuco import save_board_image

        spec = CharucoBoardSpec(
            squares_x=args.squares_x,
            squares_y=args.squares_y,
            square_size_mm=args.square_size_mm,
            marker_size_mm=args.marker_size_mm,
            dictionary_name=args.dictionary,
            legacy_pattern=args.legacy_pattern,
        )
        save_board_image(spec, args.out, pixels_per_square=args.pixels_per_square)
        print(f"已輸出board圖檔至 {args.out}，請按實際尺寸列印（不要自動縮放/符合頁面）")
        return

    if args.mode == "mono":
        if args.charuco:
            board_spec = CharucoBoardSpec(
                squares_x=args.squares_x,
                squares_y=args.squares_y,
                square_size_mm=args.square_size_mm,
                marker_size_mm=args.marker_size_mm,
                dictionary_name=args.dictionary,
                legacy_pattern=args.legacy_pattern,
            )
            capture_mono_charuco(
                args.camera, args.out, board_spec, args.target_count, min_corners=args.min_corners
            )
        else:
            spec = ChessboardSpec(
                cols=args.cols, rows=args.rows, square_size_mm=args.square_size_mm
            )
            capture_mono(args.camera, args.out, spec, args.target_count)
    else:
        if args.charuco:
            board_spec = CharucoBoardSpec(
                squares_x=args.squares_x,
                squares_y=args.squares_y,
                square_size_mm=args.square_size_mm,
                marker_size_mm=args.marker_size_mm,
                dictionary_name=args.dictionary,
                legacy_pattern=args.legacy_pattern,
            )
            if args.single_device:
                capture_stereo_charuco_single_device(
                    args.left_camera,
                    args.left_out,
                    args.right_out,
                    board_spec,
                    args.target_count,
                    vertical_split=args.vertical_split,
                    swap_lr=args.swap_lr,
                    min_shared_corners=args.min_shared_corners,
                )
            else:
                capture_stereo_charuco(
                    args.left_camera,
                    args.right_camera,
                    args.left_out,
                    args.right_out,
                    board_spec,
                    args.target_count,
                    min_shared_corners=args.min_shared_corners,
                )
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
