from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .charuco import CharucoBoardSpec, detect_charuco
from .chessboard import ChessboardSpec, find_corners
from .device import (_warn_if_not_side_by_side, describe_camera_open_failure,
                     describe_stereo_open_failure, open_camera,
                     probe_resolutions, split_merged_frame)

_MIN_SHARED_CHARUCO_CORNERS = 6


def _next_frame_index(*out_dirs: Path) -> int:
    """回傳下一個可用的檔名編號：現有檔名的最大編號加一。

    用檔案數量當編號會在編號不連續時撞名。刪掉沒對焦的那張再重跑補拍是很自然的操作，
    此時數量比最大編號小，新檔會蓋掉既有影像，而且印出的張數比實際檔案數多。
    雙目要左右一起看，確保同一個編號在兩邊都還沒被用掉。
    """
    max_index = 0
    for out_dir in out_dirs:
        for path in out_dir.glob("frame_*.png"):
            suffix = path.stem.rsplit("_", 1)[-1]
            if suffix.isdigit():
                max_index = max(max_index, int(suffix))
    return max_index + 1


def _existing_image_size(out_dir: Path) -> tuple[int, int] | None:
    """回傳資料夾裡第一張影像的(寬, 高)，沒有影像則回傳None。"""
    for path in sorted(out_dir.glob("*.png")):
        img = cv2.imread(str(path))
        if img is not None:
            return (img.shape[1], img.shape[0])
    return None


def _reject_mismatched_existing_images(out_dir: Path, expected: tuple[int, int]) -> None:
    """既有影像的尺寸與這次要拍的不同時直接拒絕。

    換過解析度之後資料夾裡還留著舊影像是很容易發生的事，而張數檢查只數數量、
    看不出這件事。工具會說「已有N張，跳過拍攝」，接著標定就在錯誤解析度的影像上
    算出一組內參。內參綁定於解析度，用錯了不會有任何外顯症狀。
    """
    found = _existing_image_size(out_dir)
    if found is None or found == expected:
        return
    raise ValueError(
        f"{out_dir} 裡的影像是 {found[0]}x{found[1]}，這次要拍的是 {expected[0]}x{expected[1]}。\n"
        f"內參綁定於解析度，兩種混在一起標定出來的結果沒有意義。\n"
        f"請先清空資料夾再重拍：rm {out_dir}/*.png"
    )


def _already_complete(
    out_dir: Path, target_count: int, expected_size: tuple[int, int] | None = None
) -> bool:
    """資料夾已有足夠張數時回傳True，直接跳過，不必開啟相機。

    沒有這個檢查的話，拍攝迴圈一次都不會執行，後面要顯示最後一幀的變數
    就從未被指派，會以UnboundLocalError結束。

    expected_size有給的話會先核對既有影像的尺寸，對不上直接拒絕，
    只數張數的話，換過解析度卻沒清資料夾就會沿用到舊影像。
    """
    if expected_size is not None:
        _reject_mismatched_existing_images(out_dir, expected_size)
    saved = len(list(out_dir.glob("*.png")))
    if saved >= target_count:
        print(f"{out_dir} 已有{saved}張（目標{target_count}），跳過拍攝；要重拍請先清空資料夾")
        return True
    return False


def _expected_eye_size(
    width: int | None, height: int | None, vertical_split: bool
) -> tuple[int, int] | None:
    """合併畫面切一半之後，單眼影像應有的尺寸。未指定解析度時回傳None。"""
    if width is None or height is None:
        return None
    return (width, height // 2) if vertical_split else (width // 2, height)


def _hold_until_keypress(*frames_by_window: tuple[str, np.ndarray]) -> None:
    for window_name, frame in frames_by_window:
        done = frame.copy()
        cv2.putText(
            done, "DONE - press any key", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
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
    # cv2.putText的Hershey字型只有ASCII，中文會全部顯示成問號，所以畫面上的文字一律使用英文
    status = f"saved {saved_count}/{target_count}   [SPACE] capture   [ESC] quit"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def capture_mono(
    camera_index: int,
    out_dir: Path,
    spec: ChessboardSpec,
    target_count: int,
    width: int | None = None,
    height: int | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if _already_complete(out_dir, target_count, (width, height) if width and height else None):
        return

    cap = open_camera(camera_index, width, height)
    if not cap.isOpened():
        raise RuntimeError(describe_camera_open_failure(camera_index))

    saved = len(list(out_dir.glob("*.png")))
    next_index = _next_frame_index(out_dir)
    cancelled = False
    try:
        while saved < target_count:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("讀取相機影格失敗")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = find_corners(gray, spec, fast_check=True)
            display = frame.copy()
            _draw_feedback(display, spec, corners, saved, target_count)
            cv2.imshow("mono capture", display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cancelled = True
                break
            if key == 32 and corners is not None:
                saved += 1
                cv2.imwrite(str(out_dir / f"frame_{next_index:04d}.png"), frame)
                next_index += 1
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
    width: int | None = None,
    height: int | None = None,
) -> None:
    left_out.mkdir(parents=True, exist_ok=True)
    right_out.mkdir(parents=True, exist_ok=True)
    if _already_complete(left_out, target_count, (width, height) if width and height else None):
        return

    cap_l = open_camera(left_index, width, height)
    cap_r = open_camera(right_index, width, height)
    if not cap_l.isOpened() or not cap_r.isOpened():
        raise RuntimeError(describe_stereo_open_failure(
            (left_index, cap_l.isOpened()), (right_index, cap_r.isOpened())
        ))

    saved = len(list(left_out.glob("*.png")))
    next_index = _next_frame_index(left_out, right_out)
    cancelled = False
    try:
        while saved < target_count:
            # 先兩邊都grab再各自retrieve。read()等於grab+retrieve，串著做的話
            # 兩張畫面可能差到一個影格間隔加上MJPG解碼時間；標定要求板子在左右影像
            # 位於同一個物理位置，中間有位移會直接污染外參R/T，而重投影誤差不會變差
            # （各相機自己的幾何仍然自洽），壞掉的剛好是這個專案最在意的相對關係。
            grabbed_l = cap_l.grab()
            grabbed_r = cap_r.grab()
            if not (grabbed_l and grabbed_r):
                raise RuntimeError("讀取雙目相機影格失敗")
            ok_l, frame_l = cap_l.retrieve()
            ok_r, frame_r = cap_r.retrieve()
            if not (ok_l and ok_r):
                raise RuntimeError("讀取雙目相機影格失敗")

            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
            corners_l = find_corners(gray_l, spec, fast_check=True)
            corners_r = find_corners(gray_r, spec, fast_check=True)

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
                name = f"frame_{next_index:04d}.png"
                next_index += 1
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
    width: int | None = None,
    height: int | None = None,
) -> None:
    """給左右眼合併輸出成同一張畫面的雙目相機用（單一USB裝置、單一video node）。

    水平並排（預設）：畫面左半=左眼、右半=右眼，對半寬度切開。
    vertical_split=True 則改成上下切開。
    """
    left_out.mkdir(parents=True, exist_ok=True)
    right_out.mkdir(parents=True, exist_ok=True)
    if _already_complete(left_out, target_count, _expected_eye_size(width, height, vertical_split)):
        return

    cap = open_camera(camera_index, width, height)
    if not cap.isOpened():
        raise RuntimeError(describe_camera_open_failure(camera_index))

    def split(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return split_merged_frame(frame, vertical_split, swap_lr)

    ok, probe = cap.read()
    if not ok:
        raise RuntimeError("讀取相機影格失敗")
    _warn_if_not_side_by_side(probe, vertical_split)

    saved = len(list(left_out.glob("*.png")))
    next_index = _next_frame_index(left_out, right_out)
    cancelled = False
    try:
        while saved < target_count:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("讀取相機影格失敗")
            frame_l, frame_r = split(frame)

            gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
            corners_l = find_corners(gray_l, spec, fast_check=True)
            corners_r = find_corners(gray_r, spec, fast_check=True)

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
                name = f"frame_{next_index:04d}.png"
                next_index += 1
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
        count_text = f"{len(ids)} corners"
    else:
        count_text = "no corners"
    status = f"{count_text}   saved {saved_count}/{target_count}   [SPACE] capture   [ESC] quit"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def capture_mono_charuco(
    camera_index: int,
    out_dir: Path,
    board_spec: CharucoBoardSpec,
    target_count: int,
    min_corners: int = _MIN_SHARED_CHARUCO_CORNERS,
    width: int | None = None,
    height: int | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if _already_complete(out_dir, target_count, (width, height) if width and height else None):
        return

    board = board_spec.build_board()
    cap = open_camera(camera_index, width, height)
    if not cap.isOpened():
        raise RuntimeError(describe_camera_open_failure(camera_index))

    saved = len(list(out_dir.glob("*.png")))
    next_index = _next_frame_index(out_dir)
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
                cv2.imwrite(str(out_dir / f"frame_{next_index:04d}.png"), frame)
                next_index += 1
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
    shared_text = f"shared corners: {shared} (need >={min_shared_corners})"
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
    width: int | None = None,
    height: int | None = None,
) -> None:
    """雙目兩顆鏡頭各自視野重疊區域小、拍不到完整board時使用這個函式。
    不需要整塊board同時入鏡，只要左右畫面有足夠共同角點就能存檔。
    """
    left_out.mkdir(parents=True, exist_ok=True)
    right_out.mkdir(parents=True, exist_ok=True)
    if _already_complete(left_out, target_count, (width, height) if width and height else None):
        return

    board = board_spec.build_board()
    cap_l = open_camera(left_index, width, height)
    cap_r = open_camera(right_index, width, height)
    if not cap_l.isOpened() or not cap_r.isOpened():
        raise RuntimeError(describe_stereo_open_failure(
            (left_index, cap_l.isOpened()), (right_index, cap_r.isOpened())
        ))

    saved = len(list(left_out.glob("*.png")))
    next_index = _next_frame_index(left_out, right_out)
    cancelled = False
    try:
        while saved < target_count:
            # 先兩邊都grab再各自retrieve。read()等於grab+retrieve，串著做的話
            # 兩張畫面可能差到一個影格間隔加上MJPG解碼時間；標定要求板子在左右影像
            # 位於同一個物理位置，中間有位移會直接污染外參R/T，而重投影誤差不會變差
            # （各相機自己的幾何仍然自洽），壞掉的剛好是這個專案最在意的相對關係。
            grabbed_l = cap_l.grab()
            grabbed_r = cap_r.grab()
            if not (grabbed_l and grabbed_r):
                raise RuntimeError("讀取雙目相機影格失敗")
            ok_l, frame_l = cap_l.retrieve()
            ok_r, frame_r = cap_r.retrieve()
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
                name = f"frame_{next_index:04d}.png"
                next_index += 1
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
    width: int | None = None,
    height: int | None = None,
) -> None:
    left_out.mkdir(parents=True, exist_ok=True)
    right_out.mkdir(parents=True, exist_ok=True)
    if _already_complete(left_out, target_count, _expected_eye_size(width, height, vertical_split)):
        return

    board = board_spec.build_board()
    cap = open_camera(camera_index, width, height)
    if not cap.isOpened():
        raise RuntimeError(describe_camera_open_failure(camera_index))

    def split(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return split_merged_frame(frame, vertical_split, swap_lr)

    ok, probe = cap.read()
    if not ok:
        raise RuntimeError("讀取相機影格失敗")
    _warn_if_not_side_by_side(probe, vertical_split)

    saved = len(list(left_out.glob("*.png")))
    next_index = _next_frame_index(left_out, right_out)
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
                name = f"frame_{next_index:04d}.png"
                next_index += 1
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
    mono_p.add_argument("--width", type=int, default=None,
        help="相機解析度寬度；雙目模組的左右並排模式通常只在特定寬解析度下才有")
    mono_p.add_argument("--height", type=int, default=None, help="相機解析度高度")
    mono_p.add_argument(
        "--charuco", action="store_true", help="用ChArUco板取代一般棋盤格（容許畫面只看到板子一部分）"
    )
    mono_p.add_argument("--squares-x", type=int, default=10, help="ChArUco板橫向方格數")
    mono_p.add_argument("--squares-y", type=int, default=8, help="ChArUco板縱向方格數")
    mono_p.add_argument("--marker-size-mm", type=float, default=18.0, help="ChArUco標記邊長")
    mono_p.add_argument("--dictionary", type=str, default="DICT_5X5_100")
    mono_p.add_argument(
        "--legacy-pattern", action="store_true", help="市售現成板子（如AndyMark）偵測不到時加上這個"
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
        "--swap-lr", action="store_true", help="左右眼顛倒時加上這個對調"
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
    stereo_p.add_argument("--width", type=int, default=None,
        help="相機解析度寬度；雙目模組的左右並排模式通常只在特定寬解析度下才有")
    stereo_p.add_argument("--height", type=int, default=None, help="相機解析度高度")
    stereo_p.add_argument(
        "--charuco",
        action="store_true",
        help="用ChArUco板取代一般棋盤格，適用於雙目兩顆鏡頭視野重疊區域小、拍不到完整棋盤格時使用",
    )
    stereo_p.add_argument("--squares-x", type=int, default=10, help="ChArUco板橫向方格數")
    stereo_p.add_argument("--squares-y", type=int, default=8, help="ChArUco板縱向方格數")
    stereo_p.add_argument("--marker-size-mm", type=float, default=18.0, help="ChArUco標記邊長")
    stereo_p.add_argument("--dictionary", type=str, default="DICT_5X5_100")
    stereo_p.add_argument(
        "--legacy-pattern", action="store_true", help="市售現成板子（如AndyMark）偵測不到時加上這個"
    )
    stereo_p.add_argument(
        "--min-shared-corners",
        type=int,
        default=_MIN_SHARED_CHARUCO_CORNERS,
        help="左右畫面至少要有幾個共同角點才允許存檔",
    )

    probe_p = sub.add_parser("probe", help="查詢相機支援哪些解析度（用於尋找雙目並排模式）")
    probe_p.add_argument("--camera", type=int, default=0)
    probe_p.add_argument("--fps-frames", type=int, default=12, help="每個模式量測幾張影格來估算fps")

    board_p = sub.add_parser("board", help="產生ChArUco板圖檔供列印")
    board_p.add_argument("--out", type=Path, default=Path("data/charuco_board.png"))
    board_p.add_argument("--squares-x", type=int, default=10)
    board_p.add_argument("--squares-y", type=int, default=8)
    board_p.add_argument("--square-size-mm", type=float, default=25.0)
    board_p.add_argument("--marker-size-mm", type=float, default=18.0)
    board_p.add_argument("--dictionary", type=str, default="DICT_5X5_100")
    board_p.add_argument(
        "--legacy-pattern", action="store_true", help="市售現成板子（如AndyMark）偵測不到時加上這個"
    )
    board_p.add_argument("--pixels-per-square", type=int, default=80)

    args = parser.parse_args()

    if args.mode == "probe":
        probe_resolutions(args.camera, args.fps_frames)
        return

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
                args.camera, args.out, board_spec, args.target_count,
                min_corners=args.min_corners, width=args.width, height=args.height,
            )
        else:
            spec = ChessboardSpec(
                cols=args.cols, rows=args.rows, square_size_mm=args.square_size_mm
            )
            capture_mono(args.camera, args.out, spec, args.target_count, args.width, args.height)
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
                    width=args.width,
                    height=args.height,
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
                    width=args.width,
                    height=args.height,
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
                    width=args.width,
                    height=args.height,
                )
            else:
                capture_stereo(
                    args.left_camera,
                    args.right_camera,
                    args.left_out,
                    args.right_out,
                    spec,
                    args.target_count,
                    args.width,
                    args.height,
                )


if __name__ == "__main__":
    main()


