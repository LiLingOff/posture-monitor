from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .charuco import CharucoBoardSpec, detect_charuco
from .chessboard import ChessboardSpec, find_corners

_MIN_SHARED_CHARUCO_CORNERS = 6


def _open_camera(
    index: int, width: int | None = None, height: int | None = None
) -> cv2.VideoCapture:
    """開相機。不指定width/height就用驅動的預設模式。

    雙目模組要特別注意：左右眼並排的輸出通常只存在於某些寬解析度模式
    （2560x720之類），驅動預設的640x480往往只給單眼或裁切畫面。
    沒指定解析度而拿到單眼畫面時，切一半會得到兩塊不重疊的裁切，
    看起來像兩個不同場景，標定永遠湊不到共同角點。
    """
    if sys.platform.startswith("linux"):
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(index)

    if not cap.isOpened() or width is None or height is None:
        return cap

    # FOURCC要先設：並排模式多半只在MJPG下提供，YUYV受USB頻寬限制開不到高解析度
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if got != (width, height):
        print(
            f"[警告] 要求{width}x{height}，相機實際給{got[0]}x{got[1]}。"
            f"用 v4l2-ctl -d /dev/video{index} --list-formats-ext 查支援的模式"
        )
    else:
        print(f"相機 index={index} 解析度 {got[0]}x{got[1]}")
    return cap


def _warn_if_not_side_by_side(frame: np.ndarray, vertical_split: bool) -> None:
    """合併畫面的長寬比不像「兩眼並排」時提醒。

    左右並排的畫面寬高比通常>=2（例如2560x720是3.6）；單眼是4:3或16:9，
    比例落在1.3~1.8。拿單眼畫面去切一半不會報錯，只會安靜地產生
    兩塊不重疊的裁切，直到標定湊不到共同角點才發現。
    """
    h, w = frame.shape[:2]
    # 單眼畫面的寬高比通常是4:3(1.33)或16:9(1.78)，合併後其中一個方向變成兩倍：
    #   左右並排 → 寬高比 2.67~3.56，門檻取2.0
    #   上下堆疊 → 高寬比 1.13~1.50（單眼的高寬比只有0.56~0.75），門檻取1.0
    if vertical_split:
        ratio, threshold, axis, layout = h / w, 1.0, "高寬比", "上下堆疊"
    else:
        ratio, threshold, axis, layout = w / h, 2.0, "寬高比", "左右並排"

    if ratio >= threshold:
        return
    print(
        f"[警告] 畫面{w}x{h}，{axis}只有{ratio:.2f}，不像{layout}的雙目輸出（應該>={threshold}）。"
        f"這張很可能是單眼視角——切一半會得到兩塊不重疊的畫面，標定湊不到共同角點。"
        f"用 --width/--height 指定相機的並排模式解析度"
    )


# 常見的單眼與雙目並排解析度。雙目模組的並排模式通常是單眼寬度的兩倍。
_PROBE_RESOLUTIONS: tuple[tuple[int, int], ...] = (
    (320, 240),
    (640, 480),
    (800, 600),
    (1280, 720),
    (1920, 1080),
    (640, 240),
    (1280, 480),
    (2560, 720),
    (2560, 960),
    (3040, 1520),
    (3840, 1080),
)


def probe_resolutions(camera_index: int, fps_frames: int = 12) -> None:
    """逐一試各種解析度，印出相機實際給的畫面尺寸與張數。

    用途是在v4l2-ctl列不出格式、或不確定哪個模式才是左右並排時，直接問相機本人。
    判斷依據是cap.read()真正拿到的frame.shape，不是cap.get()回報的值——
    驅動回報跟實際給的不一致是常態。

    也量張數，因為USB 2.0頻寬有限，高解析度的並排模式常常只剩個位數fps，
    這件事光看解析度清單看不出來。
    """
    print(f"逐一測試 index={camera_index}（以實際讀到的畫面為準）")
    print()
    print(f"{'要求':>11}  {'實際拿到':>11}  {'寬高比':>6}  {'fps':>5}  判讀")
    print("-" * 66)

    results: list[tuple[int, int, float, bool]] = []
    for want_w, want_h in _PROBE_RESOLUTIONS:
        # 每次重開，避免某些驅動在模式間切換時卡住
        cap = _open_camera(camera_index)
        if not cap.isOpened():
            print(f"無法開啟相機 index={camera_index}")
            return
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, want_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want_h)

            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"{want_w:>5}x{want_h:<5}  {'讀取失敗':>11}")
                continue

            # 前幾張通常還在暖機，不列入計時
            for _ in range(3):
                cap.read()
            start = time.perf_counter()
            got_frames = sum(1 for _ in range(fps_frames) if cap.read()[0])
            elapsed = time.perf_counter() - start
        finally:
            cap.release()

        fps = got_frames / elapsed if elapsed > 0 else 0.0
        got_h, got_w = frame.shape[:2]
        ratio = got_w / got_h
        exact = (got_w, got_h) == (want_w, want_h)

        if ratio >= 2.0:
            verdict = "★ 左右並排" + ("" if exact else f"（退回自 {want_w}x{want_h}）")
        elif exact:
            verdict = "單眼畫面"
        else:
            verdict = f"不支援，退回 {got_w}x{got_h}"
        print(f"{want_w:>5}x{want_h:<5}  {got_w:>5}x{got_h:<5}  {ratio:>6.2f}  {fps:>5.1f}  {verdict}")
        results.append((got_w, got_h, fps, exact))

    stereo = {(w, h): (fps, exact) for w, h, fps, exact in results if w / h >= 2.0}
    print()
    if not stereo:
        print("沒測到任何寬高比>=2的模式。這顆相機可能不是「左右眼合併輸出」的類型，")
        print("或並排模式不在候選清單裡。")
        return

    print("可用的並排模式：")
    for (w, h), (fps, exact) in sorted(stereo.items()):
        note = "" if exact else "（驅動退回的，不是直接支援）"
        print(f"  --width {w} --height {h}    每眼 {w // 2}x{h}，約 {fps:.1f} fps {note}")

    print()
    print("挑選原則：")
    print("  1. 優先選驅動直接支援的模式，不要選退回來的")
    print("  2. fps 要夠即時監測用；解析度再高，關鍵點偵測也是縮到 224x224 才餵進網路")
    print("  3. 標定跟執行時必須用同一個解析度——內參 fx/fy/cx/cy 是綁解析度的，")
    print("     換了解析度舊的標定參數就失效，而且不會報錯")


def _already_complete(out_dir: Path, target_count: int) -> bool:
    """資料夾已有足夠張數就回傳True，直接跳過不用開相機。

    沒有這個檢查的話，拍攝迴圈一次都不會執行，後面要顯示最後一幀的變數
    就沒被指派過，會以UnboundLocalError收場。
    """
    saved = len(list(out_dir.glob("*.png")))
    if saved >= target_count:
        print(f"{out_dir} 已有{saved}張（目標{target_count}），跳過拍攝；要重拍請先清空資料夾")
        return True
    return False


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
    # cv2.putText的Hershey字型只有ASCII，中文會全部變成問號，所以畫面文字一律用英文
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
    if _already_complete(out_dir, target_count):
        return

    cap = _open_camera(camera_index, width, height)
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
    width: int | None = None,
    height: int | None = None,
) -> None:
    left_out.mkdir(parents=True, exist_ok=True)
    right_out.mkdir(parents=True, exist_ok=True)
    if _already_complete(left_out, target_count):
        return

    cap_l = _open_camera(left_index, width, height)
    cap_r = _open_camera(right_index, width, height)
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
    width: int | None = None,
    height: int | None = None,
) -> None:
    """給左右眼合併輸出成同一張畫面的雙目相機用（單一USB裝置、單一video node）。

    水平並排（預設）：畫面左半=左眼、右半=右眼，對半寬度切開。
    vertical_split=True 則改成上下切開。
    """
    left_out.mkdir(parents=True, exist_ok=True)
    right_out.mkdir(parents=True, exist_ok=True)
    if _already_complete(left_out, target_count):
        return

    cap = _open_camera(camera_index, width, height)
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

    ok, probe = cap.read()
    if not ok:
        raise RuntimeError("讀取相機影格失敗")
    _warn_if_not_side_by_side(probe, vertical_split)

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
    if _already_complete(out_dir, target_count):
        return

    board = board_spec.build_board()
    cap = _open_camera(camera_index, width, height)
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
    """雙目兩顆鏡頭各自視野重疊區域小、拍不到完整board時用這個——
    不需要整塊board同時入鏡，只要左右畫面有足夠共同角點就能存。
    """
    left_out.mkdir(parents=True, exist_ok=True)
    right_out.mkdir(parents=True, exist_ok=True)
    if _already_complete(left_out, target_count):
        return

    board = board_spec.build_board()
    cap_l = _open_camera(left_index, width, height)
    cap_r = _open_camera(right_index, width, height)
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
    width: int | None = None,
    height: int | None = None,
) -> None:
    left_out.mkdir(parents=True, exist_ok=True)
    right_out.mkdir(parents=True, exist_ok=True)
    if _already_complete(left_out, target_count):
        return

    board = board_spec.build_board()
    cap = _open_camera(camera_index, width, height)
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

    ok, probe = cap.read()
    if not ok:
        raise RuntimeError("讀取相機影格失敗")
    _warn_if_not_side_by_side(probe, vertical_split)

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
    stereo_p.add_argument("--width", type=int, default=None,
        help="相機解析度寬度；雙目模組的左右並排模式通常只在特定寬解析度下才有")
    stereo_p.add_argument("--height", type=int, default=None, help="相機解析度高度")
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

    probe_p = sub.add_parser("probe", help="測試相機支援哪些解析度（找雙目並排模式用）")
    probe_p.add_argument("--camera", type=int, default=0)
    probe_p.add_argument("--fps-frames", type=int, default=12, help="每個模式量幾張來估fps")

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
