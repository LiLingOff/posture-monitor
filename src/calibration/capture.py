from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .charuco import CharucoBoardSpec, detect_charuco
from .chessboard import ChessboardSpec, find_corners

_MIN_SHARED_CHARUCO_CORNERS = 6


def _video_nodes() -> list[str]:
    """目前存在的 /dev/videoN，依編號排序。"""
    nodes = []
    for path in Path("/dev").glob("video*"):
        suffix = path.name[len("video"):]
        if suffix.isdigit():
            nodes.append((int(suffix), path.name))
    return [name for _, name in sorted(nodes)]


def _own_processes_holding(node: Path) -> list[str]:
    """本使用者有哪些程序開著這個裝置，回傳 "pid 名稱" 的清單。

    掃 /proc 而不是呼叫 fuser，因為 fuser 不一定裝得到，而且這段是在錯誤處理
    路徑上跑的，不該再依賴外部指令。別的使用者的程序看不到（需要 root），
    所以查不到不代表沒有人佔著，訊息裡要講清楚這件事。
    """
    holders = []
    try:
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                for fd in (proc / "fd").iterdir():
                    if fd.resolve() == node:
                        name = (proc / "comm").read_text().strip()
                        holders.append(f"pid {proc.name} ({name})")
                        break
            except OSError:
                continue  # 程序結束了，或是別的使用者的
    except OSError:
        return []
    return holders


def describe_camera_open_failure(index: int) -> str:
    """相機開不起來時，實際去查一遍再回報。

    這件事在同一台機器上時好時壞，光給一份檢查清單沒有用。清單上的每一項
    使用者都得自己跑一次，而其中兩項程式當場就查得到。所以這裡直接看
    /dev/videoN 在不在、權限夠不夠、本使用者有沒有別的程序開著它，
    把查得到的講成事實，查不到的才留成待確認項目。

    Jetson 上已知的成因：PhotonVision 服務獨佔相機；重新插拔後 /dev/videoN
    的編號整組移位（一顆 UVC 雙目模組佔兩個節點，只有編號小的那個能取像）；
    前一次執行剛結束，核心還沒把 USB 介面放掉。
    """
    if not sys.platform.startswith("linux"):
        return (
            f"無法開啟相機 index={index}。確認裝置已接上，"
            f"並用 python -m src.calibration.capture probe --camera {index} 查詢可用模式"
        )

    node = Path(f"/dev/video{index}")
    lines = [f"無法開啟相機 index={index}。當場查到的狀況："]

    if not node.exists():
        available = _video_nodes()
        lines.append(f"  /dev/video{index} 不存在。")
        if available:
            lines.append(f"  目前存在的是 {'、'.join(available)}。")
            lines.append(
                "  重新插拔之後編號會整組移位。一顆雙目模組佔用兩個節點，"
                "只有編號較小的那個能取像，所以要試的是上面編號最小的那個："
            )
            lines.append(f"       python posture.py live --camera {available[0][len('video'):]} ...")
        else:
            lines.append("  一個 /dev/video* 都沒有。裝置沒接上，或 USB 沒認到。用 dmesg | tail -30 看看。")
        return "\n".join(lines)

    lines.append(f"  /dev/video{index} 存在。")
    if not os.access(node, os.R_OK | os.W_OK):
        lines.append("  但沒有讀寫權限。使用者要加入 video 群組：")
        lines.append("       sudo usermod -aG video $USER    加完要重新登入才生效")
        return "\n".join(lines)
    lines.append("  讀寫權限正常。")

    holders = _own_processes_holding(node)
    if holders:
        lines.append(f"  有程序正開著它：{'、'.join(holders)}。先把它結束掉。")
        return "\n".join(lines)

    lines.append("  本使用者的程序沒有開著它，但別的使用者的看不到（需要 root）。接著查：")
    for command, why in (
        (f"sudo fuser -v /dev/video{index}", "誰佔著它"),
        ("dmesg | tail -30", "USB 有沒有斷開重連"),
    ):
        lines.append(f"       {command:<34}{why}")
    lines.append(
        "  如果剛結束上一次執行就立刻重跑，核心可能還沒把 USB 介面放掉，"
        "等一兩秒再試。"
    )
    return "\n".join(lines)


def describe_stereo_open_failure(
    left: tuple[int, bool], right: tuple[int, bool]
) -> str:
    """兩顆獨立相機的版本，指出是哪一顆開不了。

    原本只印出兩個 index，看不出問題在哪一顆。兩顆都開不了通常是共通原因
    （服務佔用、權限），只有一顆代表那顆的節點編號或接線有問題。
    """
    failed = [index for index, opened in (left, right) if not opened]
    which = "、".join(f"index={i}" for i in failed)
    both = "兩顆都開不了，多半是共通原因" if len(failed) == 2 else "另一顆是正常的"
    return f"雙目相機開不起來：{which}（{both}）\n" + describe_camera_open_failure(failed[0])


def _open_camera(
    index: int, width: int | None = None, height: int | None = None
) -> cv2.VideoCapture:
    """開啟相機。未指定width/height時採用驅動的預設模式。

    雙目模組要特別注意：左右眼並排的輸出通常只存在於某些寬解析度模式
    （2560x720之類），驅動預設的640x480往往只提供單眼或裁切後的畫面。
    未指定解析度而取得單眼畫面時，切成兩半會得到兩塊不重疊的裁切區域，
    看起來像兩個不同場景，標定永遠無法取得足夠的共同角點。
    """
    if sys.platform.startswith("linux"):
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    elif sys.platform.startswith("win"):
        # Windows預設走MSMF，但它對UVC的寬解析度模式支援不完整，常常無視
        # MJPG設定、只給得出640x480。DirectShow對雙目模組的並排模式相容性好得多，
        # 所以先試DSHOW，開不起來才退回預設後端。
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(index)
    else:
        cap = cv2.VideoCapture(index)

    if not cap.isOpened() or width is None or height is None:
        return cap

    # FOURCC要先設定：並排模式多半只在MJPG下提供，YUYV受USB頻寬限制無法開啟高解析度
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if got != (width, height):
        hint = (
            f"用 v4l2-ctl -d /dev/video{index} --list-formats-ext 查詢支援的模式"
            if sys.platform.startswith("linux")
            else f"用 python -m src.calibration.capture probe --camera {index} 查詢支援的模式"
        )
        print(f"[警告] 要求{width}x{height}，相機實際提供{got[0]}x{got[1]}。{hint}")
    else:
        print(f"相機 index={index} 解析度 {got[0]}x{got[1]}")
    return cap


def _warn_if_not_side_by_side(frame: np.ndarray, vertical_split: bool) -> None:
    """合併畫面的長寬比不符合左右並排特徵時發出提醒。

    左右並排的畫面寬高比通常>=2（例如2560x720是3.6）；單眼是4:3或16:9，
    比例落在1.3~1.8。把單眼畫面切成兩半不會出現錯誤訊息，得到的是
    兩塊不重疊的裁切區域，要等到標定湊不出足夠的共同角點才會發現。
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
        f"這張很可能是單眼視角，切成兩半會得到兩塊不重疊的畫面，標定無法取得足夠的共同角點。"
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


def _summarize_side_by_side(
    results: list[tuple[int, int, float, bool]]
) -> dict[tuple[int, int], tuple[float, bool]]:
    """從逐一測試的結果整理出可用的並排模式。

    同一個實際模式可能由好幾個要求解析度達成：例如要求2560x960時驅動給2560x720，
    而2560x720本身又是直接要得到的。只要有任何一次是直接要到的，這個模式就是
    原生支援，不能被後來的退回結果覆蓋掉。覆蓋掉的話會把原生模式標成退回模式，
    剛好誤導掉「優先選原生支援」這條挑選原則。
    """
    stereo: dict[tuple[int, int], tuple[float, bool]] = {}
    for w, h, fps, exact in results:
        if w / h < 2.0:
            continue
        previous = stereo.get((w, h))
        if previous is None or (exact and not previous[1]):
            stereo[(w, h)] = (fps, exact)
    return stereo


def probe_resolutions(camera_index: int, fps_frames: int = 12) -> None:
    """逐一測試各種解析度，印出相機實際提供的畫面尺寸與張數率。

    用途是在v4l2-ctl列不出格式、或不確定哪個模式才是左右並排時，直接向相機查詢。
    判斷依據是cap.read()實際取得的frame.shape，而非cap.get()回報的值，
    驅動回報值與實際提供的畫面不一致是常態。

    同時量測張數率，因為USB 2.0頻寬有限，高解析度的並排模式常常只剩個位數fps，
    這一點光看解析度清單看不出來。
    """
    print(f"逐一測試 index={camera_index}（以實際讀到的畫面為準）")
    print()
    print(f"{'要求':>11}  {'實際取得':>11}  {'寬高比':>6}  {'fps':>5}  判讀")
    print("-" * 66)

    results: list[tuple[int, int, float, bool]] = []
    for want_w, want_h in _PROBE_RESOLUTIONS:
        # 每次重新開啟，避免某些驅動在模式間切換時停住
        cap = _open_camera(camera_index)
        if not cap.isOpened():
            print(describe_camera_open_failure(camera_index))
            return
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, want_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want_h)

            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"{want_w:>5}x{want_h:<5}  {'讀取失敗':>11}")
                continue

            # 前幾張通常還在預熱，不列入計時
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

    stereo = _summarize_side_by_side(results)
    print()
    if not stereo:
        print("沒有測到任何寬高比>=2的模式。這顆相機可能不屬於左右眼合併輸出的類型，")
        print("或並排模式不在候選清單內。")
        return

    print("可用的並排模式：")
    for (w, h), (fps, exact) in sorted(stereo.items()):
        note = "" if exact else "（驅動退回的模式，並非原生支援）"
        print(f"  --width {w} --height {h}    每眼 {w // 2}x{h}，約 {fps:.1f} fps {note}")

    print()
    print("挑選原則：")
    print("  1. 優先選擇驅動原生支援的模式，不要選退回來的")
    print("  2. fps 要足夠即時監測使用；解析度再高，關鍵點偵測也會先等比例縮放到高度 256 才輸入網路")
    print("  3. 標定與執行時必須使用同一個解析度。內參 fx/fy/cx/cy 的數值綁定於解析度，")
    print("     更換解析度後舊的標定參數就失效，而且不會出現錯誤訊息")


def split_merged_frame(
    frame: np.ndarray, vertical_split: bool = False, swap_lr: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """把左右眼合併輸出的畫面切成(左, 右)。

    拍攝與執行期必須用同一套切法，否則左右眼會對調，三角測量的深度符號整個反過來。
    pose.cli也是呼叫這個函式，避免兩邊各自維護一份而慢慢分岔。
    """
    if vertical_split:
        h = frame.shape[0]
        a, b = frame[: h // 2], frame[h // 2 :]
    else:
        w = frame.shape[1]
        a, b = frame[:, : w // 2], frame[:, w // 2 :]
    return (b, a) if swap_lr else (a, b)


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

    cap = _open_camera(camera_index, width, height)
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

    cap_l = _open_camera(left_index, width, height)
    cap_r = _open_camera(right_index, width, height)
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

    cap = _open_camera(camera_index, width, height)
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
    cap = _open_camera(camera_index, width, height)
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
    cap_l = _open_camera(left_index, width, height)
    cap_r = _open_camera(right_index, width, height)
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
    cap = _open_camera(camera_index, width, height)
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
