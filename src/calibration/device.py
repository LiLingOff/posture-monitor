"""相機這個裝置本身：挑節點、開起來、切開併排的畫面、開不了的時候說人話。

與 `capture.py` 分開，因為兩者的使用者不同。這裡是量測時每一幀都會走到的
路徑（`posture.py` 與 `pose/cli.py` 都只需要這一層），而 capture 是拍標定
影像用的互動工具，整包 cv2 視窗與棋盤格偵測都在那邊。先前兩者混在一起，
於是量測的主程式得去 import 一個拍照工具的私有函式。

訊息在這裡佔的篇幅比邏輯多，那是刻意的。節點編號會整組移位、PhotonVision
會佔住相機、權限不夠、解析度退回預設值,這幾件事實機全部發生過，而每一次
真正花掉時間的都不是查不出原因，是訊息沒講到點子上。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


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


def find_camera_index(
    width: int | None = None,
    height: int | None = None,
    probe=None,
    candidates: list[int] | None = None,
) -> tuple[int, str]:
    """自動挑一個能用的相機 index，回傳（index, 怎麼挑的）。

    存在的理由是節點編號會移位：重新插拔或重開機之後 video0/1 可能變成
    video1/2，而每次都要先失敗一次才知道該改成哪個。帶受試者來量測時，
    浪費的是他的時間。

    挑選分兩層。一顆 UVC 雙目模組佔用兩個 /dev/videoN，其中一個開得起來卻
    讀不出畫面，所以「開得起來」不算數，要真的讀到一幀。另外，要求的解析度
    沒拿到時畫面多半是單眼或裁切過的，切成兩半會得到兩塊不重疊的區域，
    所以尺寸對得上的優先，兩者都沒有才退而求其次。

    probe 與 candidates 可以注入，讓挑選邏輯本身不接相機也測得到。
    """
    if candidates is None:
        candidates = [int(name[len("video"):]) for name in _video_nodes()] or [0]
    if probe is None:
        probe = _probe_camera(width, height)

    wanted = (width, height) if width and height else None
    readable: list[tuple[int, tuple[int, int] | None]] = []
    for index in candidates:
        ok, size = probe(index)
        if not ok:
            continue
        if wanted and size == wanted:
            return index, f"自動挑到 index={index}，讀得到 {size[0]}x{size[1]}"
        readable.append((index, size))

    if not readable:
        listed = "、".join(str(i) for i in candidates)
        raise RuntimeError(
            f"試過的每一個相機節點（{listed}）都讀不到畫面。\n"
            + describe_camera_open_failure(candidates[0])
        )

    index, size = readable[0]
    got = f"{size[0]}x{size[1]}" if size else "未知尺寸"
    if wanted is None:
        return index, f"自動挑到 index={index}，讀得到 {got}（沒有指定要求的解析度）"
    return index, (
        f"自動挑到 index={index}，但它只給得出 {got}，"
        f"要求的是 {wanted[0]}x{wanted[1]}。沒有一個節點拿得到要求的解析度"
    )


def _probe_camera(width: int | None, height: int | None):
    """真的去開一次相機並讀一幀，回傳（成功嗎, 實際尺寸）。"""
    def probe(index: int) -> tuple[bool, tuple[int, int] | None]:
        cap = open_camera(index, width, height, quiet=True)
        try:
            if not cap.isOpened():
                return False, None
            ok, frame = cap.read()
            if not ok or frame is None:
                return False, None
            return True, (frame.shape[1], frame.shape[0])
        finally:
            cap.release()
    return probe


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
            lowest = available[0][len("video"):]
            # 不要叫人記下這個編號。2026-10-01 實機連續兩次：掉線時訊息說改用
            # --camera 1，照做之後 video1 又不見了、訊息改說用 0。裝置每重新
            # 列舉一次編號就移位一次，所以當下量到的編號在被讀到的時候已經過期。
            # auto 本來就是預設值，照著這裡的編號去指定反而比不指定更糟。
            lines.append(
                "  重新插拔、重開機、或裝置自己重新列舉之後，編號都會整組移位，"
                "所以上面這個編號你讀到的時候可能又變了。改用 --camera auto"
                "（那本來就是預設值），它會逐一試到讀得出畫面為止。"
            )
            lines.append(
                f"  一顆雙目模組佔用兩個節點，只有編號較小的那個能取像；"
                f"現在編號最小的是 video{lowest}。"
            )
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


def open_camera(
    index: int, width: int | None = None, height: int | None = None,
    quiet: bool = False,
) -> cv2.VideoCapture:
    """開啟相機。未指定width/height時採用驅動的預設模式。

    quiet 給自動挑選用。挑選過程會把每個節點都開一次，那些訊息重複而且會與
    挑選結果本身混在一起；挑完之後真正開啟的那一次才該講話。

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
        if not quiet:
            print(f"[警告] 要求{width}x{height}，相機實際提供{got[0]}x{got[1]}。{hint}")
    elif not quiet:
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
        cap = open_camera(camera_index)
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


def merged_capture_size(
    single_eye_size: tuple[int, int], vertical_split: bool = False
) -> tuple[int, int]:
    """標定用的單眼尺寸對應到相機該輸出的合併畫面尺寸。

    UVC 雙目模組送出的是一張左右（或上下）併排的畫面，而標定存的是切開之後
    單眼的尺寸。要向相機要求的是合併後的那個數字。
    """
    width, height = single_eye_size
    return (width, height * 2) if vertical_split else (width * 2, height)


def describe_resolution_mismatch(
    calibrated: tuple[int, int],
    captured: tuple[int, int],
    vertical_split: bool = False,
) -> str | None:
    """切開後的單眼尺寸與標定不符時回傳該講的話，相符時回傳 None。

    這件事必須在取樣之前擋下來，而且要當成錯誤而不是警告。內參是綁在特定
    解析度上的：fx 用 1280 寬的畫面算出來，套到 320 寬的半幀上，視差回推的深度
    就是垃圾。2026-09-29 實機踩到一次，相機退回 640x480，於是 192 幀全部算出
    深度 104mm，而那個訊息指向的是左右配對錯誤，查錯了方向。
    """
    if tuple(calibrated) == tuple(captured):
        return None
    want = merged_capture_size(calibrated, vertical_split)
    return (
        f"畫面尺寸與標定不符：標定是單眼 {calibrated[0]}x{calibrated[1]}，"
        f"現在切開後是 {captured[0]}x{captured[1]}。\n"
        f"內參綁在解析度上，不符的話算出來的深度沒有意義。加上參數再跑一次：\n"
        f"  --width {want[0]} --height {want[1]}\n"
        f"相機拿不到這個解析度的話，要嘛是節點挑錯了，要嘛這顆模組要用 "
        f"--vertical-split，兩者都不是就得用現在的解析度重新標定。"
    )


class CameraReadError(RuntimeError):
    """相機讀不到畫面。

    與「沒偵測到人」分開，因為兩者該有的反應相反：偵測失敗是暫時的，下一幀就
    可能好了，值得繼續跑；相機不見了則重試多少次都一樣，而且重試的迴圈會全速
    空轉。2026-09-29 實機的 USB 在量測中途斷掉（errno 19），程式在 110 秒內
    寫進 162485 筆空記錄，把前面九十秒的有效資料埋在裡面。
    """


def describe_camera_loss(index: int, failures: int, seconds: float) -> str:
    """相機中途不見了。訊息要說清楚資料還在，因為那是當下最想知道的事。"""
    return (
        f"相機連續 {failures} 次讀不到畫面（約 {seconds:.0f} 秒），停止量測。\n"
        f"這通常是 USB 斷線或供電不足，節點編號也可能在重新列舉之後換掉了。\n"
        f"已經寫下的資料是完整的，斷線之前那一段照常可以分析。\n"
        f"接回去之後確認節點：ls -l /dev/video*，或直接用 --camera auto 重跑。\n"
        + describe_camera_open_failure(index)
    )

