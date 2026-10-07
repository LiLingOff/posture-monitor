"""把一幀畫成看得懂的畫面。

這裡吃 cv2 但不開視窗，所以測起來不需要顯示器：每個函式都吃一張 numpy 畫布，
斷言看的是像素。

版面是左右眼並排，下面接一條獨立的面板帶。面板不疊在影像上：兩眼各佔一半之後
沒有空位，疊上去會蓋在受試者頭上。

**先縮放再繪製。** 座標跟著縮放比一起換算，畫的像素少四分之三，而且省掉
`frame.copy()`：`cv2.resize` 本來就會配一塊新的。

繪製不得改到傳進來的畫面。`frames[0]` 接著會交給 `Snapshots.maybe_save`，
就地畫下去會把即時疊圖混進事後查證用的證據。
"""
from __future__ import annotations

import cv2
import numpy as np

from geometry.judgement import Posture
from geometry.study import TURN_LIMIT_DEG
from pose.topology import COCO18_KEYPOINT_NAMES, COCO18_LIMBS, keypoint_index

from .text import TextItem, TextPainter

# BGR。
SKELETON = (150, 150, 150)
GUIDE = (90, 90, 90)
EDGE = (0, 165, 255)        # 橘：關鍵點貼在畫面邊緣
MISPAIRED = (60, 60, 235)   # 紅：左右配對錯了
DIM = (190, 190, 190)
# 未知刻意不是灰的：判定那一段用這裡的顏色畫，而骨架也是灰的，兩者撞色的話
# 整張圖只剩一種顏色，連現在在判哪一段都看不出來。未知正是最需要看清楚的狀態。
STATE_COLOURS = {
    Posture.OK: (90, 205, 90),
    Posture.OVER: (60, 60, 235),
    Posture.UNKNOWN: (245, 245, 245),
}
# 每一條線先描一圈深色再畫本體。辦公室的背景同時有白牆與黑螢幕，單一顏色
# 一定會在其中一種背景上消失，描邊之後兩種都看得見。
HALO = (20, 20, 20)
# monitor 的階段。用字串當鍵而不是 import Mode：疊圖不該相依於流程。
# 少一個的話面板會直接印中文，退回英文時就變成一排問號，所以有測試盯著。
MODE_WORDS = {
    "待命": ("[C] 取基準", "[C] Zero"),
    "倒數": ("請坐正，倒數", "Sit up, starting in"),
    "取基準": ("取基準中，請保持不動", "Zeroing, hold still"),
    "監測": ("", ""),
}
STATE_WORDS = {
    Posture.OK: ("正常", "OK"),
    Posture.OVER: ("超標", "OVER"),
    Posture.UNKNOWN: ("未知", "UNKNOWN"),
}

# 面板、終端機、文件與報告一律用 θ_CA、θ_sym、肩部垂直位移。
#
# 先前面板刻意用白話（頭前傾、肩膀高低、肩膀下沉），理由是站在旁邊看畫面的人
# 不必先知道 θ_CA 是什麼。2026-10-08 改掉：報告書用的是 θ_CA，而畫面截圖會
# 進報告，兩套名字並存的話讀報告的人得自己對應，那個成本比學一次符號高。
#
# 沒有中日韓字型時整個面板退回英文，而 θ 不是 ASCII,英文那一側寫
# theta_CA、theta_sym，不要直接塞 θ，否則退回模式會變成一排問號。
# 底線而不是固定值。列數會變（警告列只在有警告時出現），寫死的話超出的那一列
# 會被裁掉半個字，而被裁掉的正好是只在出事時才出現的那一列。
PANEL_HEIGHT = 118
PANEL_PAD = 8
ROW_GAP = 6
STATE_SIZE = 34
TEXT_SIZE = 20
MIN_WIDTH = 640
# 轉身超過這麼多度就標出來。基準與量測之間受試者轉了身，扣基準就不再有意義，
# 而 2026-09-29 那次是事後才發現的。
TURNED_LIMIT_DEG = TURN_LIMIT_DEG

# 略過原因轉成短標。沒有字型時整個面板靠英文，而原因字串全是中文。
_REASON_TAGS = (
    ("垂直視差", "MISPAIR"),
    ("配對接反", "SWAPPED"),
    ("畫面邊緣", "EDGE"),
    ("沒有偵測到人", "NO PERSON"),
    ("共同關鍵點", "FEW POINTS"),
    ("沒有可用的耳朵或肩膀深度", "NO DEPTH"),
    ("深度", "DEPTH"),
    ("讀取相機影格失敗", "NO FRAME"),
)


def reason_tag(reason: str | None) -> str:
    """略過原因的英文短標。認不出來的回傳 SKIP 而不是猜。"""
    if reason is None:
        return ""
    for needle, tag in _REASON_TAGS:
        if needle in reason:
            return tag
    return "SKIP"


def _point(points: np.ndarray, index: int, scale: float):
    x, y = points[index]
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    # NaN 轉 int 會變成一個極大的整數，cv2 會拿它畫一條穿過整張圖的線。
    return int(round(x * scale)), int(round(y * scale))


def _stroke(canvas, pa, pb, colour, thickness: int) -> None:
    """描邊再畫線。"""
    cv2.line(canvas, pa, pb, HALO, thickness + 3, cv2.LINE_AA)
    cv2.line(canvas, pa, pb, colour, thickness, cv2.LINE_AA)


def draw_bones(canvas, keypoints, scale: float, colour=SKELETON, thickness: int = 3) -> None:
    """灰色骨架。任一端沒偵測到的那條線就不畫。"""
    if keypoints is None:
        return
    for a, b in COCO18_LIMBS:
        pa = _point(keypoints.points, a, scale)
        pb = _point(keypoints.points, b, scale)
        if pa is None or pb is None:
            continue
        _stroke(canvas, pa, pb, colour, thickness)


def draw_segment(canvas, keypoints, name_a: str, name_b: str, scale: float,
                 colour, thickness: int = 4) -> bool:
    """量角度用的那一段，用判定顏色加粗。畫得出來回傳 True。

    耳朵到肩膀與左右肩之間刻意不在 COCO18_LIMBS 裡：那不是肢體，是判定的依據，
    混進灰色骨架就看不出哪一段才是現在在判的東西。
    """
    if keypoints is None:
        return False
    pa = _point(keypoints.points, keypoint_index(name_a), scale)
    pb = _point(keypoints.points, keypoint_index(name_b), scale)
    if pa is None or pb is None:
        return False
    _stroke(canvas, pa, pb, colour, thickness)
    return True


def draw_keypoints(canvas, keypoints, scale: float,
                   edge: tuple[str, ...] = (), mispaired: tuple[str, ...] = ()) -> None:
    """全部的點畫小圓，貼邊的標橘、配錯的標紅。

    同時貼邊又配錯時固定標成配錯的紅色。順序寫死是為了讓測試不必依賴繪製順序；
    挑紅色是因為配對錯誤比貼邊少見也嚴重，那一幀的角度是錯的而不是缺的。
    """
    if keypoints is None:
        return
    for i, name in enumerate(COCO18_KEYPOINT_NAMES):
        p = _point(keypoints.points, i, scale)
        if p is None:
            continue
        if name in mispaired:
            cv2.circle(canvas, p, 7, MISPAIRED, 2, cv2.LINE_AA)
        elif name in edge:
            cv2.circle(canvas, p, 7, EDGE, 2, cv2.LINE_AA)
        cv2.circle(canvas, p, 2, SKELETON, -1, cv2.LINE_AA)


def draw_epipolar_guides(canvas, left, right, names, scale: float,
                         mispaired: tuple[str, ...] = ()) -> None:
    """在左眼每個角度關鍵點的高度畫一條橫跨兩眼的細線。

    判別配對錯誤靠的是垂直視差，而並排的版面讓水平方向好比、垂直方向難比，
    正好是相反的軸。有了這條線，右眼那個點有沒有落在線上一眼就看得出來，
    「right_shoulder 垂直視差 43.3px」不再只是一個數字。
    """
    if left is None:
        return
    for name in names:
        p = _point(left.points, keypoint_index(name), scale)
        if p is None:
            continue
        colour = MISPAIRED if name in mispaired else GUIDE
        cv2.line(canvas, (0, p[1]), (canvas.shape[1] - 1, p[1]), colour, 1, cv2.LINE_AA)


def _fmt(value: float | None, digits: int = 1, suffix: str = "") -> tuple[str, str]:
    """回傳（中文寫法, 英文寫法）。量不到寫破折號，不寫 0，0 是合法的角度。"""
    if value is None:
        return "—", "--"
    text = f"{value:+.{digits}f}{suffix}" if suffix == "°" else f"{value:.{digits}f}{suffix}"
    return text, text.replace("°", "d")


def panel_rows(view) -> list[list[tuple]]:
    """面板每一列的內容：(中文, 英文, 顏色, 字級, 粗體)。

    **版面固定。** 欄位永遠都在，量不到印破折號而不是整欄消失；字級全場統一，
    只有最上面的狀態詞比較大，而它一定在。超標靠顏色講，不靠放大。

    先前警告超標時放大到狀態詞的級數、而且只在超標時才出現，結果是同一個視窗
    在受試者眼前忽大忽小、欄位忽有忽無，每次變化後面的東西全部跟著位移。前作
    的面板從頭到尾就是固定幾行，變的只有數值與顏色，那才看得住。

    只有「整段跑完都不會變」的欄位可以不存在：沒給秒數就沒有剩餘秒數、不判定
    就沒有視窗進度。逐幀會變的一律留著位子。

    排版與繪製分開，因為排版的斷言只能確認字串裡有某段文字，那擋不住排錯欄。
    """
    # once 只量一幀、沒有基準也沒有視窗，判定在那裡結構上不可能成立，印一個
    # 大大的「未知」會讓人以為是姿勢看不出來，實際上是這個模式根本不判。
    word_zh, word_en = (STATE_WORDS[view.state] if view.judging
                        else ("單幀量測", "Single frame"))
    word_colour = STATE_COLOURS[view.state] if view.judging else DIM
    first = [(word_zh, word_en, word_colour, STATE_SIZE, True)]

    # 沒有平均值就退回單幀值。once 只量一幀，視窗結構上永遠是空的，而角度本身
    # 量到了，印破折號等於把手上的數字丟掉。
    ca_value = view.theta_ca_mean_deg
    if ca_value is None:
        ca_value = view.theta_ca_instant_deg
    sym_value = view.theta_sym_mean_deg
    if sym_value is None:
        sym_value = view.theta_sym_instant_deg
    ca = _fmt(ca_value, 1, "°")
    ca_err = _fmt(view.theta_ca_error_deg, 1, "")
    sym = _fmt(sym_value, 1, "°")

    # 誤差與角度同一欄：分成兩欄的話視窗還沒填滿時那一欄不存在，後面全部位移。
    second = [
        (f"θ_{{CA}} {ca[0]} ± {ca_err[0]}°", f"θ_{{CA}} {ca[1]} ± {ca_err[1]}d",
         DIM, TEXT_SIZE, False),
        (f"θ_{{SYM}} {sym[0]}", f"θ_{{SYM}} {sym[1]}", DIM, TEXT_SIZE, False),
    ]

    over_drop = (view.drop_mm is not None and view.drop_threshold_mm is not None
                 and view.drop_mm > view.drop_threshold_mm)
    drop = _fmt(None if view.drop_mm is None else max(0.0, view.drop_mm), 0, "mm")
    second.append((f"肩部垂直位移 {drop[0]}", f"shoulder drop {drop[1]}",
                   MISPAIRED if over_drop else DIM, TEXT_SIZE, over_drop))

    distance = _fmt(view.distance_mm, 0, "mm")
    turned = _fmt(view.turned_deg, 0, "°")
    over_turned = (view.turned_deg is not None
                   and abs(view.turned_deg) > TURNED_LIMIT_DEG)
    third = [
        (f"距離 {distance[0]}", f"Distance {distance[1]}", DIM, TEXT_SIZE, False),
        (f"轉身 {turned[0]}", f"Rotation {turned[1]}",
         MISPAIRED if over_turned else DIM, TEXT_SIZE, over_turned),
    ]
    # 畫面上的角度是扣掉基準之後的值，所以零點本身不印的話，歸零有沒有生效、
    # 扣掉的是多少，站在旁邊的人完全看不出來。前作把 Raw 與 Off 一直印著。
    # 有沒有基準整段不會變，所以這一欄可以不存在。
    if view.theta_ca_offset_deg is not None:
        zero = _fmt(view.theta_ca_offset_deg, 1, "°")
        third.append((f"零點 {zero[0]}", f"Baseline {zero[1]}", DIM, TEXT_SIZE, False))
    # 視窗的進度只有在平均得起來的時候才有意義。once 的「0/1 幀」看起來像一幀
    # 都沒量到，實際上那一幀好好的，只是沒有視窗可以填。
    if view.judging:
        third.append((f"{view.window_count}/{view.window_size} 幀",
                      f"{view.window_count}/{view.window_size}", DIM, TEXT_SIZE, False))
    if view.remaining_s is not None:
        left = max(0.0, view.remaining_s)
        third.append((f"剩 {left:.0f}s", f"{left:.0f}s left", DIM, TEXT_SIZE, False))

    # 要人當場做一件事的話擺在最後一欄，後面沒有東西，所以它出現與消失都不會
    # 把別的欄位推走。
    fourth: list[tuple] = []
    if view.mode:
        zh, en = MODE_WORDS.get(view.mode, (view.mode, view.mode))
        if zh:
            if view.phase_remaining_s is not None:
                zh += f" {max(0.0, view.phase_remaining_s):.0f}s"
                en += f" {max(0.0, view.phase_remaining_s):.0f}s"
            fourth.append((zh, en, (90, 205, 90), TEXT_SIZE, True))
    if view.keys:
        fourth.append(("　".join(k for k, _ in view.keys),
                       "  ".join(e for _, e in view.keys),
                       DIM, TEXT_SIZE, False))
    if over_turned:
        fourth.append(("請轉回正面", "FACE FRONT", MISPAIRED, TEXT_SIZE, True))

    rows = [first, second, third]
    if fourth:
        rows.append(fourth)
    return rows

def panel_height(view) -> int:
    """這一幀的面板要多高。列數會變，所以不能是常數。"""
    rows = panel_rows(view)
    needed = PANEL_PAD * 2 - ROW_GAP + sum(
        max(size for _, _, _, size, _ in row) + ROW_GAP for row in rows)
    return max(PANEL_HEIGHT, needed)


def render_panel(view, width: int, painter: TextPainter) -> np.ndarray:
    """面板那一條帶子。"""
    band = np.zeros((panel_height(view), width, 3), dtype=np.uint8)
    band[:, :] = (28, 28, 28)
    items: list[TextItem] = []
    y = PANEL_PAD
    for row in panel_rows(view):
        x = 14
        height = max(size for _, _, _, size, _ in row)
        for text, ascii_text, colour, size, bold in row:
            items.append(TextItem(x=x, y=y + (height - size) // 2, text=text,
                                  ascii_text=ascii_text, colour=colour,
                                  size=size, bold=bold))
            x += painter.width(text, ascii_text, size, bold) + 22
        y += height + ROW_GAP
    painter.paint(band, items)
    return band


def compose(view, width: int = 1280, painter: TextPainter | None = None,
            eyes: str = "both") -> np.ndarray:
    """整個視窗的畫面。"""
    painter = TextPainter() if painter is None else painter
    width = max(MIN_WIDTH, int(width))

    sources = [(view.left_bgr, view.left, view.edge_left)]
    if eyes == "both" and view.right_bgr is not None:
        sources.append((view.right_bgr, view.right, view.edge_right))

    eye_width = width // len(sources)
    scale = eye_width / view.left_bgr.shape[1]
    eye_height = int(round(view.left_bgr.shape[0] * scale))

    colour = STATE_COLOURS[view.state]
    halves = []
    for frame, keypoints, edge in sources:
        canvas = cv2.resize(frame, (eye_width, eye_height))
        draw_bones(canvas, keypoints, scale)
        if view.theta_ca_side:
            draw_segment(canvas, keypoints, f"{view.theta_ca_side}_ear",
                         f"{view.theta_ca_side}_shoulder", scale, colour)
        draw_segment(canvas, keypoints, "left_shoulder", "right_shoulder", scale, colour)
        draw_keypoints(canvas, keypoints, scale, edge, view.mispaired)
        halves.append(canvas)

    picture = halves[0] if len(halves) == 1 else cv2.hconcat(halves)
    if len(halves) == 2:
        names = ("left_shoulder", "right_shoulder")
        if view.theta_ca_side:
            names = names + (f"{view.theta_ca_side}_ear",)
        draw_epipolar_guides(picture, view.left, view.right, names, scale, view.mispaired)

    # hconcat 之後寬度可能差一兩個像素（整數除法），面板照著它做才接得起來。
    return cv2.vconcat([picture, render_panel(view, picture.shape[1], painter)])


def mark_stale(image: np.ndarray, painter: TextPainter,
               message: str = "沒有資料") -> np.ndarray:
    """在保留的畫面上蓋一條「沒有資料」。

    相機掉線時每次讀取都立刻失敗，視窗會停在最後一張還活著的畫面上。看起來
    正常的凍結畫面是這個專案一再吃虧的那一類，所以寧可難看也要標出來。
    """
    marked = image.copy()
    band = np.zeros((44, marked.shape[1], 3), dtype=np.uint8)
    band[:, :] = (0, 0, 120)
    painter.paint(band, [TextItem(x=14, y=8, text=message, ascii_text="NO DATA",
                                  colour=(255, 255, 255), size=26, bold=True)])
    marked[: band.shape[0], :] = band
    return marked


def label_keypoints(canvas, keypoints, scale: float = 1.0) -> None:
    """每個點標上名字。存檔的快照用，事後要查是哪一點跑掉。

    即時畫面不用這個：兩眼各佔一半之後，十八個標籤會疊成一團。
    """
    if keypoints is None:
        return
    for i, name in enumerate(COCO18_KEYPOINT_NAMES):
        p = _point(keypoints.points, i, scale)
        if p is None:
            continue
        cv2.putText(canvas, name, (p[0] + 6, p[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, SKELETON, 1, cv2.LINE_AA)
