"""逐幀記錄成 CSV。

找人來坐二十分鐘，結束時終端機只剩最後一行。要做 Kinovea 對標、要算準確率、
要在報告裡放任何一張圖，都需要逐幀的原始資料。

**被略過的幀也要記錄。** 略過率本身就是結果的一部分（偵測在什麼條件下會失效），
而且只記錄成功的幀會讓事後分析看不出資料有多少缺口。每一列都有 usable 欄位
與略過原因。

記的是原始角度與扣除基準之後的角度兩者。基準日後可能重取，原始值留著才能重算。

**檔案要能自己說出自己是什麼。** 標頭有一個 `# key=value` 區塊，記下受試者、
姿勢條件、第幾次、什麼時候量的，以及用了哪一份基準與它的數值。多受試者的資料
一旦開始累積，靠檔名辨識就會出錯；而基準值一起存下來，是為了日後重取基準時
還能從原始角度重算，這與「原始角度也要記」是同一個理由。

`#` 開頭的列 pandas 用 `comment="#"` 讀得掉，csv 模組要自己濾，
`session_analysis.read_rows` 已經在做。
"""
from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from pose.topology import COCO18_KEYPOINT_NAMES

_COLUMNS = (
    "frame",            # 從1開始的序號，含被略過的幀
    "elapsed_s",        # 從開始記錄起算的秒數
    "usable",           # 這一幀有沒有進入平均
    "reject_reason",    # usable=0 時的原因
    "theta_ca_deg",     # 原始角度
    "theta_sym_deg",
    "theta_ca_corrected_deg",   # 扣掉個人基準之後
    "theta_sym_corrected_deg",
    "theta_ca_mean_deg",        # 當下的移動平均，也就是判定實際看的數字
    "theta_sym_mean_deg",
    "theta_ca_standard_error_deg",
    "distance_mm",
    "azimuth_deg",
    "theta_ca_precision_deg",   # 這個距離與方位下的理論單幀誤差
    "shared_keypoints",
    "max_vertical_disparity_px",        # 所有共同關鍵點裡最差的，手腕腳踝也算進來
    "angle_max_vertical_disparity_px",  # 只看算角度用到的那幾點，門檻擋的是這個
    "shoulder_height_mm",               # 雙肩中點的高度，往上為正（相機座標）
    "shoulder_drop_mm",                 # 比端正坐姿低了多少，往下為正
    "shoulder_drop_mean_mm",            # 當下的移動平均，駝背判定看的是這個
    "posture",                          # 當下的判定：正常／超標／未知
    # 角度用到的四個點各自的垂直視差，**帶正負號**。上面兩欄只留最差一點的絕對值，
    # 分不出「整體有一個固定偏移（標定殘餘）」與「各點各自亂跳（關鍵點雜訊或被頭髮、
    # 身體擋住）」：偏移會讓同一個點的 dy 長期同號，雜訊則正負都有。
    # 加在最後面，既有的程式與讀 CSV 的人不用改欄位位置；舊檔沒有這四欄，讀的人要容許缺。
    "dy_right_ear_px",
    "dy_right_shoulder_px",
    "dy_left_shoulder_px",
    "dy_left_ear_px",
    # θ_KA（2026-11-17 補回）。同樣加在最後面；舊檔沒有這四欄，讀的人要容許缺。
    "theta_ka_deg",                 # 原始角度
    "theta_ka_corrected_deg",       # 扣掉個人基準之後（基準沒有 θ_KA 時留空）
    "theta_ka_mean_deg",            # 當下的移動平均，判定看的是這個
    "theta_ka_side",                # 髖點來源：both／left／right
    # θ_KA 為什麼量不到。2026-11-17 第一批實機資料裡有值的可用幀只佔 0～15%，
    # 而 CSV 沒記頸與髖的狀態，分不出是沒偵測到、垂直視差被擋還是別的原因。
    # dy 空白 = 左右眼至少有一眼沒有這個點；有值但絕對值 > 8px = 被當成配對錯誤。
    "dy_neck_px",
    "dy_left_hip_px",
    "dy_right_hip_px",
    "theta_ka_reason",              # θ_KA 留空的原因；有值時空白
    # 角度用到的四個點在左右眼影像上的像素座標（2026-11-17）。為了事後算前作的 2D θ_CA
    # （legacy_2d.py），表4 要拿它與 3D 的 θ_CA 比。**原始影像像素，沒有去畸變也沒有校正**，
    # 與前作拿 OpenPose 輸出直接算是同一種座標。命名：lx、ly 是左眼，rx、ry 是右眼。
    # 偵測不到的點留空，不寫 0（0 是合法的像素座標）。被略過但有影像的幀也記；
    # 相機斷線、根本沒有影像的幀整列留空。舊檔沒有這十六欄，讀的人要容許缺。
    *(f"{name}_{eye}{axis}_px"
      for name in ("right_ear", "right_shoulder", "left_shoulder", "left_ear")
      for eye in ("l", "r")
      for axis in ("x", "y")),
)

_SIGNED_DY_KEYPOINTS = ("right_ear", "right_shoulder", "left_shoulder", "left_ear")
_PIXEL_KEYPOINTS = _SIGNED_DY_KEYPOINTS
_TRUNK_DY_KEYPOINTS = ("neck", "left_hip", "right_hip")


def _number(value, digits: int = 3):
    """None 一律寫成空字串，不要寫 0，因為 0 是合法的角度值。"""
    return "" if value is None else round(float(value), digits)


def metadata_lines(
    subject: str = "",
    condition: str = "",
    trial: int | None = None,
    baseline_file=None,
    baseline=None,
    display: bool = False,
    extra: dict | None = None,
) -> list[str]:
    """標頭的 `# key=value` 區塊。

    空的欄位整列不寫。寫成空值的話，讀回來分不出「這一項沒填」與「填了空字串」，
    而彙整時「沒填」要能被看見。

    基準的數值跟著檔名一起記。只記檔名的話，那份基準日後被覆蓋就再也對不回來，
    而 `--overwrite` 存在就代表它會被覆蓋。
    """
    lines = [f"# captured_at={datetime.now().isoformat(timespec='seconds')}"]
    for key, value in (("subject", subject), ("condition", condition)):
        if value:
            lines.append(f"# {key}={value}")
    if trial is not None:
        lines.append(f"# trial={trial}")
    if baseline_file is not None:
        lines.append(f"# baseline_file={baseline_file}")
    if baseline is not None:
        lines.extend([
            f"# baseline_theta_ca_deg={baseline.theta_ca_deg:.3f}",
            f"# baseline_theta_sym_deg={baseline.theta_sym_deg:.3f}",
            f"# baseline_theta_ca_error_deg={baseline.theta_ca_standard_error_deg:.3f}",
            f"# baseline_captured_at={baseline.captured_at}",
            f"# baseline_frames={baseline.frames}",
            # 基準 JSON 每次取基準都被覆蓋，方位角與距離只存在最後一份裡。
            # 寫進標頭，事後才對得回「這一段用的是在哪個角度取的基準」。
            f"# baseline_azimuth_deg={baseline.azimuth_deg:.1f}",
            f"# baseline_distance_mm={baseline.distance_mm:.0f}",
        ])
        # 舊基準沒有這一項。空的整列不寫，日後才分得出「沒記」與「記了空值」。
        if getattr(baseline, "theta_ka_deg", None) is not None:
            lines.append(f"# baseline_theta_ka_deg={baseline.theta_ka_deg:.3f}")
    if display:
        # 開視窗會拉低每秒幀數，而視窗長度、遲滯寬度與標準誤差都是按幀數算的。
        # 事後要查得到哪幾段是開著視窗量的。
        lines.append("# display=on")
    # study 的趟次中繼資料（trip_id、desk_level…，2026-11-18）。加在最後面，
    # 舊檔沒有這幾列，讀的人要容許缺。空值整列不寫，理由同上。
    for key, value in (extra or {}).items():
        if value is not None and str(value) != "":
            lines.append(f"# {key}={value}")
    return lines


class MeasurementLog:
    """開著檔案逐幀寫入，並且每一列寫完就 flush。

    不在記憶體裡累積到結束才寫：這個程式會用 Ctrl-C 結束，量測期間也可能
    因為相機斷線而中斷，累積的話那些資料就全沒了。
    """

    def __init__(
        self,
        path: Path,
        subject: str = "",
        overwrite: bool = False,
        condition: str = "",
        trial: int | None = None,
        baseline_file: Path | str | None = None,
        baseline=None,
        display: bool = False,
        extra: dict | None = None,
    ):
        self._path = Path(path)
        if self._path.exists() and not overwrite:
            raise FileExistsError(
                f"{self._path} 已經存在。一次量測要跑二十分鐘，蓋掉就沒了。"
                f"換個檔名，或確定要覆蓋的話加上 --overwrite"
            )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        for line in metadata_lines(subject, condition, trial, baseline_file,
                                   baseline, display, extra):
            self._file.write(line + "\n")
        self._writer.writerow(_COLUMNS)
        self._start = time.perf_counter()
        self._frame = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def frames_written(self) -> int:
        return self._frame

    def write(
        self,
        measurement,
        reject_reason: str | None = None,
        corrected: tuple[float | None, float | None] = (None, None),
        ca_mean: float | None = None,
        sym_mean: float | None = None,
        ca_standard_error: float | None = None,
        posture: str = "",
        shoulder_height_mm: float | None = None,
        shoulder_drop_mm: float | None = None,
        shoulder_drop_mean_mm: float | None = None,
        theta_ka_corrected: float | None = None,
        theta_ka_mean: float | None = None,
        keypoints_2d=None,
    ) -> None:
        """keypoints_2d 是 (左眼點陣列, 右眼點陣列)，各 shape (18, 2)，沒有影像時給 None。"""
        self._frame += 1
        worst = None
        worst_angle = None
        if measurement is not None:
            worst = measurement.max_abs_vertical_disparity_px
            worst_angle = measurement.angle_max_vertical_disparity_px

        self._writer.writerow([
            self._frame,
            round(time.perf_counter() - self._start, 3),
            0 if reject_reason else 1,
            reject_reason or "",
            _number(None if measurement is None else measurement.theta_ca_deg),
            _number(None if measurement is None else measurement.theta_sym_deg),
            _number(corrected[0]),
            _number(corrected[1]),
            _number(ca_mean),
            _number(sym_mean),
            _number(ca_standard_error),
            _number(None if measurement is None else measurement.reference_depth_mm, 1),
            _number(None if measurement is None else measurement.camera_azimuth_deg, 1),
            _number(None if measurement is None else measurement.theta_ca_precision_deg, 2),
            "" if measurement is None else measurement.shared_count,
            _number(worst, 2),
            _number(worst_angle, 2),
            _number(shoulder_height_mm, 1),
            _number(shoulder_drop_mm, 1),
            _number(shoulder_drop_mean_mm, 1),
            posture,
            *self._signed_dy(measurement),
            _number(None if measurement is None else measurement.theta_ka_deg),
            _number(theta_ka_corrected),
            _number(theta_ka_mean),
            "" if measurement is None else (measurement.theta_ka_side or ""),
            *self._signed_dy(measurement, _TRUNK_DY_KEYPOINTS),
            self._trunk_reason(measurement),
            *self._pixels(keypoints_2d),
        ])
        self._file.flush()

    @staticmethod
    def _pixels(keypoints_2d) -> list:
        """四個點 × 左右眼 × (x, y)，順序與 _COLUMNS 一致。缺的、NaN 的一律空字串。"""
        names = _PIXEL_KEYPOINTS
        if keypoints_2d is None:
            return [""] * (len(names) * 4)
        left, right = keypoints_2d
        out = []
        for name in names:
            i = COCO18_KEYPOINT_NAMES.index(name)
            for points in (left, right):
                for value in points[i]:
                    out.append(_number(float(value), 2) if np.isfinite(value) else "")
        return out

    @staticmethod
    def _trunk_reason(measurement) -> str:
        """θ_KA 量不到的原因。量得到或這一幀根本沒有量測就是空字串。"""
        if measurement is None or measurement.theta_ka_deg is not None:
            return ""
        for error in measurement.angle_errors:
            if error.startswith("theta_ka:"):
                return error[len("theta_ka:"):].strip()
        return ""

    @staticmethod
    def _signed_dy(measurement, names=_SIGNED_DY_KEYPOINTS) -> list:
        """指定幾個點各自的垂直視差，帶正負號。沒有左右對應的點寫空字串，不寫 0。"""
        if measurement is None:
            return [""] * len(names)
        out = []
        for name in names:
            d = measurement.vertical_disparity_px[COCO18_KEYPOINT_NAMES.index(name)]
            out.append(_number(float(d), 2) if np.isfinite(d) else "")
        return out

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> MeasurementLog:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
