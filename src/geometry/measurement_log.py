"""逐幀記錄成 CSV。

找人來坐二十分鐘，結束時終端機只剩最後一行——要做 Kinovea 對標、要算準確率、
要在報告裡放任何一張圖，都需要逐幀的原始資料。

**被略過的幀也要記錄。** 略過率本身就是結果的一部分（偵測在什麼條件下會失效），
而且只記錄成功的幀會讓事後分析看不出資料有多少缺口。每一列都有 usable 欄位
與略過原因。

記的是原始角度與扣除基準之後的角度兩者。基準日後可能重取，原始值留著才能重算。
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

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
    "max_vertical_disparity_px",
)


def _number(value, digits: int = 3):
    """None 一律寫成空字串，不要寫 0——0 是合法的角度值。"""
    return "" if value is None else round(float(value), digits)


class MeasurementLog:
    """開著檔案逐幀寫入，並且每一列寫完就 flush。

    不在記憶體裡累積到結束才寫：這個程式會用 Ctrl-C 結束，量測期間也可能
    因為相機斷線而中斷，累積的話那些資料就全沒了。
    """

    def __init__(self, path: Path, subject: str = ""):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if subject:
            # 註解列，pandas 用 comment="#" 讀得掉
            self._file.write(f"# subject={subject}\n")
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
    ) -> None:
        self._frame += 1
        worst = None
        if measurement is not None:
            worst = measurement.max_abs_vertical_disparity_px

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
        ])
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> MeasurementLog:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
