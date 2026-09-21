"""檢視雙目標定結果並做合理性驗算。

RMS 重投影誤差只說明標定在內部自洽，說不出參數本身合不合理。
這裡做的是另一類檢查：拿標定結果去對照「這組硬體應該長什麼樣」——
剛性雙目模組的兩顆鏡頭應該幾乎平行、平移應該幾乎只有 X 分量、
像素應該接近正方形、光心應該落在畫面中央附近。
任何一項明顯不對，代表標定收斂到了一個內部自洽但物理上錯誤的解。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .stereo_calibration import StereoCalibrationResult


@dataclass(frozen=True)
class Check:
    name: str
    detail: str
    passed: bool


def _intrinsics(matrix: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(matrix[0, 0]),
        float(matrix[1, 1]),
        float(matrix[0, 2]),
        float(matrix[1, 2]),
    )


def rotation_angle_deg(R: np.ndarray) -> float:
    """兩台相機之間的夾角（度）。剛性模組應該很小。"""
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    return float(np.degrees(np.linalg.norm(rvec)))


def translation_axis_ratio(T: np.ndarray) -> float:
    """平移量有多少比例落在 X 軸上。左右並排的模組應該接近 1。"""
    t = np.asarray(T, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(t))
    if norm == 0.0:
        return 0.0
    return float(abs(t[0]) / norm)


def sanity_checks(
    calib: StereoCalibrationResult, target_error_px: float = 0.5
) -> list[Check]:
    width, height = calib.image_size
    fx_l, fy_l, cx_l, cy_l = _intrinsics(calib.camera_matrix_left)
    fx_r, fy_r, cx_r, cy_r = _intrinsics(calib.camera_matrix_right)

    angle = rotation_angle_deg(calib.R)
    axis_ratio = translation_axis_ratio(calib.T)
    baseline = calib.baseline_mm

    # P2 第四欄編碼的是校正後的平移：|P2[0,3]| = fx_rect * baseline。
    # 這一項串起了內參、外參與校正三者，對不上代表某一段接錯了。
    fx_rect = float(calib.P1[0, 0])
    encoded = abs(float(calib.P2[0, 3]))
    expected = fx_rect * baseline
    encoded_error = abs(encoded - expected) / expected if expected else 1.0

    checks = [
        Check("RMS 重投影誤差", f"{calib.rms_error:.4f} px（目標 <{target_error_px}）",
              calib.rms_error < target_error_px),
        Check("兩相機夾角", f"{angle:.3f}°（剛性模組應 <2°）", angle < 2.0),
        Check("平移方向", f"{axis_ratio * 100:.1f}% 落在 X 軸（左右並排應 >95%）",
              axis_ratio > 0.95),
        Check("左右焦距一致", f"fx 左{fx_l:.1f} / 右{fx_r:.1f}（同型鏡頭應相近）",
              abs(fx_l - fx_r) / max(fx_l, fx_r) < 0.05),
        Check("像素長寬比", f"左 fx/fy={fx_l / fy_l:.4f}，右 fx/fy={fx_r / fy_r:.4f}（應接近1）",
              abs(fx_l / fy_l - 1) < 0.05 and abs(fx_r / fy_r - 1) < 0.05),
        Check("左光心位置", f"({cx_l:.1f}, {cy_l:.1f})，畫面中心 ({width / 2:.0f}, {height / 2:.0f})",
              abs(cx_l - width / 2) < width * 0.1 and abs(cy_l - height / 2) < height * 0.1),
        Check("右光心位置", f"({cx_r:.1f}, {cy_r:.1f})，畫面中心 ({width / 2:.0f}, {height / 2:.0f})",
              abs(cx_r - width / 2) < width * 0.1 and abs(cy_r - height / 2) < height * 0.1),
        Check("P2 與基線一致", f"|P2[0,3]|={encoded:.1f}，fx×baseline={expected:.1f}",
              encoded_error < 0.01),
    ]
    return checks


def format_report(calib: StereoCalibrationResult, target_error_px: float = 0.5) -> str:
    width, height = calib.image_size
    fx_l, fy_l, cx_l, cy_l = _intrinsics(calib.camera_matrix_left)
    fx_r, fy_r, cx_r, cy_r = _intrinsics(calib.camera_matrix_right)

    lines = [
        f"影像尺寸（單眼）  {width} x {height}",
        f"基線長度          {calib.baseline_mm:.2f} mm",
        f"RMS 重投影誤差    {calib.rms_error:.4f} px",
        "",
        "內參",
        f"  左  fx={fx_l:9.2f}  fy={fy_l:9.2f}  cx={cx_l:8.2f}  cy={cy_l:8.2f}",
        f"  右  fx={fx_r:9.2f}  fy={fy_r:9.2f}  cx={cx_r:8.2f}  cy={cy_r:8.2f}",
        "",
        "畸變係數 (k1 k2 p1 p2 k3)",
        f"  左  {np.array2string(np.asarray(calib.dist_coeffs_left).reshape(-1), precision=4, suppress_small=True)}",
        f"  右  {np.array2string(np.asarray(calib.dist_coeffs_right).reshape(-1), precision=4, suppress_small=True)}",
        "",
        "外參",
        f"  兩相機夾角      {rotation_angle_deg(calib.R):.3f}°",
        f"  平移 T (mm)     {np.array2string(np.asarray(calib.T).reshape(-1), precision=3, suppress_small=True)}",
        "",
        "合理性驗算",
    ]
    checks = sanity_checks(calib, target_error_px)
    for check in checks:
        mark = "OK  " if check.passed else "注意"
        lines.append(f"  [{mark}] {check.name}：{check.detail}")

    failed = [c for c in checks if not c.passed]
    lines.append("")
    if failed:
        lines.append(f"有 {len(failed)} 項不符預期。RMS 低只代表內部自洽，")
        lines.append("這些項目對不上時，標定可能收斂到了物理上錯誤的解——建議重拍。")
    else:
        lines.append("全部通過。基線長度請再與雙目模組的規格書對照：")
        lines.append("差距若是固定比例，通常是 --square-size-mm 填錯了同樣的比例。")
    return "\n".join(lines)
