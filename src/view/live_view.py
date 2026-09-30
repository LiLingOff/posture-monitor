"""一幀要畫的所有東西，收成一個值。

繪製的函式因此只吃一個參數，測試就不必湊出相機、模型與整條量測管線；而
「從量測結果挑出哪些數字」這件事本身也變成可以單獨測的邏輯。

這個模組只用 numpy。cv2 在 `overlay`、Pillow 在 `text`，都不在這裡。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from geometry.judgement import Posture
from geometry.pipeline import keypoints_at_frame_edge, mispaired_keypoints
from pose.keypoints import PersonKeypoints


@dataclass(frozen=True)
class LiveView:
    """一幀的畫面與數字。"""

    left_bgr: np.ndarray
    right_bgr: np.ndarray | None
    left: PersonKeypoints | None
    right: PersonKeypoints | None

    # 要標出來的東西
    theta_ca_side: str | None = None
    edge_left: tuple[str, ...] = ()
    edge_right: tuple[str, ...] = ()
    mispaired: tuple[str, ...] = ()

    # 狀態詞
    state: Posture = Posture.UNKNOWN
    skip_reason: str | None = None

    # 數字
    theta_ca_mean_deg: float | None = None
    theta_ca_error_deg: float | None = None
    theta_ca_instant_deg: float | None = None
    theta_sym_mean_deg: float | None = None
    theta_sym_error_deg: float | None = None
    theta_sym_instant_deg: float | None = None
    window_count: int = 0
    window_size: int = 0
    distance_mm: float | None = None
    turned_deg: float | None = None
    drop_mm: float | None = None
    drop_threshold_mm: float | None = None
    rejected: int = 0
    frames: int = 0
    remaining_s: float | None = None

    # 第二階段的 monitor 會用到，第一階段一律留空
    mode: str | None = None
    phase_remaining_s: float | None = None
    notice: str | None = None
    # (中文, 英文) 成對。先前只存中文、畫英文時從第一個字猜，於是 once 的
    # 「任意鍵關閉」在退回模式下整段變成問號。成對存的話漏了就編譯不過。
    keys: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @classmethod
    def build(
        cls, *, frames, match, measurement, corrected, ca_window, sym_window,
        state: Posture, skip_reason: str | None, rejected: int, frames_seen: int,
        remaining_s: float | None = None, baseline=None,
        drop_mm: float | None = None, drop_threshold_mm: float | None = None,
        mode: str | None = None, phase_remaining_s: float | None = None,
        notice: str | None = None,
        keys: tuple[tuple[str, str], ...] = (),
    ) -> LiveView:
        """從量測迴圈手上已經有的東西組一幀。"""
        left_bgr, right_bgr = frames[0], frames[1]
        height, width = left_bgr.shape[:2]
        image_size = (width, height)

        return cls(
            left_bgr=left_bgr,
            right_bgr=right_bgr,
            left=match.left,
            right=match.right,
            theta_ca_side=measurement.theta_ca_side,
            # 同一隻眼傳兩次不是寫錯。keypoints_at_frame_edge 只要任一眼貼邊就
            # 回報那個名字，兩眼都標的話會在沒問題的那一眼標出紅點，正好抵銷
            # 並排兩眼的用意：要看的就是哪一眼不對。
            edge_left=tuple(keypoints_at_frame_edge(match.left, match.left, image_size)),
            edge_right=tuple(keypoints_at_frame_edge(match.right, match.right, image_size)),
            mispaired=tuple(mispaired_keypoints(measurement)),
            state=state,
            skip_reason=skip_reason,
            theta_ca_mean_deg=ca_window.mean,
            theta_ca_error_deg=ca_window.standard_error,
            theta_ca_instant_deg=corrected[0],
            theta_sym_mean_deg=sym_window.mean,
            theta_sym_error_deg=sym_window.standard_error,
            theta_sym_instant_deg=corrected[1],
            window_count=ca_window.count,
            window_size=ca_window.window,
            distance_mm=measurement.reference_depth_mm,
            turned_deg=turned_deg(measurement, baseline),
            drop_mm=drop_mm,
            drop_threshold_mm=drop_threshold_mm,
            rejected=rejected,
            frames=frames_seen,
            remaining_s=remaining_s,
            mode=mode,
            phase_remaining_s=phase_remaining_s,
            notice=notice,
            keys=keys,
        )


def turned_deg(measurement, baseline) -> float | None:
    """受試者相對取基準時轉了多少。

    量的是差值不是方位角本身。方位角由雙肩連線算出，所以它同時含有「模組架在
    哪」與「受試者面向哪」兩件事；扣掉基準之後剩下的才是後者。2026-09-29 誤報率
    17% 那次就是量測中轉頭看螢幕，事後才從資料反推出來，當場沒有人發現。

    雙肩要同時偵測到才算得出方位角，所以這個值常常是 None；沒有基準時也是
    None，不是 0。
    """
    if baseline is None or measurement.camera_azimuth_deg is None:
        return None
    return float(measurement.camera_azimuth_deg - baseline.azimuth_deg)
