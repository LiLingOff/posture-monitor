"""前處理與座標還原。

座標還原是這個模組最容易無聲出錯的地方：漏乘stride/upsample_ratio、
忘了扣補邊、或忘了除以縮放倍率，都不會有任何錯誤訊息，
只會讓三角測量拿到系統性偏移的2D點。所以用往返還原來鎖住。
"""
import numpy as np
import pytest

from pose.preprocess import (DEFAULT_STRIDE, DEFAULT_UPSAMPLE_RATIO,
                             LetterboxInfo, resize_and_pad,
                             restore_keypoint_coordinates)


def _frame(h: int, w: int) -> np.ndarray:
    return np.full((h, w, 3), 200, dtype=np.uint8)


def test_output_height_matches_requested_and_is_stride_aligned():
    padded, _ = resize_and_pad(_frame(480, 640), input_height=256, stride=8)
    assert padded.shape[0] == 256
    assert padded.shape[1] % 8 == 0
    assert padded.dtype == np.float32


def test_preserves_aspect_ratio():
    """等比例縮放，不像先前的作法直接拉成正方形讓人體變形。"""
    padded, info = resize_and_pad(_frame(480, 640), input_height=256, stride=8)
    assert info.scale == pytest.approx(256 / 480)
    # 640 * (256/480) = 341.33 -> 縮放後寬度341，補到8的倍數是344
    assert padded.shape[1] == 344


def test_normalization_is_mean128_scale256_not_imagenet():
    """上游用的是(img-128)/256，不是ImageNet的mean/std。用錯權重就對不上。"""
    padded, _ = resize_and_pad(_frame(64, 64), input_height=64, stride=8)
    assert padded.max() == pytest.approx((200 - 128) / 256.0, abs=1e-5)


def test_bgr_is_not_converted_to_rgb():
    """上游直接把OpenCV的BGR餵進網路，跟著做才對得上權重。"""
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:, :, 0] = 255  # 只有B通道
    padded, _ = resize_and_pad(frame, input_height=64, stride=8)
    assert padded[0, 0, 0] > padded[0, 0, 2], "通道順序被換掉了"


def test_restore_round_trips_a_known_point():
    """把原始座標正向換算到熱圖尺度，再還原，要回到原點。"""
    info = LetterboxInfo(scale=256 / 480, pad_top=0, pad_left=3)
    factor = DEFAULT_STRIDE / DEFAULT_UPSAMPLE_RATIO

    original = np.array([[320.0, 240.0], [100.0, 50.0]])
    # 正向：縮放 -> 補邊 -> 除以熱圖倍率
    forward = np.empty_like(original)
    forward[:, 0] = (original[:, 0] * info.scale + info.pad_left) / factor
    forward[:, 1] = (original[:, 1] * info.scale + info.pad_top) / factor

    restored = restore_keypoint_coordinates(forward, info)
    assert restored == pytest.approx(original, abs=1e-9)


def test_restore_without_padding_is_pure_scaling():
    info = LetterboxInfo(scale=0.5, pad_top=0, pad_left=0)
    restored = restore_keypoint_coordinates(np.array([[10.0, 20.0]]), info)
    # 10 * (8/4) / 0.5 = 40
    assert restored == pytest.approx(np.array([[40.0, 80.0]]))


def test_restore_propagates_nan():
    """未偵測到的點是NaN，不能在還原過程中變成數字。"""
    info = LetterboxInfo(scale=0.5, pad_top=4, pad_left=6)
    restored = restore_keypoint_coordinates(np.array([[np.nan, np.nan], [10.0, 20.0]]), info)
    assert np.isnan(restored[0]).all()
    assert np.isfinite(restored[1]).all()


def test_restore_rejects_wrong_shape():
    info = LetterboxInfo(scale=1.0, pad_top=0, pad_left=0)
    with pytest.raises(ValueError):
        restore_keypoint_coordinates(np.zeros((5, 3)), info)


def test_empty_frame_rejected():
    with pytest.raises(ValueError):
        resize_and_pad(np.zeros((0, 64, 3), dtype=np.uint8))
