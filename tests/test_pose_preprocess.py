import numpy as np

from pose.preprocess import resize_and_normalize


def test_output_shape_and_dtype():
    frame = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
    out = resize_and_normalize(frame, input_size=224)
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.float32


def test_solid_color_normalizes_around_zero():
    # 128/255約0.502，接近ImageNet mean，normalize後應該落在小範圍內
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    out = resize_and_normalize(frame, input_size=64)
    assert np.all(np.abs(out) < 1.0)


def test_different_input_size():
    frame = np.zeros((300, 200, 3), dtype=np.uint8)
    out = resize_and_normalize(frame, input_size=96)
    assert out.shape == (96, 96, 3)
