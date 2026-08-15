import numpy as np
import pytest

from facial_emotion.constants import IMG_SIZE
from facial_emotion.data.preprocessing import (
    apply_clahe,
    normalize,
    preprocess_image,
    to_grayscale,
)


def test_to_grayscale_from_bgr():
    bgr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    gray = to_grayscale(bgr)
    assert gray.shape == (32, 32)
    assert gray.dtype == np.uint8


def test_to_grayscale_passthrough():
    gray_in = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    assert np.array_equal(to_grayscale(gray_in), gray_in)


def test_apply_clahe_preserves_shape_and_range():
    gray = np.random.randint(0, 256, (48, 48), dtype=np.uint8)
    out = apply_clahe(gray)
    assert out.shape == gray.shape
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


def test_normalize_scales_to_unit_range():
    gray = np.array([[0, 128], [255, 64]], dtype=np.uint8)
    out = normalize(gray)
    assert out.dtype == np.float32
    assert out.max() <= 1.0 and out.min() >= 0.0
    assert out[0, 0] == pytest.approx(0.0)
    assert out[1, 0] == pytest.approx(1.0)


@pytest.mark.parametrize("use_clahe", [True, False])
def test_preprocess_image_pipeline(use_clahe):
    raw = np.random.randint(0, 256, (96, 96, 3), dtype=np.uint8)
    out = preprocess_image(raw, use_clahe=use_clahe)
    assert out.shape == (IMG_SIZE, IMG_SIZE, 1)
    assert out.dtype == np.float32
    assert out.max() <= 1.0 and out.min() >= 0.0


def test_preprocess_image_already_target_size_skips_resize():
    raw = np.random.randint(0, 256, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    out = preprocess_image(raw, use_clahe=False)
    assert out.shape == (IMG_SIZE, IMG_SIZE, 1)
