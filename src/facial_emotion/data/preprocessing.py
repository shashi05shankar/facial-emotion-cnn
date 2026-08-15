"""CLAHE + grayscale + normalization preprocessing for face crops."""
from __future__ import annotations

import cv2
import numpy as np

from facial_emotion.constants import IMG_SIZE


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a single-channel uint8 image, converting from BGR/RGB if needed."""
    if image.ndim == 3 and image.shape[-1] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[-1] == 1:
        return image[..., 0]
    return image


def apply_clahe(gray_image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization on a uint8 grayscale image."""
    if gray_image.dtype != np.uint8:
        gray_image = np.clip(gray_image, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(gray_image)


def normalize(gray_image: np.ndarray) -> np.ndarray:
    """Scale a uint8 [0, 255] image to float32 [0, 1]."""
    return gray_image.astype(np.float32) / 255.0


def preprocess_image(image: np.ndarray, img_size: int = IMG_SIZE, use_clahe: bool = True) -> np.ndarray:
    """Full pipeline: grayscale -> resize -> CLAHE -> normalize -> reshape to (H, W, 1).

    Accepts a raw uint8 image of any size/channel count (RGB, BGR, or
    already-grayscale) and returns a model-ready float32 array.
    """
    gray = to_grayscale(image)
    if gray.shape[:2] != (img_size, img_size):
        gray = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    if use_clahe:
        gray = apply_clahe(gray)
    normalized = normalize(gray)
    return normalized.reshape(img_size, img_size, 1)
