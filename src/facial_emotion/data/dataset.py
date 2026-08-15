"""Build tf.data pipelines from the FER2013 train/test folder layout
(Kaggle `msambare/fer2013`: train/<emotion>/*.jpg, test/<emotion>/*.jpg).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from facial_emotion.constants import EMOTION_LABELS, IMG_SIZE
from facial_emotion.data.preprocessing import apply_clahe, normalize

AUTOTUNE = tf.data.AUTOTUNE


def _clahe_normalize_batch(images: np.ndarray) -> np.ndarray:
    processed = np.empty_like(images, dtype=np.float32)
    for i in range(images.shape[0]):
        gray = images[i, ..., 0].astype(np.uint8)
        gray = apply_clahe(gray)
        processed[i, ..., 0] = normalize(gray)
    return processed


def _preprocess_batch(images: tf.Tensor, labels: tf.Tensor, img_size: int, use_clahe: bool):
    if use_clahe:
        images = tf.py_function(_clahe_normalize_batch, [images], tf.float32)
    else:
        images = tf.cast(images, tf.float32) / 255.0
    images.set_shape([None, img_size, img_size, 1])
    return images, labels


def build_datasets(
    data_dir: str | Path,
    img_size: int = IMG_SIZE,
    batch_size: int = 64,
    val_split: float = 0.1,
    use_clahe: bool = True,
    seed: int = 42,
):
    """Return (train_ds, val_ds, test_ds, class_names).

    train/val come from `data_dir/train` via a stratified-by-shuffle split;
    test comes from the dataset's own held-out `data_dir/test` folder.
    """
    data_dir = Path(data_dir)
    common = dict(
        image_size=(img_size, img_size),
        color_mode="grayscale",
        batch_size=batch_size,
        label_mode="int",
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train", validation_split=val_split, subset="training", seed=seed, **common
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train", validation_split=val_split, subset="validation", seed=seed, **common
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "test", shuffle=False, **common
    )

    class_names = train_ds.class_names
    if class_names != EMOTION_LABELS:
        raise ValueError(
            f"Dataset class folders {class_names} don't match expected "
            f"EMOTION_LABELS {EMOTION_LABELS} — check the download layout."
        )

    def prep(ds, shuffle_buffer: int | None = None):
        ds = ds.map(
            lambda x, y: _preprocess_batch(x, y, img_size, use_clahe),
            num_parallel_calls=AUTOTUNE,
        )
        return ds.prefetch(AUTOTUNE)

    return prep(train_ds), prep(val_ds), prep(test_ds), class_names
