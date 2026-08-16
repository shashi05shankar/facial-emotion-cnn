"""Build tf.data pipelines from the FER2013 train/test folder layout
(class-named subfolders under `train/` and `test/`).

Preprocessing (CLAHE/normalize) is applied with plain NumPy/OpenCV before
building the `tf.data.Dataset`, rather than via `tf.py_function` inside
`.map()` — FER2013 is small enough to fully materialize in memory, and this
avoids a Keras 3 + `tf.py_function` incompatibility inside `.map()`
(`OptionalFromValue ... length 0`) seen in `model.fit`.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from facial_emotion.constants import EMOTION_LABELS, IMG_SIZE
from facial_emotion.data.preprocessing import apply_clahe, normalize

AUTOTUNE = tf.data.AUTOTUNE
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_split_as_arrays(directory: Path, class_names: list[str], img_size: int, use_clahe: bool):
    images, labels = [], []
    for idx, cls in enumerate(class_names):
        cls_dir = directory / cls
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (img_size, img_size):
                img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            if use_clahe:
                img = apply_clahe(img)
            images.append(normalize(img))
            labels.append(idx)
    x = np.asarray(images, dtype=np.float32).reshape(-1, img_size, img_size, 1)
    y = np.asarray(labels, dtype=np.int64)
    return x, y


def build_datasets(
    data_dir: str | Path,
    img_size: int = IMG_SIZE,
    batch_size: int = 64,
    val_split: float = 0.1,
    use_clahe: bool = True,
    seed: int = 42,
):
    """Return (train_ds, val_ds, test_ds, class_names).

    train/val come from `data_dir/train` via a shuffled split; test comes
    from the dataset's own held-out `data_dir/test` folder.
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    class_names = sorted(c.name for c in train_dir.iterdir() if c.is_dir())
    if class_names != EMOTION_LABELS:
        raise ValueError(
            f"Dataset class folders {class_names} don't match expected "
            f"EMOTION_LABELS {EMOTION_LABELS} — check the download layout."
        )

    x_all, y_all = load_split_as_arrays(train_dir, class_names, img_size, use_clahe)
    x_test, y_test = load_split_as_arrays(test_dir, class_names, img_size, use_clahe)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x_all))
    x_all, y_all = x_all[perm], y_all[perm]
    n_val = int(len(x_all) * val_split)
    x_val, y_val = x_all[:n_val], y_all[:n_val]
    x_train, y_train = x_all[n_val:], y_all[n_val:]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(min(len(x_train), 4096), seed=seed)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).prefetch(AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
