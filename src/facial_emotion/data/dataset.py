"""Build tf.data pipelines from an FER2013 image-folder dataset.

Handles both folder-naming conventions — numeric ('0'..'6') and emotion
names — by mapping every class folder to its canonical FER2013 label index
(see `fer_index_for_folder`), so the trained model's output indices always
line up with `EMOTION_LABELS` regardless of the source dataset.

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

from facial_emotion.constants import EMOTION_LABELS, IMG_SIZE, fer_index_for_folder
from facial_emotion.data.preprocessing import apply_clahe, normalize

AUTOTUNE = tf.data.AUTOTUNE
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_split_as_arrays(directory: Path, img_size: int, use_clahe: bool):
    """Load every class folder under `directory`, labeling by canonical FER
    index. Returns (x, y) with x shape (N, img_size, img_size, 1)."""
    images, labels = [], []
    for cls_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
        label = fer_index_for_folder(cls_dir.name)
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
            labels.append(label)
    x = np.asarray(images, dtype=np.float32).reshape(-1, img_size, img_size, 1)
    y = np.asarray(labels, dtype=np.int64)
    return x, y


def _to_ds(x, y, batch_size, shuffle, seed):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(min(len(x), 4096), seed=seed)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def build_datasets(
    data_dir: str | Path,
    img_size: int = IMG_SIZE,
    batch_size: int = 64,
    val_split: float = 0.1,
    test_split: float = 0.1,
    use_clahe: bool = True,
    seed: int = 42,
):
    """Return (train_ds, val_ds, test_ds, class_names).

    Layout resolution under `data_dir`:
      * `test/` present -> held-out test is that folder; val is `val/` if it
        exists, else a shuffled slice of `train/`.
      * no `test/` -> the source dataset shipped a single pooled `train/`;
        three disjoint slices are carved out of it so test never overlaps
        train/val.
    class_names is always `EMOTION_LABELS` (canonical FER order).
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    val_dir = data_dir / "val"

    rng = np.random.default_rng(seed)

    if test_dir.exists():
        x_train, y_train = load_split_as_arrays(train_dir, img_size, use_clahe)
        x_test, y_test = load_split_as_arrays(test_dir, img_size, use_clahe)
        if val_dir.exists():
            x_val, y_val = load_split_as_arrays(val_dir, img_size, use_clahe)
        else:
            perm = rng.permutation(len(x_train))
            x_train, y_train = x_train[perm], y_train[perm]
            n_val = int(len(x_train) * val_split)
            x_val, y_val = x_train[:n_val], y_train[:n_val]
            x_train, y_train = x_train[n_val:], y_train[n_val:]
    else:
        x_pool, y_pool = load_split_as_arrays(train_dir, img_size, use_clahe)
        perm = rng.permutation(len(x_pool))
        x_pool, y_pool = x_pool[perm], y_pool[perm]
        n_val = int(len(x_pool) * val_split)
        n_test = int(len(x_pool) * test_split)
        x_val, y_val = x_pool[:n_val], y_pool[:n_val]
        x_test, y_test = x_pool[n_val : n_val + n_test], y_pool[n_val : n_val + n_test]
        x_train, y_train = x_pool[n_val + n_test :], y_pool[n_val + n_test :]

    train_ds = _to_ds(x_train, y_train, batch_size, shuffle=True, seed=seed)
    val_ds = _to_ds(x_val, y_val, batch_size, shuffle=False, seed=seed)
    test_ds = _to_ds(x_test, y_test, batch_size, shuffle=False, seed=seed)

    return train_ds, val_ds, test_ds, list(EMOTION_LABELS)
