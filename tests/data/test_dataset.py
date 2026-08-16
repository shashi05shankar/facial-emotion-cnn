import numpy as np
import cv2

from facial_emotion.constants import EMOTION_LABELS
from facial_emotion.data.dataset import build_datasets


def _write_fake_split(root, n_per_class: int, seed_offset: int = 0):
    rng = np.random.default_rng(seed_offset)
    for cls in EMOTION_LABELS:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            img = rng.integers(0, 256, (48, 48), dtype=np.uint8)
            cv2.imwrite(str(cls_dir / f"{i}.png"), img)


def _all_images(ds):
    return np.concatenate([x.numpy() for x, _ in ds], axis=0)


def test_build_datasets_carves_disjoint_test_split_when_no_test_folder(tmp_path):
    data_dir = tmp_path / "data"
    _write_fake_split(data_dir / "train", n_per_class=20)

    train_ds, val_ds, test_ds, class_names = build_datasets(
        data_dir, batch_size=8, val_split=0.2, test_split=0.2, seed=1
    )

    assert class_names == EMOTION_LABELS
    train_imgs = _all_images(train_ds)
    val_imgs = _all_images(val_ds)
    test_imgs = _all_images(test_ds)

    assert len(train_imgs) > 0 and len(val_imgs) > 0 and len(test_imgs) > 0
    assert len(train_imgs) + len(val_imgs) + len(test_imgs) == 20 * len(EMOTION_LABELS)

    # No exact-duplicate rows across splits (random per-pixel images make
    # accidental collisions astronomically unlikely).
    def as_set(arr):
        return {tuple(row.round(4).tolist()) for row in arr.reshape(len(arr), -1)}

    assert as_set(train_imgs) & as_set(test_imgs) == set()
    assert as_set(val_imgs) & as_set(test_imgs) == set()
    assert as_set(train_imgs) & as_set(val_imgs) == set()


def test_build_datasets_uses_provided_test_folder_when_present(tmp_path):
    data_dir = tmp_path / "data"
    _write_fake_split(data_dir / "train", n_per_class=10, seed_offset=1)
    _write_fake_split(data_dir / "test", n_per_class=5, seed_offset=2)

    train_ds, val_ds, test_ds, class_names = build_datasets(data_dir, batch_size=8, val_split=0.2, seed=1)

    test_imgs = _all_images(test_ds)
    assert len(test_imgs) == 5 * len(EMOTION_LABELS)
