import cv2
import numpy as np

from facial_emotion.constants import EMOTION_LABELS, NUM_CLASSES
from facial_emotion.data.dataset import build_datasets


def _write_numeric_split(root, n_per_class: int, seed_offset: int = 0):
    """Write a FER-style split with numeric class folders '0'..'6'."""
    rng = np.random.default_rng(seed_offset)
    for cls in range(NUM_CLASSES):
        cls_dir = root / str(cls)
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            img = rng.integers(0, 256, (48, 48), dtype=np.uint8)
            cv2.imwrite(str(cls_dir / f"{i}.png"), img)


def _all(ds):
    xs = np.concatenate([x.numpy() for x, _ in ds], axis=0)
    ys = np.concatenate([y.numpy() for _, y in ds], axis=0)
    return xs, ys


def _as_set(arr):
    return {tuple(row.round(4).tolist()) for row in arr.reshape(len(arr), -1)}


def test_carves_disjoint_test_split_when_no_test_folder(tmp_path):
    data_dir = tmp_path / "data"
    _write_numeric_split(data_dir / "train", n_per_class=20)

    train_ds, val_ds, test_ds, class_names = build_datasets(
        data_dir, batch_size=8, val_split=0.2, test_split=0.2, seed=1
    )

    assert class_names == list(EMOTION_LABELS)
    x_train, _ = _all(train_ds)
    x_val, _ = _all(val_ds)
    x_test, y_test = _all(test_ds)

    assert len(x_train) + len(x_val) + len(x_test) == 20 * NUM_CLASSES
    assert set(y_test.tolist()).issubset(set(range(NUM_CLASSES)))
    # disjoint splits — no leakage
    assert _as_set(x_train) & _as_set(x_test) == set()
    assert _as_set(x_val) & _as_set(x_test) == set()
    assert _as_set(x_train) & _as_set(x_val) == set()


def test_uses_provided_test_and_val_folders(tmp_path):
    data_dir = tmp_path / "data"
    _write_numeric_split(data_dir / "train", n_per_class=10, seed_offset=1)
    _write_numeric_split(data_dir / "val", n_per_class=4, seed_offset=2)
    _write_numeric_split(data_dir / "test", n_per_class=5, seed_offset=3)

    train_ds, val_ds, test_ds, class_names = build_datasets(data_dir, batch_size=8, seed=1)

    x_train, _ = _all(train_ds)
    x_val, _ = _all(val_ds)
    x_test, _ = _all(test_ds)
    assert len(x_train) == 10 * NUM_CLASSES
    assert len(x_val) == 4 * NUM_CLASSES
    assert len(x_test) == 5 * NUM_CLASSES


def test_numeric_folders_map_to_canonical_fer_index(tmp_path):
    """Folder '4' must become label 4 (sad), not an alphabetical position."""
    data_dir = tmp_path / "data"
    for cls in range(NUM_CLASSES):
        (data_dir / "train" / str(cls)).mkdir(parents=True, exist_ok=True)
    # only populate class '4'; every loaded label must therefore equal 4
    rng = np.random.default_rng(0)
    for i in range(12):
        cv2.imwrite(str(data_dir / "train" / "4" / f"{i}.png"), rng.integers(0, 256, (48, 48), dtype=np.uint8))

    train_ds, val_ds, test_ds, class_names = build_datasets(
        data_dir, batch_size=8, val_split=0.2, test_split=0.2, seed=1
    )
    assert class_names[4] == "sad"
    all_labels = np.concatenate(
        [y.numpy() for ds in (train_ds, val_ds, test_ds) for _, y in ds]
    )
    assert set(all_labels.tolist()) == {4}
