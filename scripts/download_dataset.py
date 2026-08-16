"""Download an FER2013 image-folder dataset into data/raw/{train,test}.

Requires Kaggle API credentials configured (~/.kaggle/kaggle.json or
KAGGLE_USERNAME/KAGGLE_KEY env vars). See
https://github.com/Kaggle/kaggle-api#api-credentials

Default source: astraszab/facial-expression-dataset-image-folders-fer2013
(35,887 files, matches the original FER2013 train/PublicTest/PrivateTest
split count). Auto-detects the actual train/test folder paths inside the
downloaded cache so it's robust to whatever nesting the dataset uses.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub

from facial_emotion.constants import EMOTION_LABELS

DATASET_SLUG = "astraszab/facial-expression-dataset-image-folders-fer2013"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
EMOTION_SET = {e.lower() for e in EMOTION_LABELS}


def find_split_folders(root: Path) -> list[tuple[Path, int]]:
    hits = []
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        child_names = {c.name.lower() for c in p.iterdir() if c.is_dir()}
        if EMOTION_SET.issubset(child_names):
            n_images = sum(1 for c in p.rglob("*") if c.is_file())
            hits.append((p, n_images))
    return sorted(hits, key=lambda h: -h[1])


def pick(hits: list[tuple[Path, int]], keyword_groups: list[list[str]], exclude: Path | None = None) -> Path | None:
    for keywords in keyword_groups:
        for p, _ in hits:
            if exclude is not None and p == exclude:
                continue
            if any(k in p.name.lower() for k in keywords):
                return p
    return None


def main() -> None:
    print(f"Downloading {DATASET_SLUG} via kagglehub ...")
    cache_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    print(f"Downloaded to cache: {cache_path}")

    hits = find_split_folders(cache_path)
    if not hits:
        raise SystemExit(
            f"Could not find a folder with subfolders {sorted(EMOTION_SET)} under {cache_path}. "
            "Inspect the download manually and update this script's detection logic."
        )

    train_src = pick(hits, [["train", "training"]]) or hits[0][0]
    test_src = pick(
        hits, [["privatetest"], ["publictest"], ["test", "testing", "val", "validation"]], exclude=train_src
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dst = DATA_DIR / "train"
    if dst.exists():
        print(f"{dst} already exists, skipping copy")
    else:
        shutil.copytree(train_src, dst)
        print(f"Copied {train_src} -> {dst}")

    if test_src is None:
        # No distinct test folder in this dataset — leave data/raw/test
        # absent rather than duplicating train_src into it (which would
        # leak training images into "test"). build_datasets() carves a
        # proper held-out test split out of train/ itself in this case.
        print(
            "No distinct test folder found in this dataset — data/raw/test "
            "will NOT be created. build_datasets() will carve train/val/test "
            "out of data/raw/train instead."
        )
    else:
        dst = DATA_DIR / "test"
        if dst.exists():
            print(f"{dst} already exists, skipping copy")
        else:
            shutil.copytree(test_src, dst)
            print(f"Copied {test_src} -> {dst}")

    print(f"Done. Dataset ready at {DATA_DIR}")


if __name__ == "__main__":
    main()
