"""Download an FER2013 image-folder dataset into data/raw/{train,val,test}.

Requires Kaggle API credentials configured (~/.kaggle/kaggle.json or
KAGGLE_USERNAME/KAGGLE_KEY env vars). See
https://github.com/Kaggle/kaggle-api#api-credentials

Default source: astraszab/facial-expression-dataset-image-folders-fer2013,
which ships a proper train/val/test split with numeric class folders
('0'..'6', the canonical FER2013 label encoding). Auto-detects the split
parent inside the downloaded cache so it's robust to whatever nesting the
dataset uses.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub

from facial_emotion.constants import EMOTION_LABELS

DATASET_SLUG = "astraszab/facial-expression-dataset-image-folders-fer2013"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

_NUMERIC_CLASSES = {str(i) for i in range(len(EMOTION_LABELS))}
_NAMED_CLASSES = {e.lower() for e in EMOTION_LABELS}


def _is_class_dir(d: Path) -> bool:
    children = {c.name.strip().lower() for c in d.iterdir() if c.is_dir()}
    return _NUMERIC_CLASSES.issubset(children) or _NAMED_CLASSES.issubset(children)


def find_split_parent(root: Path) -> Path | None:
    """Find a directory that has 'train' and 'test' subfolders which each
    contain the 7 FER class folders."""
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        subs = {c.name.lower(): c for c in p.iterdir() if c.is_dir()}
        if "train" in subs and "test" in subs and _is_class_dir(subs["train"]) and _is_class_dir(subs["test"]):
            return p
    return None


def find_single_class_parent(root: Path) -> Path | None:
    best, best_n = None, -1
    for p in root.rglob("*"):
        if p.is_dir() and _is_class_dir(p):
            n = sum(1 for c in p.rglob("*") if c.is_file())
            if n > best_n:
                best, best_n = p, n
    return best


def _copy(src: Path, split_name: str) -> None:
    dst = DATA_DIR / split_name
    if dst.exists():
        print(f"{dst} already exists, skipping copy")
    else:
        shutil.copytree(src, dst)
        print(f"Copied {src} -> {dst}")


def main() -> None:
    print(f"Downloading {DATASET_SLUG} via kagglehub ...")
    cache_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    print(f"Downloaded to cache: {cache_path}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    parent = find_split_parent(cache_path)

    if parent is not None:
        subs = {c.name.lower(): c for c in parent.iterdir() if c.is_dir()}
        _copy(subs["train"], "train")
        _copy(subs["test"], "test")
        if "val" in subs:
            _copy(subs["val"], "val")
        print(f"Done. Dataset ready at {DATA_DIR}")
        return

    pool = find_single_class_parent(cache_path)
    if pool is None:
        raise SystemExit(
            f"Could not locate FER class folders (numeric 0..6 or {sorted(_NAMED_CLASSES)}) "
            f"under {cache_path}. Inspect the download and update this script."
        )
    print(f"No train/test split in dataset — copying single pool {pool} -> data/raw/train")
    print("build_datasets() will carve a disjoint held-out test split out of it.")
    _copy(pool, "train")
    print(f"Done. Dataset ready at {DATA_DIR}")


if __name__ == "__main__":
    main()
