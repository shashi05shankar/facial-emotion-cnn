"""Download FER2013 (Kaggle: msambare/fer2013) into data/raw/.

Requires Kaggle API credentials configured (~/.kaggle/kaggle.json or
KAGGLE_USERNAME/KAGGLE_KEY env vars). See
https://github.com/Kaggle/kaggle-api#api-credentials
"""
from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    print("Downloading msambare/fer2013 via kagglehub ...")
    cache_path = Path(kagglehub.dataset_download("msambare/fer2013"))
    print(f"Downloaded to cache: {cache_path}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        src = cache_path / split
        dst = DATA_DIR / split
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)
            print(f"Copied {src} -> {dst}")
        elif not src.exists():
            print(f"WARNING: expected folder {src} not found in dataset download")

    print(f"Done. Dataset ready at {DATA_DIR}")


if __name__ == "__main__":
    main()
