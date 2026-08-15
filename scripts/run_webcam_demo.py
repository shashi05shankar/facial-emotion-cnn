"""Launch the real-time webcam emotion detection demo.

Requires artifacts/model.keras (from notebooks/kaggle_train.ipynb) and a
connected webcam.
"""
from __future__ import annotations

from pathlib import Path

from facial_emotion.app.webcam_app import run

MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "model.keras"


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"No trained model at {MODEL_PATH}. Run notebooks/kaggle_train.ipynb "
            "on Kaggle first and download model.keras into artifacts/."
        )
    run(MODEL_PATH)


if __name__ == "__main__":
    main()
