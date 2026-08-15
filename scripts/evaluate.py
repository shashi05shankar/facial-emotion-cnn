"""Re-evaluate a saved model against the FER2013 test split and refresh
artifacts/metrics.json + artifacts/confusion_matrix.png. Run this locally
after downloading model.keras from the Kaggle training run.
"""
from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf

from facial_emotion.constants import IMG_SIZE
from facial_emotion.data.dataset import build_datasets
from facial_emotion.eval.metrics import evaluate_model, plot_confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
ARTIFACTS_DIR = ROOT / "artifacts"


def main() -> None:
    model_path = ARTIFACTS_DIR / "model.keras"
    if not model_path.exists():
        raise SystemExit(
            f"No trained model at {model_path}. Run notebooks/kaggle_train.ipynb "
            "on Kaggle first and download model.keras into artifacts/."
        )

    model = tf.keras.models.load_model(model_path)
    _, _, test_ds, class_names = build_datasets(DATA_DIR, img_size=IMG_SIZE, use_clahe=True)

    metrics = evaluate_model(model, test_ds, class_names)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(metrics["confusion_matrix"], class_names, ARTIFACTS_DIR / "confusion_matrix.png")

    print(f"Test accuracy: {metrics['test_accuracy']:.4f} on {metrics['num_samples']} samples")
    print(f"Wrote {ARTIFACTS_DIR / 'metrics.json'} and {ARTIFACTS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
