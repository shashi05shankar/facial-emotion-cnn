"""Evaluation utilities: run a trained model over a labeled tf.data
dataset and produce a metrics dict + confusion matrix plot."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model: tf.keras.Model, dataset: tf.data.Dataset, class_names: list[str]) -> dict:
    """Run `model` over every batch in `dataset` and compute real metrics
    (no training happens here)."""
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_true.extend(labels.numpy().tolist())

    labels = list(range(len(class_names)))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))

    return {
        "test_accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "num_samples": len(y_true),
    }


def plot_confusion_matrix(cm: list[list[int]] | np.ndarray, class_names: list[str], out_path: str | Path) -> None:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("FER2013 test confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
