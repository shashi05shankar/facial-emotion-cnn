import numpy as np
import tensorflow as tf

from facial_emotion.constants import IMG_SIZE, NUM_CLASSES
from facial_emotion.eval.metrics import evaluate_model


def _synthetic_dataset(num_samples: int = 12, batch_size: int = 4):
    x = np.random.rand(num_samples, IMG_SIZE, IMG_SIZE, 1).astype("float32")
    y = np.random.randint(0, NUM_CLASSES, size=(num_samples,))
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size), y


def test_evaluate_model_produces_full_report():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    class_names = [f"class_{i}" for i in range(NUM_CLASSES)]
    dataset, y_true = _synthetic_dataset()

    metrics = evaluate_model(model, dataset, class_names)

    assert 0.0 <= metrics["test_accuracy"] <= 1.0
    assert metrics["num_samples"] == len(y_true)
    assert metrics["class_names"] == class_names
    cm = np.array(metrics["confusion_matrix"])
    assert cm.shape == (NUM_CLASSES, NUM_CLASSES)
    assert cm.sum() == len(y_true)
    assert set(metrics["classification_report"].keys()) >= set(class_names)
