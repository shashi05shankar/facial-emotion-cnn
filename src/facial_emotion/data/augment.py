"""Training-time data augmentation, applied only to the training split."""
from __future__ import annotations

import tensorflow as tf


def build_augmentation_layer() -> tf.keras.Sequential:
    """Random rotation/shift/zoom/flip block, mirroring the resume bullet's
    "trained with data augmentation" step. Applied inline in the model so it
    only runs during training (`training=True`) and is a no-op at inference.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomFlip("horizontal"),
        ],
        name="augmentation",
    )
