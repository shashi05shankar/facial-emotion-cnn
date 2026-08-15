"""CNN architecture: stacked Conv2D/BatchNorm/MaxPooling blocks with Dropout,
matching the resume bullet's named layers."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers

from facial_emotion.constants import IMG_SIZE, NUM_CLASSES
from facial_emotion.data.augment import build_augmentation_layer


def _conv_block(x, filters: int, dropout: float):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout)(x)
    return x


def build_cnn(
    input_shape: tuple[int, int, int] = (IMG_SIZE, IMG_SIZE, 1),
    num_classes: int = NUM_CLASSES,
    use_augmentation: bool = True,
) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)
    x = inputs
    if use_augmentation:
        x = build_augmentation_layer()(x)

    x = _conv_block(x, 32, dropout=0.25)
    x = _conv_block(x, 64, dropout=0.25)
    x = _conv_block(x, 128, dropout=0.30)

    x = layers.Flatten()(x)
    x = layers.Dense(256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="facial_emotion_cnn")


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-3) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
