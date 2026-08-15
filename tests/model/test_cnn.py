import numpy as np

from facial_emotion.constants import IMG_SIZE, NUM_CLASSES
from facial_emotion.model.cnn import build_cnn, compile_model


def test_build_cnn_output_shape():
    model = build_cnn()
    batch = np.random.rand(4, IMG_SIZE, IMG_SIZE, 1).astype("float32")
    out = model(batch, training=False)
    assert out.shape == (4, NUM_CLASSES)


def test_build_cnn_output_is_probability_distribution():
    model = build_cnn()
    batch = np.random.rand(2, IMG_SIZE, IMG_SIZE, 1).astype("float32")
    out = model(batch, training=False).numpy()
    row_sums = out.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-4)


def test_build_cnn_without_augmentation_layer():
    model = build_cnn(use_augmentation=False)
    layer_names = [layer.name for layer in model.layers]
    assert "augmentation" not in layer_names


def test_compile_model_runs_one_training_step():
    model = compile_model(build_cnn())
    x = np.random.rand(8, IMG_SIZE, IMG_SIZE, 1).astype("float32")
    y = np.random.randint(0, NUM_CLASSES, size=(8,))
    history = model.fit(x, y, epochs=1, verbose=0)
    assert "loss" in history.history
