import numpy as np
import tensorflow as tf

from facial_emotion.app.webcam_app import BAR_PANEL_WIDTH, render_frame
from facial_emotion.constants import IMG_SIZE, NUM_CLASSES
from facial_emotion.infer.pipeline import EmotionPipeline


def _dummy_model():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )


def test_render_frame_adds_probability_panel_no_faces():
    pipeline = EmotionPipeline(_dummy_model())
    frame = np.zeros((100, 150, 3), dtype=np.uint8)  # blank frame, no faces detected

    output = render_frame(frame, pipeline)

    assert output.shape == (100, 150 + BAR_PANEL_WIDTH, 3)


def test_render_frame_with_mocked_face(monkeypatch):
    pipeline = EmotionPipeline(_dummy_model())
    monkeypatch.setattr(pipeline, "detect_faces", lambda frame: [(10, 10, 48, 48)])
    frame = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)

    output = render_frame(frame, pipeline)

    assert output.shape == (120, 160 + BAR_PANEL_WIDTH, 3)
    # bounding box drawing should have modified pixels near the box edge
    assert not np.array_equal(output[10, 10:58], frame[10, 10:58])
