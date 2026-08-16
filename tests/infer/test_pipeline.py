import numpy as np
import tensorflow as tf

from facial_emotion.constants import EMOTION_LABELS, IMG_SIZE, NUM_CLASSES
from facial_emotion.infer.pipeline import EmotionPipeline


def _dummy_model():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )


def test_predict_face_returns_valid_probability_distribution():
    pipeline = EmotionPipeline(_dummy_model())
    frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    bbox = (20, 20, 80, 80)

    pred = pipeline.predict_face(frame, bbox)

    assert pred.bbox == bbox
    assert pred.probabilities.shape == (NUM_CLASSES,)
    assert np.isclose(pred.probabilities.sum(), 1.0, atol=1e-4)
    assert pred.label in EMOTION_LABELS


def test_detect_faces_returns_list_of_bboxes():
    pipeline = EmotionPipeline(_dummy_model())
    frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)

    faces = pipeline.detect_faces(frame)

    assert isinstance(faces, list)
    for bbox in faces:
        assert len(bbox) == 4


def test_process_frame_predicts_for_every_detected_face(monkeypatch):
    pipeline = EmotionPipeline(_dummy_model())
    fake_boxes = [(0, 0, 48, 48), (50, 50, 48, 48)]
    monkeypatch.setattr(pipeline, "detect_faces", lambda frame: fake_boxes)
    frame = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)

    predictions = pipeline.process_frame(frame)

    assert len(predictions) == len(fake_boxes)
    assert [p.bbox for p in predictions] == fake_boxes
