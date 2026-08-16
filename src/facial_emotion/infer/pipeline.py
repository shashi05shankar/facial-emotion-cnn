"""Face detection + preprocessing + CNN inference, chained into a single
per-frame call used by both the webcam app and offline single-image
inference."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from facial_emotion.constants import EMOTION_LABELS, IMG_SIZE
from facial_emotion.data.preprocessing import preprocess_image


@dataclass
class FacePrediction:
    bbox: tuple[int, int, int, int]  # x, y, w, h in the original frame
    probabilities: np.ndarray  # shape (NUM_CLASSES,), sums to 1
    label: str


class EmotionPipeline:
    """Wraps a Haar-cascade face detector and a trained emotion CNN."""

    def __init__(self, model: tf.keras.Model, use_clahe: bool = True, min_face_size: int = 48):
        self.model = model
        self.use_clahe = use_clahe
        self.min_face_size = min_face_size
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_detector = cv2.CascadeClassifier(cascade_path)
        if self.face_detector.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")

    def detect_faces(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
        )
        return [tuple(int(v) for v in f) for f in faces]

    def predict_face(self, frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> FacePrediction:
        x, y, w, h = bbox
        crop = frame_bgr[y : y + h, x : x + w]
        model_input = preprocess_image(crop, img_size=IMG_SIZE, use_clahe=self.use_clahe)
        probs = self.model.predict(model_input[np.newaxis, ...], verbose=0)[0]
        label = EMOTION_LABELS[int(np.argmax(probs))]
        return FacePrediction(bbox=bbox, probabilities=probs, label=label)

    def process_frame(self, frame_bgr: np.ndarray) -> list[FacePrediction]:
        """Detect every face in `frame_bgr` and return a prediction for each."""
        return [self.predict_face(frame_bgr, bbox) for bbox in self.detect_faces(frame_bgr)]


def load_pipeline(model_path: str | Path, use_clahe: bool = True) -> EmotionPipeline:
    model = tf.keras.models.load_model(model_path)
    return EmotionPipeline(model, use_clahe=use_clahe)
