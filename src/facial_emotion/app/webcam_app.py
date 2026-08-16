"""Real-time webcam demo: detect faces, run the emotion CNN, overlay the
bounding box, top label, and a live 7-class probability bar chart."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from facial_emotion.constants import EMOTION_LABELS
from facial_emotion.infer.pipeline import EmotionPipeline, load_pipeline

BAR_PANEL_WIDTH = 220
BAR_HEIGHT = 18
BAR_MAX_LEN = 180


def draw_probability_bars(panel: np.ndarray, probabilities: np.ndarray) -> None:
    panel[:] = (30, 30, 30)
    for i, (label, prob) in enumerate(zip(EMOTION_LABELS, probabilities)):
        y = 10 + i * (BAR_HEIGHT + 6)
        bar_len = int(prob * BAR_MAX_LEN)
        cv2.rectangle(panel, (90, y), (90 + bar_len, y + BAR_HEIGHT), (80, 200, 80), -1)
        cv2.putText(panel, label, (4, y + BAR_HEIGHT - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(
            panel, f"{prob:.2f}", (95 + bar_len, y + BAR_HEIGHT - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
        )


def render_frame(frame_bgr: np.ndarray, pipeline: EmotionPipeline) -> np.ndarray:
    """Run the pipeline on one frame and return an annotated frame with a
    side panel showing the 7-class probability bars for the most prominent
    face (largest bbox). Pure function — no camera/window dependency, so
    it's directly testable on a still image."""
    predictions = pipeline.process_frame(frame_bgr)
    annotated = frame_bgr.copy()

    for pred in predictions:
        x, y, w, h = pred.bbox
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (80, 200, 80), 2)
        cv2.putText(annotated, pred.label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 200, 80), 2)

    panel = np.zeros((annotated.shape[0], BAR_PANEL_WIDTH, 3), dtype=np.uint8)
    if predictions:
        largest = max(predictions, key=lambda p: p.bbox[2] * p.bbox[3])
        draw_probability_bars(panel, largest.probabilities)
    else:
        panel[:] = (30, 30, 30)

    return np.hstack([annotated, panel])


def run(model_path: str | Path, camera_index: int = 0) -> None:
    pipeline = load_pipeline(model_path)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    print("Press 'q' to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            output = render_frame(frame, pipeline)
            cv2.imshow("Facial Emotion Detection (press q to quit)", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
