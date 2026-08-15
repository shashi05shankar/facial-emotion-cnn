# Facial Emotion Detection using CNN

Deep CNN for 7-class facial emotion recognition on FER2013, with a
CLAHE/grayscale/normalization preprocessing pipeline and a real-time OpenCV
webcam demo.

> Status: scaffolding in progress. Real metrics and demo instructions land
> once the Kaggle training run completes (see `notebooks/kaggle_train.ipynb`).

## Dataset

[FER2013, image-folder layout](https://www.kaggle.com/datasets/astraszab/facial-expression-dataset-image-folders-fer2013) — 35,887
grayscale 48x48 face crops, 7 classes: angry, disgust, fear, happy, sad,
surprise, neutral. Uses the dataset's own train/test split.

## Quick Start

```bash
pip install -e ".[dev]"
python scripts/download_dataset.py
pytest
python scripts/run_webcam_demo.py
```
