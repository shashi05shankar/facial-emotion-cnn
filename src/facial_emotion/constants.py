IMG_SIZE = 48
NUM_CLASSES = 7

# Canonical FER2013 numeric label order: emotion at integer label i is
# EMOTION_LABELS[i]. This matches the original FER2013 CSV encoding and the
# numeric class folders ('0'..'6') used by the image-folder datasets. The
# trained model outputs probabilities in this order, so inference/eval/webcam
# must all use it.
EMOTION_LABELS = [
    "angry",     # 0
    "disgust",   # 1
    "fear",      # 2
    "happy",     # 3
    "sad",       # 4
    "surprise",  # 5
    "neutral",   # 6
]

# Map a class-folder name (numeric '0'..'6' OR an emotion name in any case)
# to its canonical FER2013 label index, so datasets with either folder-naming
# convention train a model with consistent, correctly-labeled outputs.
_EMOTION_TO_INDEX = {name: i for i, name in enumerate(EMOTION_LABELS)}


def fer_index_for_folder(folder_name: str) -> int:
    name = folder_name.strip().lower()
    if name.isdigit():
        idx = int(name)
        if not 0 <= idx < NUM_CLASSES:
            raise ValueError(f"Numeric class folder {folder_name!r} out of range 0..{NUM_CLASSES - 1}")
        return idx
    if name in _EMOTION_TO_INDEX:
        return _EMOTION_TO_INDEX[name]
    raise ValueError(f"Unrecognized class folder {folder_name!r}; expected 0..6 or one of {EMOTION_LABELS}")
