import json
from pathlib import Path

import numpy as np
from tensorflow import keras


def save_labels(labels: list[str], path: Path) -> None:
    path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")


def load_labels(path: Path) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))


def predict_labels(model: keras.Model, labels: list[str], batch: np.ndarray) -> list[str]:
    predictions = model.predict(batch, verbose=0)
    indices = predictions.argmax(axis=1)
    return [labels[index] for index in indices]
