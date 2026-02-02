import json
from pathlib import Path

import numpy as np
from tensorflow import keras


def build_model(num_classes: int, input_shape: tuple[int, int, int]) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv2D(32, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(128, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def save_labels(labels: list[str], path: Path) -> None:
    path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")


def load_labels(path: Path) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))


def predict_labels(model: keras.Model, labels: list[str], batch: np.ndarray) -> list[str]:
    predictions = model.predict(batch, verbose=0)
    indices = predictions.argmax(axis=1)
    return [labels[index] for index in indices]
