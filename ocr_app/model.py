import json
from pathlib import Path

import numpy as np
from tensorflow import keras


def build_model(
    num_classes: int,
    input_shape: tuple[int, int, int],
    learning_rate: float = 1e-3,
    label_smoothing: float = 0.0,
) -> keras.Model:
    augmentation = keras.Sequential(
        [
            keras.layers.RandomRotation(0.05),
            keras.layers.RandomTranslation(0.05, 0.05),
            keras.layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )
    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            augmentation,
            keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(256, 3, padding="same", activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    if label_smoothing > 0:
        try:
            loss = keras.losses.SparseCategoricalCrossentropy(label_smoothing=label_smoothing)
        except TypeError:
            print("Label smoothing is not supported in this TensorFlow version. Continuing without it.")
            loss = keras.losses.SparseCategoricalCrossentropy()
    else:
        loss = keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


def save_labels(labels: list[str], path: Path) -> None:
    path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")


def load_labels(path: Path) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))


def predict_labels(model: keras.Model, labels: list[str], batch: np.ndarray) -> list[str]:
    predictions = model.predict(batch, verbose=0)
    indices = predictions.argmax(axis=1)
    return [labels[index] for index in indices]
