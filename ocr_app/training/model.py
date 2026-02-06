from __future__ import annotations

from tensorflow import keras


def build_cnn(input_shape: tuple[int, int, int], num_classes: int) -> keras.Model:
    augmentation = keras.Sequential(
        [
            keras.layers.RandomRotation(0.05),
            keras.layers.RandomTranslation(0.06, 0.06),
            keras.layers.RandomZoom(0.1),
            keras.layers.RandomContrast(0.1),
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
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(384, 3, padding="same", activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.35),
            keras.layers.Dense(384, activation="relu"),
            keras.layers.Dropout(0.35),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def compile_model(
    model: keras.Model,
    learning_rate: float,
    label_smoothing: float,
) -> None:
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    try:
        loss = keras.losses.SparseCategoricalCrossentropy(label_smoothing=label_smoothing)
    except TypeError:
        loss = keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])