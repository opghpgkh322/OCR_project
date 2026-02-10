from __future__ import annotations
from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(input_shape: tuple[int, int, int], num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=input_shape)

    # === 1. Легкая аугментация (прямо в модели) ===
    # Не перегружаем CPU, но добавляем вариативности
    x = layers.RandomRotation(0.06)(inputs)
    x = layers.RandomTranslation(0.08, 0.08)(x)
    x = layers.RandomZoom(0.05)(x)

    # === 2. Входной блок ===
    # Обычная свертка только один раз в начале
    x = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # === 3. Основные блоки (SeparableConv2D для скорости) ===
    # Используем Residual Connections (связи "через голову") для высокой точности

    previous_block_activation = x  # Запоминаем для skip-connection

    for size in [64, 128, 256]:
        # -- Первая часть блока --
        x = layers.Activation("relu")(x)
        # SeparableConv2D в 9 раз быстрее обычной Conv2D!
        x = layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)

        # -- Вторая часть блока --
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)

        # -- Уменьшение размерности --
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # -- Skip Connection (добавляем то, что было до блока) --
        # Проецируем старый слой в новую размерность
        residual = layers.Conv2D(size, 1, strides=2, padding="same", use_bias=False)(previous_block_activation)
        x = layers.add([x, residual])  # Складываем (суть ResNet)
        previous_block_activation = x  # Обновляем для следующего шага

    # === 4. Финальная классификация ===
    x = layers.SeparableConv2D(512, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)  # Дропаут 40% от переобучения

    # Softmax для получения вероятностей классов
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="FastOcrNet_CPU")


def compile_model(
        model: keras.Model,
        learning_rate: float,
        label_smoothing: float,
) -> None:
    # Используем AdamW для лучшей сходимости (если версия Keras позволяет), иначе Adam
    try:
        optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
    except AttributeError:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # ИСПРАВЛЕНИЕ ОШИБКИ: Убрали label_smoothing для SparseCategoricalCrossentropy
    # Это чинит ошибку TypeError, которая у вас возникала
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
        # jit_compile=False безопаснее для CPU на Windows
        jit_compile=False
    )
