from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from tensorflow import keras
from .data import load_images, scan_datasets, stratified_split
from .model import build_cnn, compile_model


@dataclass
class TrainingConfig:
    data_root: Path
    review_root: Path
    image_size: int = 64
    epochs: int = 25
    batch_size: int = 32
    train_ratio: float = 0.9
    output_dir: Path = Path("model")
    is_fine_tuning: bool = False  # Новый флаг


def train_model(config: TrainingConfig) -> None:
    # 1. Собираем все данные (основные + исправленные вручную)
    roots = [config.data_root]
    if config.review_root.exists():
        roots.append(config.review_root)

    items = scan_datasets(roots)
    if not items:
        raise SystemExit("Ошибка: Датасет пуст!")

    labels = sorted({item.label for item in items})
    print(f"Всего изображений: {len(items)}. Классов: {len(labels)}")

    # 2. Простое случайное разбиение (без сложной кластеризации, раз стиль один)
    # Используем заглушку для style_groups, так как она нам не нужна
    dummy_groups = np.zeros(len(items))
    train_items, val_items = stratified_split(items, dummy_groups, config.train_ratio, seed=42)

    print(f"Загрузка: Train={len(train_items)}, Val={len(val_items)}")

    # Важно: отключаем augment=True, так как стиль у нас теперь один и строгий
    # Но если данных мало (<1000 на букву), лучше включить легкую аугментацию в load_images
    x_train, y_train = load_images(train_items, (config.image_size, config.image_size), labels)
    x_val, y_val = load_images(val_items, (config.image_size, config.image_size), labels)

    # 3. Работа с моделью
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "ocr_model.keras"

    if config.is_fine_tuning and checkpoint_path.exists():
        print(">>> ЗАГРУЗКА СУЩЕСТВУЮЩЕЙ МОДЕЛИ ДЛЯ ДООБУЧЕНИЯ...")
        model = keras.models.load_model(checkpoint_path)
        # При дообучении ставим очень маленький LR, чтобы аккуратно корректировать веса
        initial_lr = 1e-5
    else:
        print(">>> СОЗДАНИЕ НОВОЙ МОДЕЛИ...")
        # Используем быструю архитектуру из прошлого совета (MobileNetV3 или FastOcrNet_CPU)
        model = build_cnn((config.image_size, config.image_size, 1), len(labels))
        initial_lr = 1e-3

    # Компилируем заново, чтобы применить новый LR
    compile_model(model, learning_rate=initial_lr, label_smoothing=0.0)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,  # Если 5 эпох нет прогресса - стоп
            restore_best_weights=True,
            verbose=1
        )
    ]

    print("Начинаем обучение...")
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Сохраняем метаданные
    model.save(checkpoint_path)
    (config.output_dir / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(config.output_dir / "image_size.npy", np.array([config.image_size, config.image_size]))
    print("Обучение завершено!")
