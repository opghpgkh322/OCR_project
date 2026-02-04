from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from tensorflow import keras

from .data import compute_style_matrix, kmeans_cluster, load_images, scan_dataset, stratified_split
from .model import build_cnn, compile_model


@dataclass
class TrainingConfig:
    data_root: Path
    image_size: int = 64
    epochs: int = 36
    batch_size: int = 64
    train_ratio: float = 0.9
    style_clusters: int = 3
    seed: int = 42
    label_smoothing: float = 0.03
    output_dir: Path = Path("model")
    log_every: int = 1000
    use_class_weight: bool = True


def train_model(config: TrainingConfig) -> None:
    items = scan_dataset(config.data_root)
    if not items:
        raise SystemExit("No dataset images found. Check dataset_external.")

    labels = sorted({item.label for item in items})
    print(f"Computing style features for {len(items)} images...")
    style_features = compute_style_matrix(
        items,
        (config.image_size, config.image_size),
        log_every=config.log_every,
    )
    style_groups = kmeans_cluster(style_features, k=config.style_clusters)
    train_items, val_items = stratified_split(items, style_groups, config.train_ratio, config.seed)

    print(f"Loading training images ({len(train_items)})...")
    x_train, y_train = load_images(
        train_items,
        (config.image_size, config.image_size),
        labels,
        log_every=config.log_every,
    )
    print(f"Loading validation images ({len(val_items)})...")
    x_val, y_val = load_images(
        val_items,
        (config.image_size, config.image_size),
        labels,
        log_every=config.log_every,
    )

    counts = np.bincount(y_train, minlength=len(labels))
    total = counts.sum()
    class_weight = None
    if config.use_class_weight:
        class_weight = {
            index: (total / (len(labels) * count)) if count > 0 else 0.0
            for index, count in enumerate(counts)
        }

    model = build_cnn((config.image_size, config.image_size, 1), len(labels))
    total_steps = int(np.ceil(len(x_train) / config.batch_size) * config.epochs)
    compile_model(
        model,
        learning_rate=8e-4,
        label_smoothing=config.label_smoothing,
        total_steps=total_steps,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "ocr_model.keras"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(checkpoint_path)
    (config.output_dir / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(config.output_dir / "image_size.npy", np.array([config.image_size, config.image_size]))
