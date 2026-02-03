import argparse
import json
from pathlib import Path

import numpy as np
from tensorflow import keras

from ocr_app.dataset import (
    DatasetItem,
    build_dataset_index,
    compute_ahash,
    load_images,
    split_items,
    style_bucket_from_hash,
    write_dataset_index,
)
from ocr_app.model import build_model, save_labels


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_dataset_root = repo_root / "dataset_external"
    parser = argparse.ArgumentParser(description="Train OCR model on external datasets.")
    parser.add_argument(
        "--data-root",
        default=str(default_dataset_root),
        help="Root folder with datasets.",
    )
    parser.add_argument("--image-size", type=int, default=32, help="Square size for model inputs.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train/validation split ratio.")
    parser.add_argument(
        "--style-bucket-bits",
        type=int,
        default=8,
        help="How many high-order hash bits to use for style buckets.",
    )
    parser.add_argument(
        "--style-cache",
        default=None,
        help="Path to cache style buckets (defaults to data root).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Log image loading progress every N images (0 to disable).",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing factor.")
    parser.add_argument("--output-dir", default="model", help="Directory to save model artifacts.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise SystemExit(f"Dataset root not found: {data_root}")
    items = build_dataset_index(data_root)
    if not items:
        raise SystemExit("No dataset images found. Run download script or add datasets.")

    index_path = data_root / "dataset_index.json"
    write_dataset_index(items, index_path)

    labels = sorted({item.label for item in items})
    model = build_model(
        len(labels),
        (args.image_size, args.image_size, 1),
        learning_rate=1e-3,
        label_smoothing=args.label_smoothing,
    )
    model.summary()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def build_callbacks(stage: str) -> list[keras.callbacks.Callback]:
        checkpoint_path = output_dir / f"ocr_model_{stage}.keras"
        return [
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
                patience=6,
                min_lr=1e-6,
                verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=12,
                restore_best_weights=True,
                verbose=1,
            ),
        ]

    items_by_label: dict[str, list[DatasetItem]] = {}
    for item in items:
        items_by_label.setdefault(item.label, []).append(item)

    style_cache_path = Path(args.style_cache) if args.style_cache else data_root / "style_buckets.json"
    style_cache: dict[str, int] = {}
    if style_cache_path.exists():
        style_cache = json.loads(style_cache_path.read_text(encoding="utf-8"))
    missing_items = [item for item in items if str(item.path) not in style_cache]
    if missing_items:
        print(f"Computing style buckets for {len(missing_items)} images...")
        for index, item in enumerate(missing_items, start=1):
            hash_value = compute_ahash(item.path)
            bucket = style_bucket_from_hash(hash_value, bucket_bits=args.style_bucket_bits)
            style_cache[str(item.path)] = bucket
            if args.log_every and index % args.log_every == 0:
                print(f"Hashed {index}/{len(missing_items)} images...")
        style_cache_path.write_text(json.dumps(style_cache, ensure_ascii=False, indent=2), encoding="utf-8")

    train_items: list[DatasetItem] = []
    val_items: list[DatasetItem] = []
    for label, label_items in items_by_label.items():
        items_by_bucket: dict[int, list[DatasetItem]] = {}
        for item in label_items:
            bucket = style_cache[str(item.path)]
            items_by_bucket.setdefault(bucket, []).append(item)
        label_train = 0
        label_val = 0
        for bucket_items in items_by_bucket.values():
            train_split, val_split = split_items(bucket_items, train_ratio=args.train_ratio)
            train_items.extend(train_split)
            val_items.extend(val_split)
            label_train += len(train_split)
            label_val += len(val_split)
        print(f"Label {label}: {label_train} train / {label_val} val (style-stratified)")

    print("Loading training images...")
    x_train, y_train, _ = load_images(
        train_items,
        (args.image_size, args.image_size),
        labels=labels,
        log_every=args.log_every,
    )
    print("Loading validation images...")
    x_val, y_val, _ = load_images(
        val_items,
        (args.image_size, args.image_size),
        labels=labels,
        log_every=args.log_every,
    )

    counts = np.bincount(y_train, minlength=len(labels))
    total = counts.sum()
    class_weight = {
        index: (total / (len(labels) * count)) if count > 0 else 0.0
        for index, count in enumerate(counts)
    }

    stage1_epochs = max(1, int(args.epochs * 0.7))
    stage2_epochs = max(1, args.epochs - stage1_epochs)
    print(f"Stage 1 epochs: {stage1_epochs}, Stage 2 epochs: {stage2_epochs}")

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=stage1_epochs,
        batch_size=args.batch_size,
        callbacks=build_callbacks("stage1"),
        class_weight=class_weight,
        verbose=1,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(label_smoothing=args.label_smoothing),
        metrics=["accuracy"],
    )
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=stage1_epochs + stage2_epochs,
        initial_epoch=stage1_epochs,
        batch_size=args.batch_size,
        callbacks=build_callbacks("stage2"),
        class_weight=class_weight,
        verbose=1,
    )

    model.save(output_dir / "ocr_model.keras")
    save_labels(labels, output_dir / "labels.json")
    np.save(output_dir / "image_size.npy", np.array([args.image_size, args.image_size]))

    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
