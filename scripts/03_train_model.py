import argparse
from pathlib import Path

import numpy as np
from tensorflow import keras

from ocr_app.dataset import build_dataset_index, group_items_by_style, load_images, split_items, write_dataset_index
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
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train/validation split ratio.")
    parser.add_argument(
        "--split-by-style",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train sequentially per filename style group.",
    )
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
    model = build_model(len(labels), (args.image_size, args.image_size, 1))
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

    if args.split_by_style:
        style_groups = group_items_by_style(items)
    else:
        style_groups = {"all": items}

    sorted_groups = sorted(style_groups.items(), key=lambda item: len(item[1]), reverse=True)
    for index, (style_key, style_items) in enumerate(sorted_groups, start=1):
        stage_name = f"stage_{index}_{style_key}" if len(sorted_groups) > 1 else "all"
        print(f"Training stage {index}/{len(sorted_groups)}: {style_key} ({len(style_items)} samples)")
        train_items, val_items = split_items(style_items, train_ratio=args.train_ratio)
        x_train, y_train, _ = load_images(train_items, (args.image_size, args.image_size), labels=labels)
        x_val, y_val, _ = load_images(val_items, (args.image_size, args.image_size), labels=labels)
        model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=build_callbacks(stage_name),
            verbose=1,
        )

    model.save(output_dir / "ocr_model.keras")
    save_labels(labels, output_dir / "labels.json")
    np.save(output_dir / "image_size.npy", np.array([args.image_size, args.image_size]))

    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
