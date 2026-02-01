import argparse
from pathlib import Path

import numpy as np
from tensorflow import keras

from ocr_app.dataset import build_dataset_index, load_images, split_items, write_dataset_index
from ocr_app.model import build_model, save_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OCR model on external datasets.")
    parser.add_argument("--data-root", default="external_database", help="Root folder with datasets.")
    parser.add_argument("--image-size", type=int, default=32, help="Square size for model inputs.")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs.")
    parser.add_argument("--output-dir", default="model", help="Directory to save model artifacts.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    items = build_dataset_index(data_root)
    if not items:
        raise SystemExit("No dataset images found. Run download script or add datasets.")

    index_path = data_root / "dataset_index.json"
    write_dataset_index(items, index_path)

    train_items, val_items = split_items(items)
    x_train, y_train, labels = load_images(train_items, (args.image_size, args.image_size))
    x_val, y_val, _ = load_images(val_items, (args.image_size, args.image_size))

    model = build_model(len(labels), (args.image_size, args.image_size, 1))
    model.summary()
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=args.epochs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "ocr_model.keras")
    save_labels(labels, output_dir / "labels.json")
    np.save(output_dir / "image_size.npy", np.array([args.image_size, args.image_size]))

    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
