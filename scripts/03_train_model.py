import argparse
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras

from ocr_app.dataset import build_dataset_index, load_images, split_items, write_dataset_index
from ocr_app.model import build_model, save_labels


def _prompt_int(label: str, default: int) -> int:
    raw = input(f"{label} [{default}]: ").strip()
    if not raw:
        return default
    return int(raw)


def _prompt_str(label: str, default: str) -> str:
    raw = input(f"{label} [{default}]: ").strip()
    return raw or default


def _prompt_bool(label: str, default: bool = True) -> bool:
    default_text = "y" if default else "n"
    raw = input(f"{label} (y/n) [{default_text}]: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes", "1"}


def ensure_mnist_digits(root: Path, max_per_digit: int) -> None:
    target_root = root / "mnist_digits"
    if target_root.exists():
        return
    print("Preparing MNIST digits dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    counts = {digit: 0 for digit in range(10)}
    for image, label in zip(x, y):
        if counts[label] >= max_per_digit:
            continue
        label_dir = target_root / str(label)
        label_dir.mkdir(parents=True, exist_ok=True)
        filename = label_dir / f"{counts[label]:05d}.png"
        cv2.imwrite(str(filename), image)
        counts[label] += 1
        if all(count >= max_per_digit for count in counts.values()):
            break
    print(f"MNIST digits prepared in {target_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OCR model on external datasets.")
    parser.add_argument("--data-root", default="dataset_external", help="Root folder with datasets.")
    parser.add_argument("--image-size", type=int, default=32, help="Square size for model inputs.")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs.")
    parser.add_argument("--output-dir", default="model", help="Directory to save model artifacts.")
    parser.add_argument("--include-mnist", action="store_true", help="Include MNIST digits dataset.")
    parser.add_argument("--mnist-max-per-digit", type=int, default=2000, help="MNIST samples per digit.")
    args = parser.parse_args()

    data_root = Path(_prompt_str("Dataset root folder", args.data_root))
    image_size = _prompt_int("Image size (square)", args.image_size)
    epochs = _prompt_int("Epochs", args.epochs)
    output_dir = Path(_prompt_str("Output model dir", args.output_dir))
    include_mnist = args.include_mnist or _prompt_bool("Include MNIST digits dataset", default=True)
    mnist_limit = _prompt_int("MNIST samples per digit", args.mnist_max_per_digit)

    if not data_root.exists():
        raise SystemExit(f"Dataset root not found: {data_root}")

    if include_mnist:
        ensure_mnist_digits(data_root, mnist_limit)
    items = build_dataset_index(data_root)
    if not items:
        raise SystemExit("No dataset images found. Run download script or add datasets.")

    index_path = data_root / "dataset_index.json"
    write_dataset_index(items, index_path)

    train_items, val_items = split_items(items)
    x_train, y_train, labels = load_images(train_items, (image_size, image_size))
    x_val, y_val, _ = load_images(val_items, (image_size, image_size))

    model = build_model(len(labels), (image_size, image_size, 1))
    model.summary()
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "ocr_model.keras")
    save_labels(labels, output_dir / "labels.json")
    np.save(output_dir / "image_size.npy", np.array([image_size, image_size]))

    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
