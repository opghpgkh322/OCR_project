import argparse
import csv
from pathlib import Path

import numpy as np
from tensorflow import keras

from ocr_app.config import SheetConfig
from ocr_app.model import load_labels, predict_labels
from ocr_app.preprocessing import align_image, load_image, preprocess_cell


def group_cells(cells):
    grouped = {}
    for cell in cells:
        grouped.setdefault(cell.label, []).append(cell)
    for label, items in grouped.items():
        grouped[label] = sorted(items, key=lambda item: item.index)
    return grouped


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run OCR on scans and export CSV.")
    parser.add_argument(
        "--scans",
        default=str(repo_root / "scans"),
        help="Folder with scanned forms.",
    )
    parser.add_argument(
        "--config",
        default=str(repo_root / "sheet_config.json"),
        help="Path to config JSON.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(repo_root / "scripts" / "model"),
        help="Directory with trained model.",
    )
    parser.add_argument(
        "--output",
        default=str(repo_root / "output.csv"),
        help="Output CSV file.",
    )
    args = parser.parse_args()

    config = SheetConfig.load(args.config)
    grouped = group_cells(config.cells)

    model_dir = Path(args.model_dir)
    model = keras.models.load_model(model_dir / "ocr_model.keras")
    labels = load_labels(model_dir / "labels.json")
    image_size = np.load(model_dir / "image_size.npy")
    size = (int(image_size[0]), int(image_size[1]))

    scan_paths = sorted(Path(args.scans).glob("*"))
    if not scan_paths:
        raise SystemExit("No scans found.")

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "last_name", "first_name", "patronymic", "birth_date", "phone"],
        )
        writer.writeheader()
        for scan_path in scan_paths:
            image = load_image(str(scan_path))
            aligned = align_image(image, (config.image_width, config.image_height))

            row = {
                "filename": scan_path.name,
                "last_name": "",
                "first_name": "",
                "patronymic": "",
                "birth_date": "",
                "phone": "",
            }

            for label, cells in grouped.items():
                crops = []
                for cell in cells:
                    crop = aligned[cell.y : cell.y + cell.h, cell.x : cell.x + cell.w]
                    processed = preprocess_cell(crop, size)
                    crops.append(processed)
                batch = np.expand_dims(np.array(crops), axis=-1)
                predictions = predict_labels(model, labels, batch)
                row[label] = "".join(predictions)

            writer.writerow(row)

    print(f"Saved CSV to {args.output}")


if __name__ == "__main__":
    main()
