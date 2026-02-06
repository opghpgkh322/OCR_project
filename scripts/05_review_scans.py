import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras

from ocr_app.config import SheetConfig
from ocr_app.labels import CHAR_TO_LABEL, DIGIT_LABELS, LABEL_TO_CHAR, LETTER_LABELS, choose_allowed_label
from ocr_app.preprocessing import align_image, load_image, preprocess_cell


def normalize_label(text: str) -> str | None:
    cleaned = text.strip()
    if not cleaned:
        return None
    if cleaned in LABEL_TO_CHAR:
        return cleaned
    if cleaned in DIGIT_LABELS:
        return cleaned
    if cleaned in CHAR_TO_LABEL:
        return CHAR_TO_LABEL[cleaned]
    return None


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Review OCR predictions and save corrections.")
    parser.add_argument("--scans", default=str(repo_root / "scans"), help="Folder with scanned forms.")
    parser.add_argument("--config", default=str(repo_root / "sheet_config.json"), help="Path to config JSON.")
    parser.add_argument("--model-dir", default=str(repo_root / "scripts" / "model"), help="Directory with trained model.")
    parser.add_argument("--review-root", default=str(repo_root / "dataset_review"), help="Folder to store corrected samples.")
    parser.add_argument("--log", default=str(repo_root / "review_log.csv"), help="CSV log for corrections.")
    args = parser.parse_args()

    config = SheetConfig.load(args.config)
    model_dir = Path(args.model_dir)
    model = keras.models.load_model(model_dir / "ocr_model.keras")
    labels = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
    image_size = np.load(model_dir / "image_size.npy")
    size = (int(image_size[0]), int(image_size[1]))

    review_root = Path(args.review_root)
    review_root.mkdir(parents=True, exist_ok=True)

    scan_paths = sorted(Path(args.scans).glob("*"))
    if not scan_paths:
        raise SystemExit("No scans found.")

    print("Review mode: press Enter to accept prediction, type correction, or ':q' to stop early.")

    stop_requested = False
    with open(args.log, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scan", "field", "index", "prediction", "corrected", "image_path"])
        for scan_path in scan_paths:
            if stop_requested:
                break
            image = load_image(str(scan_path))
            aligned = align_image(image, (config.image_width, config.image_height))
            for cell in config.cells:
                crop = aligned[cell.y : cell.y + cell.h, cell.x : cell.x + cell.w]
                processed = preprocess_cell(crop, size)
                batch = np.expand_dims(np.array([processed]), axis=-1)
                probabilities = model.predict(batch, verbose=0)[0]
                if cell.label in {"last_name", "first_name", "patronymic"}:
                    allowed = LETTER_LABELS
                elif cell.label in {"birth_date", "phone"}:
                    allowed = DIGIT_LABELS
                else:
                    allowed = set(labels)
                predicted = choose_allowed_label(probabilities, labels, allowed)
                predicted_char = LABEL_TO_CHAR.get(predicted, predicted)

                # Save review samples as black-ink-on-white to match dataset_external.
                preview = (255.0 - (processed * 255.0)).astype(np.uint8)
                output_dir = review_root / predicted
                output_dir.mkdir(parents=True, exist_ok=True)
                output_name = f"{scan_path.stem}_{cell.label}_{cell.index}.png"
                output_path = output_dir / output_name
                cv2.imwrite(str(output_path), preview)

                prompt = f"{scan_path.name} [{cell.label} #{cell.index}] -> {predicted_char}: "
                corrected_input = input(prompt).strip()
                if corrected_input.lower() in {":q", "q!", "quit", "exit"}:
                    stop_requested = True
                    print("Early stop requested. Saving collected corrections...")
                    break

                corrected_label = normalize_label(corrected_input) if corrected_input else predicted
                if corrected_label is None:
                    corrected_label = predicted

                corrected_dir = review_root / corrected_label
                corrected_dir.mkdir(parents=True, exist_ok=True)
                corrected_path = corrected_dir / output_name
                if corrected_path != output_path:
                    output_path.replace(corrected_path)
                    output_path = corrected_path

                writer.writerow([scan_path.name, cell.label, cell.index, predicted, corrected_label, str(output_path)])

    print(f"Review data saved to {review_root}")


if __name__ == "__main__":
    main()