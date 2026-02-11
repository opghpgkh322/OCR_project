import argparse
import csv
from pathlib import Path
import cv2
import numpy as np
from tensorflow import keras

from ocr_app.config import SheetConfig
from ocr_app.labels import DIGIT_LABELS, LABEL_TO_CHAR, LETTER_LABELS, choose_allowed_label
from ocr_app.model import load_labels
from ocr_app.preprocessing import align_image, load_image, preprocess_cell


def group_cells(cells):
    grouped = {}
    for cell in cells:
        grouped.setdefault(cell.label, []).append(cell)
    for label, items in grouped.items():
        grouped[label] = sorted(items, key=lambda item: item.index)
    return grouped


def is_empty_crop(image: np.ndarray, threshold: float = 0.01) -> bool:
    """
    Проверяет, пустой ли кроп.
    image: изображение 0..1, где чем выше значение, тем "белее" буква.
    Если среднее значение очень низкое, значит там почти сплошная чернота (фон).
    """
    return np.mean(image) < threshold


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run OCR on scans and export CSV.")
    parser.add_argument("--scans", default=str(repo_root / "scans"))
    parser.add_argument("--config", default=str(repo_root / "sheet_config.json"))
    parser.add_argument("--model-dir", default=str(repo_root / "scripts" / "model"))
    parser.add_argument("--output", default=str(repo_root / "output.csv"))
    # Добавляем параметр для паддинга, чтобы он совпадал с тем, что был при создании aligned_form
    parser.add_argument("--padding", type=int, default=15, help="Top padding used in alignment")
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
            print(f"Processing {scan_path.name}...")
            # Читаем сразу в цвете, align_image сам разберется, но preprocess_cell умеет работать и так
            image = load_image(str(scan_path))

            try:
                # --- ИСПРАВЛЕНИЕ КООРДИНАТ ---
                # Функция align_image сама добавляет args.padding к высоте.
                # Но config.image_height УЖЕ включает этот паддинг (так как конфиг делался по aligned_form).
                # Поэтому мы должны ВЫЧЕСТЬ его перед передачей, чтобы не получить двойной отступ.
                target_height = config.image_height - args.padding

                aligned = align_image(
                    image,
                    output_size=(config.image_width, target_height),
                    top_padding=args.padding
                )
            except Exception as e:
                print(f"Skipping {scan_path.name}: alignment failed ({e})")
                continue

            row = {
                "filename": scan_path.name,
                "last_name": "", "first_name": "", "patronymic": "",
                "birth_date": "", "phone": "",
            }

            for label_name, cells in grouped.items():
                crops = []
                # Уменьшаем паддинг при нарезке, чтобы не цеплять соседей
                crop_padding = 2

                for cell in cells:
                    y1 = max(0, cell.y - crop_padding)
                    y2 = min(aligned.shape[0], cell.y + cell.h + crop_padding)
                    x1 = max(0, cell.x - crop_padding)
                    x2 = min(aligned.shape[1], cell.x + cell.w + crop_padding)

                    crop = aligned[y1:y2, x1:x2]

                    # Препроцессинг (он же инвертирует цвета, если вы применили мои прошлые правки)
                    processed = preprocess_cell(crop, size)

                    # --- ФИЛЬТРАЦИЯ МУСОРА ---
                    if is_empty_crop(processed, threshold=0.015):
                        continue

                    crops.append(processed)

                if not crops:
                    continue

                batch = np.expand_dims(np.array(crops), axis=-1)
                probabilities = model.predict(batch, verbose=0)

                if label_name in {"last_name", "first_name", "patronymic"}:
                    allowed = LETTER_LABELS
                elif label_name in {"birth_date", "phone"}:
                    allowed = DIGIT_LABELS
                else:
                    allowed = set(labels)

                predictions = []
                for idx in range(len(crops)):
                    pred_label = choose_allowed_label(probabilities[idx], labels, allowed)

                    # Пропускаем явный класс Empty
                    if pred_label == "Empty":
                        continue

                    char = LABEL_TO_CHAR.get(pred_label, "")
                    predictions.append(char)

                row[label_name] = "".join(predictions)

            writer.writerow(row)

    print(f"Saved CSV to {args.output}")


if __name__ == "__main__":
    main()
