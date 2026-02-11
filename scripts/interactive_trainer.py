import argparse
from pathlib import Path
import uuid
import cv2
import numpy as np
from tensorflow import keras

from ocr_app.config import SheetConfig
from ocr_app.labels import DIGIT_LABELS, LABEL_TO_CHAR, LETTER_LABELS, choose_allowed_label
from ocr_app.model import load_labels
from ocr_app.preprocessing import align_image, load_image, preprocess_cell

# Обратный маппинг для удобства (а -> A_cyr)
CHAR_TO_LABEL_LOWER = {char.lower(): label for label, char in LABEL_TO_CHAR.items() if char}


def group_cells(cells):
    grouped = {}
    for cell in cells:
        grouped.setdefault(cell.label, []).append(cell)
    for label, items in grouped.items():
        grouped[label] = sorted(items, key=lambda item: item.index)
    return grouped


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Interactive trainer for full fields.")
    parser.add_argument("--scans", default=str(repo_root / "scans"))
    parser.add_argument("--config", default=str(repo_root / "sheet_config.json"))
    parser.add_argument("--model-dir", default=str(repo_root / "scripts" / "model"))
    parser.add_argument("--dataset", default=str(repo_root / "dataset_review"))
    # Возвращаем дефолтный паддинг 15, как в align_form
    parser.add_argument("--padding", type=int, default=15, help="Top padding used in alignment")
    args = parser.parse_args()

    config = SheetConfig.load(args.config)
    grouped = group_cells(config.cells)

    # Порядок полей: сначала ФИО, потом остальное
    field_order = ["last_name", "first_name", "patronymic", "birth_date", "phone"]

    model_dir = Path(args.model_dir)
    model = keras.models.load_model(model_dir / "ocr_model.keras")
    labels = load_labels(model_dir / "labels.json")
    image_size = np.load(model_dir / "image_size.npy")
    size = (int(image_size[0]), int(image_size[1]))

    dataset_root = Path(args.dataset)
    dataset_root.mkdir(parents=True, exist_ok=True)

    scan_paths = sorted(Path(args.scans).glob("*"))
    if not scan_paths:
        raise SystemExit("No scans found.")

    cv2.namedWindow("Field Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Field Preview", 800, 200)

    for scan_path in scan_paths:
        print(f"\n📄 Обработка файла: {scan_path.name}")
        image = load_image(str(scan_path))

        try:
            # Используем ту же логику выравнивания, что и в старых скриптах
            # Важно: здесь мы НЕ вычитаем padding из высоты, если в конфиге записана полная высота.
            # Но чтобы не ломать логику, используем align_image как в генераторе CSV
            target_height = config.image_height - args.padding
            aligned = align_image(
                image,
                output_size=(config.image_width, target_height),
                top_padding=args.padding
            )
        except Exception as e:
            print(f"❌ Ошибка выравнивания {scan_path.name}: {e}")
            continue

        for field_name in field_order:
            if field_name not in grouped:
                continue

            cells = grouped[field_name]
            if not cells:
                continue

            # 1. Сбор данных
            crops = []
            crop_images_to_save = []  # Сырые кропы для сохранения (как было раньше)

            # Определяем границы поля для превью
            min_x = min(c.x for c in cells)
            min_y = min(c.y for c in cells)
            max_x = max(c.x + c.w for c in cells)
            max_y = max(c.y + c.h for c in cells)

            # Делаем превью с запасом
            margin = 10
            field_preview = aligned[
                            max(0, min_y - margin):min(aligned.shape[0], max_y + margin),
                            max(0, min_x - margin):min(aligned.shape[1], max_x + margin)
                            ]

            for cell in cells:
                # Строго по координатам из конфига, без лишних паддингов
                crop = aligned[cell.y: cell.y + cell.h, cell.x: cell.x + cell.w]

                # Для предсказания - препроцессинг
                processed = preprocess_cell(crop, size)
                crops.append(processed)

                # Для сохранения - ОРИГИНАЛЬНЫЙ кроп (но в ч/б, если надо)
                # Если preprocess_cell делает инверсию, то лучше сохранять результат препроцессинга,
                # но вы просили "как было". В старой версии сохранялся crop (возможно, прошедший threshold).
                # Чтобы не было рамок, используем crop как есть.
                # Но если мы хотим, чтобы в датасете были готовые к обучению картинки,
                # лучше сохранить processed, конвертированный в uint8.

                # ВАРИАНТ "КАК БЫЛО": Сохраняем processed, но без агрессивной очистки границ,
                # полагаясь на то, что координаты в конфиге точные.
                to_save = (processed * 255).astype(np.uint8)
                crop_images_to_save.append(to_save)

            if not crops:
                continue

            # 2. Распознавание
            batch = np.expand_dims(np.array(crops), axis=-1)
            probabilities = model.predict(batch, verbose=0)

            if field_name in {"last_name", "first_name", "patronymic"}:
                allowed = LETTER_LABELS
            else:
                allowed = DIGIT_LABELS

            predicted_chars = []

            for idx in range(len(crops)):
                pred_label = choose_allowed_label(probabilities[idx], labels, allowed)
                char = LABEL_TO_CHAR.get(pred_label, "")
                if pred_label == "Empty":
                    char = "_"
                predicted_chars.append(char)

            predicted_text = "".join(predicted_chars).replace("_", "")

            # 3. Ввод пользователя
            cv2.imshow("Field Preview", field_preview)
            cv2.waitKey(100)

            print(f"Поле [{field_name}]. Распознано: {predicted_text}")
            user_input = input(f"Верный текст (Enter='{predicted_text}', 'skip'=дальше): ").strip()

            if user_input.lower() == 'skip':
                continue

            final_text = predicted_text if user_input == "" else user_input.upper().replace(" ", "")

            # 4. Сохранение ошибок
            # Сопоставляем буквы слева направо.
            # Если пользователь ввел "ИВАНОВ", берем 1-ю ячейку -> И, 2-ю -> В...

            count_saved = 0
            for i, correct_char in enumerate(final_text):
                if i >= len(crop_images_to_save):
                    break  # Ячеек меньше, чем букв

                char_lower = correct_char.lower()

                # Определяем папку назначения
                if char_lower in CHAR_TO_LABEL_LOWER:
                    label_dir_name = CHAR_TO_LABEL_LOWER[char_lower]
                elif char_lower.isdigit():
                    label_dir_name = char_lower
                else:
                    print(f"⚠️ Пропуск символа: {correct_char}")
                    continue

                # Берем картинку i-й ячейки
                img_to_save = crop_images_to_save[i]

                # Сохраняем
                target_dir = dataset_root / label_dir_name
                target_dir.mkdir(exist_ok=True)

                fname = f"{scan_path.stem}_{field_name}_{i}_{uuid.uuid4().hex[:6]}.jpg"

                # imencode для кириллицы
                is_success, buf = cv2.imencode(".jpg", img_to_save)
                if is_success:
                    buf.tofile(str(target_dir / fname))
                    count_saved += 1

            if count_saved > 0:
                print(f"✅ Сохранено {count_saved} примеров")

    cv2.destroyAllWindows()
    print("Готово.")


if __name__ == "__main__":
    main()
