import argparse
import csv
import json
import difflib
from pathlib import Path
import cv2
import numpy as np
from tensorflow import keras

from ocr_app.config import SheetConfig
from ocr_app.labels import DIGIT_LABELS, LABEL_TO_CHAR, LETTER_LABELS, choose_allowed_label
from ocr_app.model import load_labels
from ocr_app.preprocessing import align_image, load_image, preprocess_cell

# --- КОНФИГУРАЦИЯ ---
CORRECTION_THRESHOLD = 0.75  # Чуть снизим порог, так как фильтр по полу убирает много ложных вариантов


class GenderedDict:
    """Хранит словари, разделенные по полу."""

    def __init__(self):
        self.all = set()  # Все слова (для первого поиска)
        self.male = set()  # Только мужские
        self.female = set()  # Только женские
        self.map = {}  # Слово -> Пол ('m', 'f', или None)

    def add(self, word: str, gender: str = None):
        if not word: return
        w = word.strip().upper()
        self.all.add(w)

        # Запоминаем пол. Если слово уже есть, но с другим полом -> ставим None (унисекс)
        if w not in self.map:
            self.map[w] = gender
        elif self.map[w] != gender:
            self.map[w] = None  # Конфликт полов (например, САША м/ж)

        if gender == 'm':
            self.male.add(w)
        elif gender == 'f':
            self.female.add(w)


def load_jsonl_dataset(path: Path) -> GenderedDict:
    """Загружает JSONL в структуру с учетом пола."""
    db = GenderedDict()
    if not path.exists():
        print(f"⚠️ База не найдена: {path}")
        return db

    print(f"⏳ Загрузка словаря: {path.name}...")
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    word = data.get("text") or data.get("name") or data.get("surname") or data.get("midname")
                    gender = data.get("gender")  # Ожидаем 'm' или 'f'

                    if word:
                        db.add(word, gender)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"❌ Ошибка чтения {path.name}: {e}")

    print(f"✓ Загружено {len(db.all)} слов")
    return db


def correct_text(text: str, db: GenderedDict, target_gender: str = None) -> str:
    """
    Исправляет текст.
    Если target_gender задан ('m' или 'f'), ищет только в соответствующем подмножестве.
    """
    if not text:
        return text

    # 1. Выбираем пространство поиска
    if target_gender == 'm':
        search_space = list(db.male)  # difflib требует list
    elif target_gender == 'f':
        search_space = list(db.female)
    else:
        search_space = list(db.all)

    # Если словарь пуст (например, нет фильтрованных данных), ищем везде
    if not search_space:
        search_space = list(db.all)

    # 2. Если слово уже есть в (правильном) словаре — не трогаем
    if text in search_space:
        return text  # Идеальное совпадение

    # 3. Ищем похожие
    matches = difflib.get_close_matches(text, search_space, n=1, cutoff=CORRECTION_THRESHOLD)
    if matches:
        suggestion = matches[0]
        print(
            f"🔧 Исправление ({'М' if target_gender == 'm' else 'Ж' if target_gender == 'f' else '?'}) : {text} -> {suggestion}")
        return suggestion

    return text


def group_cells(cells):
    grouped = {}
    for cell in cells:
        grouped.setdefault(cell.label, []).append(cell)
    for label, items in grouped.items():
        grouped[label] = sorted(items, key=lambda item: item.index)
    return grouped


def is_empty_crop(image: np.ndarray, threshold: float = 0.015) -> bool:
    return np.mean(image) < threshold


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dict_dir = repo_root / "dictionaries"

    parser = argparse.ArgumentParser(description="Run OCR on scans and export CSV.")
    parser.add_argument("--scans", default=str(repo_root / "scans"))
    parser.add_argument("--config", default=str(repo_root / "sheet_config.json"))
    parser.add_argument("--model-dir", default=str(repo_root / "scripts" / "model"))
    parser.add_argument("--output", default=str(repo_root / "output.csv"))
    parser.add_argument("--padding", type=int, default=15)
    parser.add_argument("--no-correct", action="store_true")
    args = parser.parse_args()

    # 1. Загрузка умных словарей
    surnames_db = GenderedDict()
    names_db = GenderedDict()
    midnames_db = GenderedDict()

    if not args.no_correct:
        print("--- Инициализация словарей ---")
        surnames_db = load_jsonl_dataset(dict_dir / "surnames_table.jsonl")
        names_db = load_jsonl_dataset(dict_dir / "names_table.jsonl")
        midnames_db = load_jsonl_dataset(dict_dir / "midnames_table.jsonl")
        print("------------------------------")

    config = SheetConfig.load(args.config)
    grouped = group_cells(config.cells)

    # Модель
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
            image = load_image(str(scan_path))

            try:
                target_height = config.image_height - args.padding
                aligned = align_image(
                    image,
                    output_size=(config.image_width, target_height),
                    top_padding=args.padding
                )
            except Exception as e:
                print(f"Skipping {scan_path.name}: alignment failed ({e})")
                continue

            # Сначала распознаем ВСЁ в сыром виде
            raw_data = {
                "filename": scan_path.name,
                "last_name": "", "first_name": "", "patronymic": "",
                "birth_date": "", "phone": "",
            }

            for label_name, cells in grouped.items():
                crops = []
                crop_padding = 2

                for cell in cells:
                    y1 = max(0, cell.y - crop_padding)
                    y2 = min(aligned.shape[0], cell.y + cell.h + crop_padding)
                    x1 = max(0, cell.x - crop_padding)
                    x2 = min(aligned.shape[1], cell.x + cell.w + crop_padding)

                    crop = aligned[y1:y2, x1:x2]
                    processed = preprocess_cell(crop, size)

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
                    if pred_label == "Empty": continue
                    char = LABEL_TO_CHAR.get(pred_label, "")
                    predictions.append(char)

                raw_data[label_name] = "".join(predictions)

            # --- ИНТЕЛЛЕКТУАЛЬНАЯ КОРРЕКЦИЯ ---
            final_row = raw_data.copy()

            if not args.no_correct:
                detected_gender = None

                # 1. Сначала исправляем ИМЯ (самый надежный индикатор пола)
                raw_name = raw_data["first_name"]
                if raw_name:
                    # Ищем без фильтра пола сначала
                    corrected_name = correct_text(raw_name, names_db)
                    final_row["first_name"] = corrected_name

                    # Определяем пол по ИСПРАВЛЕННОМУ имени
                    detected_gender = names_db.map.get(corrected_name)
                    if detected_gender:
                        print(f"   Пол определен по имени ({corrected_name}): {detected_gender}")
                    else:
                        print(f"   Пол не определен (имя {corrected_name} нет в базе или унисекс)")

                # 2. Исправляем Фамилию (с учетом пола)
                if raw_data["last_name"]:
                    final_row["last_name"] = correct_text(
                        raw_data["last_name"],
                        surnames_db,
                        target_gender=detected_gender
                    )

                # 3. Исправляем Отчество (с учетом пола)
                if raw_data["patronymic"]:
                    final_row["patronymic"] = correct_text(
                        raw_data["patronymic"],
                        midnames_db,
                        target_gender=detected_gender
                    )

            writer.writerow(final_row)

    print(f"Saved CSV to {args.output}")


if __name__ == "__main__":
    main()
