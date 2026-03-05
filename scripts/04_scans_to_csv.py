import argparse
import csv
import json
import difflib
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
from tensorflow import keras

from ocr_app.config import SheetConfig
from ocr_app.labels import DIGIT_LABELS, LABEL_TO_CHAR, LETTER_LABELS, choose_allowed_label
from ocr_app.model import load_labels
from ocr_app.preprocessing import align_image, load_image, preprocess_cell


class GenderedDict:
    """
    Хранит словари частот, разделенные по полу.
    Вместо set() используем Counter() для подсчета популярности.
    """

    def __init__(self):
        self.all = Counter()  # Все слова: Word -> Count
        self.male = Counter()  # Мужские
        self.female = Counter()  # Женские
        self.map = {}  # Слово -> Пол

    def add(self, word: str, gender: str = None):
        if not word: return
        w = word.strip().upper()
        if len(w) < 2: return

        # Увеличиваем счетчик популярности
        self.all[w] += 1

        # Логика определения пола
        if w not in self.map:
            self.map[w] = gender
        elif self.map[w] is not None and self.map[w] != gender:
            self.map[w] = None

        if gender == 'm':
            self.male[w] += 1
        elif gender == 'f':
            self.female[w] += 1


def load_kaggle_dataset(path: Path, names_db: GenderedDict, surnames_db: GenderedDict, midnames_db: GenderedDict):
    if not path.exists():
        print(f"⚠️ Kaggle датасет не найден: {path}")
        return

    print(f"⏳ Чтение базы и подсчет частот: {path.name}...")
    count = 0
    try:
        try:
            f = open(path, "r", encoding="utf-8")
        except UnicodeDecodeError:
            f = open(path, "r", encoding="cp1251")

        with f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 4: continue
                fam, im, otch, g_raw = row[0], row[1], row[2], row[3]

                g_raw = g_raw.strip().upper()
                gender = None
                if g_raw in ('M', 'М'):
                    gender = 'm'
                elif g_raw in ('F', 'Ж'):
                    gender = 'f'

                surnames_db.add(fam, gender)
                names_db.add(im, gender)
                midnames_db.add(otch, gender)

                count += 1
                if count % 500000 == 0:
                    print(f"   ... обработано {count} строк")

    except Exception as e:
        print(f"❌ Ошибка чтения CSV: {e}")
    print(f"✓ Загружено {count} записей")


def load_jsonl_dataset(path: Path) -> GenderedDict:
    """Для JSONL считаем частоту = 1 (так как там просто список уникальных)"""
    db = GenderedDict()
    if not path.exists(): return db
    print(f"⏳ Подгрузка JSONL: {path.name}...")
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    word = data.get("text") or data.get("name") or data.get("surname") or data.get("midname")
                    gender = data.get("gender")
                    if word:
                        db.add(word, gender)
                except:
                    continue
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    return db


MIN_FREQUENCY = 3  # Минимальная частота слова в базе, чтобы считаться "надежным"


def correct_text(text: str, db: GenderedDict, target_gender: str = None) -> str:
    """
    Автокоррекция:
    1. Строгая длина.
    2. Фильтрация редких слов (опечаток в базе).
    3. Максимум совпадений (Hamming score).
    4. При равенстве -> самое популярное.
    """
    if not text: return text
    text = text.upper()
    target_len = len(text)

    # 1. Выбираем словарь частот
    if target_gender == 'm':
        candidates_counter = db.male
    elif target_gender == 'f':
        candidates_counter = db.female
    else:
        candidates_counter = db.all

    # 2. Фильтруем кандидатов по длине
    strict_candidates = [w for w in candidates_counter.keys() if len(w) == target_len]

    if not strict_candidates:
        return text

    # Если исходное слово есть в базе и оно достаточно популярное — оставляем
    if text in strict_candidates and candidates_counter[text] >= MIN_FREQUENCY:
        return text

    # 3. Разделяем кандидатов на "надежных" и "редких"
    trusted_candidates = [w for w in strict_candidates if candidates_counter[w] >= MIN_FREQUENCY]

    # Если есть надежные кандидаты, ищем ТОЛЬКО среди них
    # Это отсечет опечатки базы вида "ИВАНЕВ" (freq=1)
    search_pool = trusted_candidates if trusted_candidates else strict_candidates

    # 4. Поиск лучшего совпадения (Hamming)
    best_score = -1
    best_candidates = []

    for candidate in search_pool:
        score = sum(1 for i in range(target_len) if text[i] == candidate[i])

        if score > best_score:
            best_score = score
            best_candidates = [candidate]
        elif score == best_score:
            best_candidates.append(candidate)

    # Порог совпадения: если совпало меньше половины букв, может не стоит менять?
    # Но пока оставим как есть (берем максимум)

    if best_candidates and best_score > 0:
        # 5. Выбираем победителя по частоте
        winner = sorted(best_candidates, key=lambda w: candidates_counter[w], reverse=True)[0]

        if winner != text:
            freq = candidates_counter[winner]
            # Логируем, было ли это "надежное" исправление
            quality = "HighConf" if freq >= MIN_FREQUENCY else "LowConf"
            print(f"   🔧 {text} -> {winner} (Match: {best_score}/{target_len}, Freq: {freq}, {quality})")

        return winner

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

    surnames_db = GenderedDict()
    names_db = GenderedDict()
    midnames_db = GenderedDict()

    if not args.no_correct:
        print("=== ЗАГРУЗКА БАЗ С ЧАСТОТАМИ ===")
        # JSONL (частота = 1, так как данных о популярности там нет)
        s_old = load_jsonl_dataset(dict_dir / "surnames_table.jsonl")
        n_old = load_jsonl_dataset(dict_dir / "names_table.jsonl")
        m_old = load_jsonl_dataset(dict_dir / "midnames_table.jsonl")

        # Объединяем (сложение Counter'ов работает корректно)
        surnames_db.all.update(s_old.all);
        surnames_db.male.update(s_old.male);
        surnames_db.female.update(s_old.female);
        surnames_db.map.update(s_old.map)
        names_db.all.update(n_old.all);
        names_db.male.update(n_old.male);
        names_db.female.update(n_old.female);
        names_db.map.update(n_old.map)
        midnames_db.all.update(m_old.all);
        midnames_db.male.update(m_old.male);
        midnames_db.female.update(m_old.female);
        midnames_db.map.update(m_old.map)

        # CSV (тут реальные частоты)
        load_kaggle_dataset(dict_dir / "data.csv", names_db, surnames_db, midnames_db)

        print(f"ИТОГО уникальных: Имен {len(names_db.all)}, Фам {len(surnames_db.all)}, Отч {len(midnames_db.all)}")
        print("=================================")

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
            image = load_image(str(scan_path))
            try:
                target_height = config.image_height - args.padding
                aligned = align_image(image, output_size=(config.image_width, target_height), top_padding=args.padding)
            except Exception as e:
                print(f"Skipping {scan_path.name}: {e}")
                continue

            row_data = {"filename": scan_path.name, "last_name": "", "first_name": "", "patronymic": "",
                        "birth_date": "", "phone": ""}

            for label_name, cells in grouped.items():
                crops = []
                for cell in cells:
                    crop = aligned[max(0, cell.y - 2):min(aligned.shape[0], cell.y + cell.h + 2),
                           max(0, cell.x - 2):min(aligned.shape[1], cell.x + cell.w + 2)]
                    processed = preprocess_cell(crop, size)
                    if not is_empty_crop(processed):
                        crops.append(processed)

                if not crops: continue
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
                    if pred_label != "Empty":
                        predictions.append(LABEL_TO_CHAR.get(pred_label, ""))

                row_data[label_name] = "".join(predictions)

            final_row = row_data.copy()
            if not args.no_correct:
                detected_gender = None
                raw_name = row_data["first_name"]
                if raw_name:
                    corr_name = correct_text(raw_name, names_db)
                    final_row["first_name"] = corr_name
                    detected_gender = names_db.map.get(corr_name)
                    if detected_gender: print(f"   Пол: {detected_gender}")

                if row_data["last_name"]:
                    final_row["last_name"] = correct_text(row_data["last_name"], surnames_db, detected_gender)
                if row_data["patronymic"]:
                    final_row["patronymic"] = correct_text(row_data["patronymic"], midnames_db, detected_gender)

            writer.writerow(final_row)
    print(f"Saved CSV to {args.output}")


if __name__ == "__main__":
    main()
