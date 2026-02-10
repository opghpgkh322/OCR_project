from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ocr_app.preprocessing import preprocess_cell


@dataclass(frozen=True)
class DatasetItem:
    path: Path
    label: str


def scan_dataset(root: Path, allowed_ext: tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    # Используем rglob для рекурсивного поиска, если датасет вложенный
    for path in root.rglob("*"):
        if path.suffix.lower() not in allowed_ext:
            continue
        # Пропускаем системные файлы
        if path.name.startswith("."):
            continue
        # Метка - это имя папки, в которой лежит файл
        items.append(DatasetItem(path=path, label=path.parent.name))
    return items


def scan_datasets(roots: list[Path], allowed_ext: tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    for root in roots:
        if not root.exists():
            continue
        items.extend(scan_dataset(root, allowed_ext=allowed_ext))
    return items


# --- НОВАЯ ФУНКЦИЯ: АДАПТАЦИЯ СТИЛЯ ---
def augment_style(image: np.ndarray) -> np.ndarray:
    """Случайно меняет толщину линий, чтобы уравнять Paint и ручку."""

    # Шанс 50% вообще ничего не делать
    if random.random() > 0.5:
        return image

    kernel = np.ones((2, 2), np.uint8)

    # Шанс 25%: сделать линии тоньше (Erosion) - превращает Paint в ручку
    if random.random() < 0.5:
        # Для темного текста на светлом фоне эрозия "съедает" черное
        # Но в OpenCV эрозия работает по белому.
        # Если у нас черный текст (0) на белом (255), нам нужна DILATION, чтобы расширить белое (сузить черное)
        return cv2.dilate(image, kernel, iterations=1)

    # Шанс 25%: сделать линии толще (Dilation) - превращает ручку в Paint
    else:
        # EROSION сужает белое -> расширяет черное
        return cv2.erode(image, kernel, iterations=1)


def load_images(
        items: list[DatasetItem],
        image_size: tuple[int, int],
        labels: list[str],
        log_every: int = 0,
        augment: bool = False  # Флаг для включения аугментации
) -> tuple[np.ndarray, np.ndarray]:
    label_to_index = {label: i for i, label in enumerate(labels)}
    features: list[np.ndarray] = []
    targets: list[int] = []
    total = len(items)

    print(f"Loading {total} images (Augmentation={'ON' if augment else 'OFF'})...")

    for index, item in enumerate(items, start=1):
        try:
            # Читаем через numpy для поддержки кириллицы в путях на Windows
            file_bytes = np.fromfile(str(item.path), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)  # Сразу в ЧБ

            if image is None:
                continue

            # Применяем стилевую аугментацию ДО ресайза, пока есть детали
            if augment:
                image = augment_style(image)

            # Обычный препроцессинг (ресайз, центрирование)
            # preprocess_cell ожидает, что image может быть любым, но вернет float 0..1
            # Важно: передаем ЧБ image, preprocess_cell должен это уметь
            processed = preprocess_cell(image, image_size)

            features.append(processed)
            targets.append(label_to_index[item.label])

        except Exception as e:
            print(f"Error loading {item.path}: {e}")
            continue

        if log_every and index % log_every == 0:
            print(f"Loaded {index}/{total} images...")

    # Добавляем измерение канала: (N, 64, 64) -> (N, 64, 64, 1)
    x = np.expand_dims(np.array(features), axis=-1)
    y = np.array(targets)
    return x, y


# --- ФУНКЦИИ ДЛЯ STYLE CLUSTERING (Можно оставить как заглушки, если не используем) ---
# Если вы используете trainer.py, он их вызовет. Оставим их рабочими.

def extract_style_features(image: np.ndarray) -> np.ndarray:
    # Упрощенная заглушка, чтобы не ломать старый пайплайн
    # Если мы не используем кластеризацию, это не важно.
    return np.zeros(6, dtype=np.float32)


def compute_style_matrix(items: list[DatasetItem], *args, **kwargs) -> np.ndarray:
    # Возвращаем пустышку
    return np.zeros((len(items), 6), dtype=np.float32)


def kmeans_cluster(features: np.ndarray, k: int = 3, **kwargs) -> np.ndarray:
    # Возвращаем случайные кластеры, так как мы теперь полагаемся на аугментацию,
    # а не на разделение по стилям.
    return np.random.randint(0, k, size=len(features))


def stratified_split(
        items: list[DatasetItem],
        style_groups: np.ndarray,
        train_ratio: float,
        seed: int,
) -> tuple[list[DatasetItem], list[DatasetItem]]:
    # Простое случайное разбиение без учета стилей
    rng = random.Random(seed)
    items_copy = list(items)
    rng.shuffle(items_copy)

    split_idx = int(len(items_copy) * train_ratio)
    return items_copy[:split_idx], items_copy[split_idx:]
