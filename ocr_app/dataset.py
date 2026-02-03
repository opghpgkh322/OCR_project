import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .preprocessing import preprocess_cell


@dataclass
class DatasetItem:
    path: Path
    label: str


def load_dataset_index(index_path: Path) -> list[DatasetItem]:
    data = json.loads(index_path.read_text(encoding="utf-8"))
    items = [DatasetItem(path=Path(item["path"]), label=item["label"]) for item in data]
    return items


def build_dataset_index(root: Path, allowed_ext: tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    for path in root.rglob("*"):
        if path.suffix.lower() not in allowed_ext:
            continue
        label = path.parent.name
        items.append(DatasetItem(path=path, label=label))
    return items


def write_dataset_index(items: list[DatasetItem], index_path: Path) -> None:
    payload = [
        {
            "path": str(item.path),
            "label": item.label,
        }
        for item in items
    ]
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def split_items(items: list[DatasetItem], train_ratio: float = 0.9) -> tuple[list[DatasetItem], list[DatasetItem]]:
    shuffled = items[:]
    random.shuffle(shuffled)
    split = int(len(shuffled) * train_ratio)
    return shuffled[:split], shuffled[split:]


def load_images(
    items: list[DatasetItem],
    image_size: tuple[int, int],
    labels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if labels is None:
        labels = sorted({item.label for item in items})
    label_to_index = {label: i for i, label in enumerate(labels)}
    features = []
    targets = []
    for item in items:
        data = np.fromfile(str(item.path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            continue
        processed = preprocess_cell(image, image_size)
        features.append(processed)
        targets.append(label_to_index[item.label])
    if not features:
        raise RuntimeError("No images loaded from dataset.")
    x = np.expand_dims(np.array(features), axis=-1)
    y = np.array(targets)
    return x, y, labels


def infer_style_key(path: Path) -> str:
    stem = path.stem
    match = re.match(r"^(.*?)(?:[-_ ]?\d+)$", stem)
    key = match.group(1) if match else stem
    key = key.rstrip("-_ ").lower()
    return key or stem.lower()


def group_items_by_style(items: list[DatasetItem]) -> dict[str, list[DatasetItem]]:
    groups: dict[str, list[DatasetItem]] = {}
    for item in items:
        key = infer_style_key(item.path)
        groups.setdefault(key, []).append(item)
    return groups
