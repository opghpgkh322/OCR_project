import json
import random
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
    log_every: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if labels is None:
        labels = sorted({item.label for item in items})
    label_to_index = {label: i for i, label in enumerate(labels)}
    features = []
    targets = []
    for index, item in enumerate(items, start=1):
        data = np.fromfile(str(item.path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            continue
        processed = preprocess_cell(image, image_size)
        features.append(processed)
        targets.append(label_to_index[item.label])
        if log_every and index % log_every == 0:
            print(f"Loaded {index}/{len(items)} images...")
    if not features:
        raise RuntimeError("No images loaded from dataset.")
    x = np.expand_dims(np.array(features), axis=-1)
    y = np.array(targets)
    return x, y, labels
