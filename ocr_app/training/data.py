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
    for path in root.rglob("*"):
        if path.suffix.lower() not in allowed_ext:
            continue
        items.append(DatasetItem(path=path, label=path.parent.name))
    return items


def scan_datasets(roots: list[Path], allowed_ext: tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    for root in roots:
        if not root.exists():
            continue
        items.extend(scan_dataset(root, allowed_ext=allowed_ext))
    return items


def extract_style_features(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        scaled = image.astype(np.float32)
        if scaled.max() <= 1.0:
            scaled = scaled * 255.0
        scaled = np.clip(scaled, 0, 255).astype(np.uint8)
    else:
        scaled = image
    resized = cv2.resize(scaled, (32, 32), interpolation=cv2.INTER_AREA)
    if resized.dtype != np.uint8:
        resized = resized.astype(np.uint8)
    mean = float(resized.mean())
    std = float(resized.std())
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_ratio = float(thresh.mean() / 255.0)
    edges = cv2.Canny(resized, 50, 150)
    edge_ratio = float(edges.mean() / 255.0)
    vert_proj = resized.mean(axis=0)
    horiz_proj = resized.mean(axis=1)
    return np.array(
        [
            mean,
            std,
            ink_ratio,
            edge_ratio,
            float(vert_proj.std()),
            float(horiz_proj.std()),
        ],
        dtype=np.float32,
    )


def compute_style_matrix(
    items: list[DatasetItem],
    image_size: tuple[int, int],
    log_every: int = 0,
) -> np.ndarray:
    features: list[np.ndarray] = []
    total = len(items)
    for index, item in enumerate(items, start=1):
        data = np.fromfile(str(item.path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            features.append(np.zeros(6, dtype=np.float32))
            continue
        processed = preprocess_cell(image, image_size)
        features.append(extract_style_features(processed))
        if log_every and index % log_every == 0:
            print(f"Style features: {index}/{total} images processed...")
    return np.vstack(features)


def kmeans_cluster(features: np.ndarray, k: int = 3, iterations: int = 25) -> np.ndarray:
    if len(features) == 0:
        return np.array([], dtype=np.int64)
    rng = np.random.default_rng(42)
    indices = rng.choice(len(features), size=min(k, len(features)), replace=False)
    centroids = features[indices]
    for _ in range(iterations):
        distances = ((features[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        assignments = distances.argmin(axis=1)
        new_centroids = []
        for i in range(len(centroids)):
            cluster_points = features[assignments == i]
            if len(cluster_points) == 0:
                new_centroids.append(centroids[i])
            else:
                new_centroids.append(cluster_points.mean(axis=0))
        new_centroids = np.vstack(new_centroids)
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids
    distances = ((features[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1)


def stratified_split(
    items: list[DatasetItem],
    style_groups: np.ndarray,
    train_ratio: float,
    seed: int,
) -> tuple[list[DatasetItem], list[DatasetItem]]:
    rng = random.Random(seed)
    by_label: dict[str, list[int]] = {}
    for idx, item in enumerate(items):
        by_label.setdefault(item.label, []).append(idx)

    train_items: list[DatasetItem] = []
    val_items: list[DatasetItem] = []
    for indices in by_label.values():
        by_style: dict[int, list[int]] = {}
        for idx in indices:
            by_style.setdefault(int(style_groups[idx]), []).append(idx)
        for style_indices in by_style.values():
            rng.shuffle(style_indices)
            split = max(1, int(len(style_indices) * train_ratio))
            train_items.extend(items[idx] for idx in style_indices[:split])
            val_items.extend(items[idx] for idx in style_indices[split:])
    return train_items, val_items


def load_images(
    items: list[DatasetItem],
    image_size: tuple[int, int],
    labels: list[str],
    log_every: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    label_to_index = {label: i for i, label in enumerate(labels)}
    features: list[np.ndarray] = []
    targets: list[int] = []
    total = len(items)
    for index, item in enumerate(items, start=1):
        data = np.fromfile(str(item.path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            continue
        processed = preprocess_cell(image, image_size)
        features.append(processed)
        targets.append(label_to_index[item.label])
        if log_every and index % log_every == 0:
            print(f"Loaded {index}/{total} images...")
    x = np.expand_dims(np.array(features), axis=-1)
    y = np.array(targets)
    return x, y
