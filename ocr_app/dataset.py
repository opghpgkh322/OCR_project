import json
from dataclasses import dataclass
from pathlib import Path


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

