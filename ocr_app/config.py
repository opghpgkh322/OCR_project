import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class CellConfig:
    label: str
    index: int
    x: int
    y: int
    w: int
    h: int


@dataclass
class SheetConfig:
    version: int
    image_width: int
    image_height: int
    cells: list[CellConfig]

    def save(self, path: str | Path) -> None:
        payload = asdict(self)
        payload["cells"] = [asdict(cell) for cell in self.cells]
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "SheetConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cells = [CellConfig(**item) for item in data["cells"]]
        return SheetConfig(
            version=data["version"],
            image_width=data["image_width"],
            image_height=data["image_height"],
            cells=cells,
        )
