import argparse
from pathlib import Path

import cv2
import numpy as np

from ocr_app.config import CellConfig, SheetConfig
from ocr_app.preprocessing import load_image

WINDOW_NAME = "OCR Configurator"
LABEL_OPTIONS = ["last_name", "first_name", "patronymic", "birth_date", "phone"]


class ClickCollector:
    def __init__(self) -> None:
        self.points: list[tuple[int, int]] = []

    def reset(self) -> None:
        self.points = []

    def on_click(self, event, x, y, _flags, _params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))


def find_cell_at(cells: list[CellConfig], x: int, y: int) -> int | None:
    for idx in range(len(cells) - 1, -1, -1):
        cell = cells[idx]
        if cell.x <= x <= cell.x + cell.w and cell.y <= y <= cell.y + cell.h:
            return idx
    return None


def draw_preview(image, cells, scale: int, selected_idx: int | None):
    preview = cv2.resize(
        image,
        (max(1, image.shape[1] * scale), max(1, image.shape[0] * scale)),
        interpolation=cv2.INTER_NEAREST,
    )

    for idx, cell in enumerate(cells):
        color = (0, 255, 255) if idx == selected_idx else (0, 255, 0)
        cv2.rectangle(
            preview,
            (cell.x * scale, cell.y * scale),
            ((cell.x + cell.w) * scale, (cell.y + cell.h) * scale),
            color,
            2,
        )
        cv2.putText(
            preview,
            f"{idx}:{cell.label}:{cell.index}",
            (cell.x * scale, max(14, (cell.y - 4) * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return preview


def choose_category_dialog(current: str | None = None) -> str:
    window = "Choose category"
    canvas = np.zeros((220, 520, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)
    cv2.putText(canvas, "Select category (press 1..5)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for i, label in enumerate(LABEL_OPTIONS, start=1):
        color = (0, 255, 255) if label == current else (200, 200, 200)
        cv2.putText(canvas, f"{i}. {label}", (30, 35 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(canvas, "Esc: cancel", (30, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    while True:
        cv2.imshow(window, canvas)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            cv2.destroyWindow(window)
            return current or LABEL_OPTIONS[0]
        if ord("1") <= key <= ord("5"):
            chosen = LABEL_OPTIONS[key - ord("1")]
            cv2.destroyWindow(window)
            return chosen


def choose_index_dialog(current: int | None = None) -> int:
    window = "Choose index"
    value = max(0, current or 0)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    while True:
        canvas = np.zeros((180, 520, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)
        cv2.putText(canvas, "Choose index", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"Current: {value}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(canvas, "Left/Right or -/+ to change", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.putText(canvas, "Enter: confirm, Esc: cancel", (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.imshow(window, canvas)

        key = cv2.waitKey(0) & 0xFF
        if key in (13, 10):
            cv2.destroyWindow(window)
            return value
        if key == 27:
            cv2.destroyWindow(window)
            return current if current is not None else 0
        if key in (81, ord("-")):  # left
            value = max(0, value - 1)
        elif key in (83, ord("+"), ord("=")):  # right
            value += 1


def edit_cell_dialog(cell: CellConfig) -> None:
    cell.label = choose_category_dialog(cell.label)
    cell.index = choose_index_dialog(cell.index)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Configure OCR form cells.")
    parser.add_argument(
        "--image",
        default=str(repo_root / "scripts" / "aligned_form.jpg"),
        help="Path to an aligned form image.",
    )
    parser.add_argument(
        "--output",
        default=str(repo_root / "sheet_config.json"),
        help="Path to save config JSON.",
    )
    parser.add_argument(
        "--input-config",
        default="",
        help="Optional existing config to load at startup (defaults to --output if file exists).",
    )
    parser.add_argument("--scale", type=int, default=3, help="Initial zoom scale.")
    parser.add_argument("--inset", type=int, default=1, help="Inset in pixels to avoid touching borders.")
    args = parser.parse_args()

    aligned = load_image(args.image)
    scale = max(1, args.scale)
    collector = ClickCollector()
    output_path = Path(args.output).resolve()
    input_config_path = Path(args.input_config).resolve() if args.input_config else output_path

    cells: list[CellConfig] = []
    if input_config_path.exists():
        try:
            loaded = SheetConfig.load(input_config_path)
            cells = list(loaded.cells)
            print(f"Loaded {len(cells)} cells from {input_config_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to load existing config {input_config_path}: {exc}")
    selected_idx: int | None = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, collector.on_click)
    print("Controls: draw with 2 clicks | tab next | enter edit selected | +/- zoom | del remove | u undo | q save")

    while True:
        preview = draw_preview(aligned, cells, scale, selected_idx)
        cv2.imshow(WINDOW_NAME, preview)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("+"), ord("=")):
            scale = min(12, scale + 1)
        elif key == ord("-"):
            scale = max(1, scale - 1)
        elif key == 9:  # TAB
            if cells:
                selected_idx = 0 if selected_idx is None else (selected_idx + 1) % len(cells)
        elif key in (13, 10):
            if selected_idx is not None and 0 <= selected_idx < len(cells):
                edit_cell_dialog(cells[selected_idx])
        elif key == 127:
            if selected_idx is not None and 0 <= selected_idx < len(cells):
                cells.pop(selected_idx)
                selected_idx = None if not cells else min(selected_idx, len(cells) - 1)
        elif key == ord("u"):
            if cells:
                cells.pop()
                selected_idx = None if not cells else min((selected_idx or 0), len(cells) - 1)
        elif key == ord("q"):
            break

        while collector.points:
            x_click, y_click = collector.points.pop(0)
            x_img = x_click // scale
            y_img = y_click // scale
            hit_idx = find_cell_at(cells, x_img, y_img)

            if hit_idx is not None:
                selected_idx = hit_idx
                edit_cell_dialog(cells[selected_idx])
                collector.reset()
                continue

            collector.points.insert(0, (x_click, y_click))
            break

        if len(collector.points) >= 2:
            (x1, y1), (x2, y2) = collector.points[:2]
            collector.points = collector.points[2:]
            x = min(x1, x2) // scale
            y = min(y1, y2) // scale
            w = abs(x2 - x1) // scale
            h = abs(y2 - y1) // scale

            inset = max(0, args.inset)
            x += inset
            y += inset
            w -= inset * 2
            h -= inset * 2
            if w <= 0 or h <= 0:
                continue

            cell = CellConfig(label=LABEL_OPTIONS[0], index=0, x=x, y=y, w=w, h=h)
            edit_cell_dialog(cell)
            cells.append(cell)
            selected_idx = len(cells) - 1

    cv2.destroyAllWindows()

    cells = sorted(cells, key=lambda c: (c.label, c.index, c.y, c.x))
    config = SheetConfig(version=1, image_width=aligned.shape[1], image_height=aligned.shape[0], cells=cells)
    config.save(output_path)
    print(f"Saved {len(cells)} cells to {output_path}")


if __name__ == "__main__":
    main()