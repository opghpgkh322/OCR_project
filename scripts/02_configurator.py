import argparse
from pathlib import Path

import cv2

from ocr_app.config import CellConfig, SheetConfig
from ocr_app.preprocessing import load_image

WINDOW_NAME = "OCR Configurator"
LABEL_OPTIONS = ["last_name", "first_name", "patronymic", "birth_date", "phone"]
LABEL_SHORTCUTS = {ord(str(i + 1)): label for i, label in enumerate(LABEL_OPTIONS)}


class ClickCollector:
    def __init__(self) -> None:
        self.points: list[tuple[int, int]] = []

    def reset(self) -> None:
        self.points = []

    def on_click(self, event, x, y, _flags, _params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))


def next_index(cells: list[CellConfig], label: str) -> int:
    indices = [cell.index for cell in cells if cell.label == label]
    return (max(indices) + 1) if indices else 0


def draw_preview(image, cells, scale: int, active_label: str, collector_points, selected_idx: int | None):
    preview = cv2.resize(
        image,
        (max(1, image.shape[1] * scale), max(1, image.shape[0] * scale)),
        interpolation=cv2.INTER_NEAREST,
    )

    for idx, cell in enumerate(cells):
        color = (0, 255, 255) if idx == selected_idx else (0, 255, 0)
        thickness = 3 if idx == selected_idx else 2
        cv2.rectangle(
            preview,
            (cell.x * scale, cell.y * scale),
            ((cell.x + cell.w) * scale, (cell.y + cell.h) * scale),
            color,
            thickness,
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

    if len(collector_points) == 1:
        px, py = collector_points[0]
        cv2.circle(preview, (px, py), 4, (255, 255, 0), -1)

    lines = [
        f"scale={scale} active={active_label}",
        "Mouse: click 2 points to add cell",
        "1-5 set category | +/- zoom | tab next cell",
        "[ ] change selected index | r reset points",
        "u undo last | del delete selected | q save&quit",
    ]
    y = 24
    for line in lines:
        cv2.putText(preview, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return preview


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure OCR form cells.")
    parser.add_argument("--image", default="aligned_form.jpg", help="Path to an aligned form image.")
    parser.add_argument("--output", default="sheet_config.json", help="Path to save config JSON.")
    parser.add_argument("--scale", type=int, default=3, help="Initial zoom scale.")
    parser.add_argument("--inset", type=int, default=1, help="Inset in pixels to avoid touching borders.")
    args = parser.parse_args()

    aligned = load_image(args.image)
    scale = max(1, args.scale)
    active_label = LABEL_OPTIONS[0]
    collector = ClickCollector()
    cells: list[CellConfig] = []
    selected_idx: int | None = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, collector.on_click)

    while True:
        preview = draw_preview(aligned, cells, scale, active_label, collector.points, selected_idx)
        cv2.imshow(WINDOW_NAME, preview)
        key = cv2.waitKey(30) & 0xFF

        if key in LABEL_SHORTCUTS:
            chosen = LABEL_SHORTCUTS[key]
            active_label = chosen
            if selected_idx is not None and 0 <= selected_idx < len(cells):
                cells[selected_idx].label = chosen
                if cells[selected_idx].index < 0:
                    cells[selected_idx].index = next_index(cells, chosen)
        elif key in (ord("+"), ord("=")):
            scale = min(12, scale + 1)
        elif key == ord("-"):
            scale = max(1, scale - 1)
        elif key == 9:  # TAB
            if cells:
                if selected_idx is None:
                    selected_idx = 0
                else:
                    selected_idx = (selected_idx + 1) % len(cells)
        elif key == ord("["):
            if selected_idx is not None and 0 <= selected_idx < len(cells):
                cells[selected_idx].index -= 1
        elif key == ord("]"):
            if selected_idx is not None and 0 <= selected_idx < len(cells):
                cells[selected_idx].index += 1
        elif key == ord("u"):
            if cells:
                cells.pop()
                if selected_idx is not None and selected_idx >= len(cells):
                    selected_idx = len(cells) - 1 if cells else None
        elif key == ord("r"):
            collector.reset()
        elif key == 127:  # delete
            if selected_idx is not None and 0 <= selected_idx < len(cells):
                cells.pop(selected_idx)
                if not cells:
                    selected_idx = None
                else:
                    selected_idx = min(selected_idx, len(cells) - 1)
        elif key == ord("q"):
            break

        if len(collector.points) >= 2:
            (x1, y1), (x2, y2) = collector.points[:2]
            collector.reset()
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

            cell = CellConfig(label=active_label, index=next_index(cells, active_label), x=x, y=y, w=w, h=h)
            cells.append(cell)
            selected_idx = len(cells) - 1

    cv2.destroyAllWindows()

    cells = sorted(cells, key=lambda c: (c.label, c.index, c.y, c.x))

    config = SheetConfig(
        version=1,
        image_width=aligned.shape[1],
        image_height=aligned.shape[0],
        cells=cells,
    )
    output_path = Path(args.output)
    config.save(output_path)
    print(f"Saved {len(cells)} cells to {output_path}")


if __name__ == "__main__":
    main()
