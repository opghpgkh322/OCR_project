import argparse
from collections import defaultdict
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

    def pop_last(self) -> None:
        if self.points:
            self.points.pop()

    def on_click(self, event, x, y, _flags, _params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))


def draw_preview(image, cells, scale: int, active_label: str, cursor_points: list[tuple[int, int]]):
    preview = cv2.resize(
        image,
        (max(1, image.shape[1] * scale), max(1, image.shape[0] * scale)),
        interpolation=cv2.INTER_NEAREST,
    )
    for cell in cells:
        cv2.rectangle(
            preview,
            (cell.x * scale, cell.y * scale),
            ((cell.x + cell.w) * scale, (cell.y + cell.h) * scale),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            preview,
            f"{cell.label}:{cell.index}",
            (cell.x * scale, max(14, (cell.y - 4) * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    if len(cursor_points) == 1:
        px, py = cursor_points[0]
        cv2.circle(preview, (px, py), 4, (255, 255, 0), -1)

    cv2.putText(
        preview,
        f"scale={scale} active={active_label}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        "1-5:label  +/-:zoom  u:undo  c:clear points  n:rename idx  q:save&quit",
        (10, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return preview


def next_index(cells: list[CellConfig], label: str) -> int:
    indices = [cell.index for cell in cells if cell.label == label]
    return (max(indices) + 1) if indices else 0


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

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, collector.on_click)

    print("Quick config mode")
    print("- Click top-left and bottom-right to add a cell.")
    print("- Labels: 1:last_name 2:first_name 3:patronymic 4:birth_date 5:phone")
    print("- Press +/- to zoom, u to undo last cell, n for manual index correction, q to finish.")

    while True:
        preview = draw_preview(aligned, cells, scale, active_label, collector.points)
        cv2.imshow(WINDOW_NAME, preview)
        key = cv2.waitKey(30) & 0xFF

        if key in LABEL_SHORTCUTS:
            active_label = LABEL_SHORTCUTS[key]
            print(f"Active label -> {active_label}")
        elif key in (ord("+"), ord("=")):
            scale = min(12, scale + 1)
            print(f"Zoom scale -> {scale}")
        elif key == ord("-"):
            scale = max(1, scale - 1)
            print(f"Zoom scale -> {scale}")
        elif key == ord("u"):
            if cells:
                removed = cells.pop()
                print(f"Removed: {removed.label}:{removed.index}")
        elif key == ord("c"):
            collector.reset()
        elif key == ord("n"):
            if not cells:
                continue
            target = input("Edit index for label (blank to skip): ").strip()
            if target:
                by_label = [c for c in cells if c.label == target]
                if not by_label:
                    print("No cells with this label.")
                else:
                    for idx, cell in enumerate(sorted(by_label, key=lambda x: x.index)):
                        print(f"{idx}: current index {cell.index} at ({cell.x},{cell.y})")
                    try:
                        row = int(input("Select row number: ").strip())
                        new_idx = int(input("New index: ").strip())
                        selected = sorted(by_label, key=lambda x: x.index)[row]
                        selected.index = new_idx
                    except (ValueError, IndexError):
                        print("Invalid selection.")
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
                print("Skipped invalid rectangle.")
                continue

            auto_idx = next_index(cells, active_label)
            manual = input(f"Cell -> {active_label}[{auto_idx}] Enter custom index or Enter: ").strip()
            if manual:
                try:
                    auto_idx = int(manual)
                except ValueError:
                    print("Invalid index. Keeping auto index.")
            cells.append(CellConfig(label=active_label, index=auto_idx, x=x, y=y, w=w, h=h))

    cv2.destroyAllWindows()

    # Normalize indexes per label for stable output ordering.
    grouped: dict[str, list[CellConfig]] = defaultdict(list)
    for cell in cells:
        grouped[cell.label].append(cell)
    normalized: list[CellConfig] = []
    for label, label_cells in grouped.items():
        for idx, cell in enumerate(sorted(label_cells, key=lambda c: (c.index, c.y, c.x))):
            cell.index = idx
            normalized.append(cell)

    config = SheetConfig(
        version=1,
        image_width=aligned.shape[1],
        image_height=aligned.shape[0],
        cells=normalized,
    )
    output_path = Path(args.output)
    config.save(output_path)
    print(f"Saved {len(normalized)} cells to {output_path}")


if __name__ == "__main__":
    main()
