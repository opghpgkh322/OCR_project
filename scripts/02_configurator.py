import argparse
from pathlib import Path

import cv2

from ocr_app.config import CellConfig, SheetConfig
from ocr_app.preprocessing import load_image

WINDOW_NAME = "OCR Configurator"


class ClickCollector:
    def __init__(self) -> None:
        self.points: list[tuple[int, int]] = []

    def reset(self) -> None:
        self.points = []

    def on_click(self, event, x, y, _flags, _params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))


def draw_preview(image, cells, scale: int):
    if scale > 1:
        preview = cv2.resize(
            image, (image.shape[1] * scale, image.shape[0] * scale), interpolation=cv2.INTER_NEAREST
        )
    else:
        preview = image.copy()
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
            (cell.x * scale, max(10, (cell.y - 5) * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return preview


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure OCR form cells.")
    parser.add_argument(
        "--image",
        default="aligned_form.jpg",
        help="Path to an aligned form image (use the alignment step first).",
    )
    parser.add_argument("--output", default="sheet_config.json", help="Path to save config JSON.")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for pixel-precise selection.")
    parser.add_argument("--inset", type=int, default=1, help="Inset in pixels to avoid touching borders.")
    args = parser.parse_args()

    aligned = load_image(args.image)
    scale = max(1, args.scale)

    collector = ClickCollector()
    cells: list[CellConfig] = []
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, collector.on_click)

    print("Instructions:")
    print("- Click two points (top-left and bottom-right) to create a cell rectangle.")
    print("- After each rectangle, enter label (last_name, first_name, patronymic, birth_date, phone).")
    print("- Enter index for order (0..n).")
    print("- Press 'q' in the image window when finished.")

    while True:
        preview = draw_preview(aligned, cells, scale)
        cv2.imshow(WINDOW_NAME, preview)
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
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
            if w == 0 or h == 0:
                print("Skipped zero-sized rectangle.")
                continue
            if w < 0 or h < 0:
                print("Skipped rectangle after inset.")
                continue
            label = input("Label for this cell: ").strip()
            index = int(input("Index for this cell: ").strip())
            cells.append(CellConfig(label=label, index=index, x=x, y=y, w=w, h=h))

    cv2.destroyAllWindows()

    config = SheetConfig(
        version=1,
        image_width=aligned.shape[1],
        image_height=aligned.shape[0],
        cells=cells,
    )
    output_path = Path(args.output)
    config.save(output_path)
    print(f"Saved config to {output_path}")


if __name__ == "__main__":
    main()