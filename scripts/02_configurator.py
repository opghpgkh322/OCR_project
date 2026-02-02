import argparse
from pathlib import Path

import cv2

from ocr_app.config import CellConfig, SheetConfig
from ocr_app.preprocessing import align_image, load_image

WINDOW_NAME = "OCR Configurator"


class ClickCollector:
    def __init__(self) -> None:
        self.points: list[tuple[int, int]] = []

    def reset(self) -> None:
        self.points = []

    def on_click(self, event, x, y, _flags, _params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))


def draw_preview(image, cells):
    preview = image.copy()
    for cell in cells:
        cv2.rectangle(
            preview,
            (cell.x, cell.y),
            (cell.x + cell.w, cell.y + cell.h),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            preview,
            f"{cell.label}:{cell.index}",
            (cell.x, max(10, cell.y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return preview


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure OCR form cells.")
    parser.add_argument("--image", help="Path to a sample scan image.")
    parser.add_argument("--output", default="sheet_config.json", help="Path to save config JSON.")
    args = parser.parse_args()

    image_path = args.image or input("Path to sample scan image: ").strip()
    if not image_path:
        raise SystemExit("Image path is required.")
    image = load_image(image_path)
    aligned = align_image(image)

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
        preview = draw_preview(aligned, cells)
        cv2.imshow(WINDOW_NAME, preview)
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            break
        if len(collector.points) >= 2:
            (x1, y1), (x2, y2) = collector.points[:2]
            collector.reset()
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            if w == 0 or h == 0:
                print("Skipped zero-sized rectangle.")
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
