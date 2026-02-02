import argparse
from pathlib import Path

import cv2

from ocr_app.config import CellConfig, SheetConfig
from ocr_app.preprocessing import align_image, load_image

WINDOW_NAME = "OCR Configurator"
MAX_DISPLAY_WIDTH = 1400
MAX_DISPLAY_HEIGHT = 900


class ClickCollector:
    def __init__(self, scale: float) -> None:
        self.points: list[tuple[int, int]] = []
        self.scale = scale

    def reset(self) -> None:
        self.points = []

    def on_click(self, event, x, y, _flags, _params):
        if event == cv2.EVENT_LBUTTONDOWN:
            orig_x = int(x / self.scale)
            orig_y = int(y / self.scale)
            self.points.append((orig_x, orig_y))


def draw_preview(image, cells, scale: float):
    if scale != 1.0:
        preview = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        preview = image.copy()
    for cell in cells:
        cv2.rectangle(
            preview,
            (int(cell.x * scale), int(cell.y * scale)),
            (int((cell.x + cell.w) * scale), int((cell.y + cell.h) * scale)),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            preview,
            f"{cell.label}:{cell.index}",
            (int(cell.x * scale), max(10, int((cell.y - 5) * scale))),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.4, 0.5 * scale),
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return preview


def compute_scale(image) -> float:
    return min(
        MAX_DISPLAY_WIDTH / image.shape[1],
        MAX_DISPLAY_HEIGHT / image.shape[0],
        1.0,
    )


def is_mostly_blank(image) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_val, std_val = cv2.meanStdDev(gray)
    mean_scalar = float(mean_val[0][0])
    std_scalar = float(std_val[0][0])
    return mean_scalar > 245.0 and std_scalar < 5.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure OCR form cells.")
    parser.add_argument("--image", help="Path to a sample scan image.")
    parser.add_argument("--output", default="sheet_config.json", help="Path to save config JSON.")
    args, _unknown = parser.parse_known_args()

    image_path = (args.image or "").strip() or input("Path to sample scan image: ").strip()
    if not image_path:
        raise SystemExit("Image path is required.")
    original = load_image(image_path)
    aligned = None
    try:
        aligned = align_image(original)
        if aligned is None or aligned.size == 0:
            raise ValueError("Aligned image is empty.")
        if is_mostly_blank(aligned):
            raise ValueError("Aligned image looks blank.")
    except Exception as exc:
        print(f"Alignment failed ({exc}).")
        aligned = None

    if aligned is None:
        raise SystemExit("Failed to align image using marker squares. Check marker visibility.")

    current_image = aligned
    scale = compute_scale(current_image)
    collector = ClickCollector(scale)
    cells: list[CellConfig] = []
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, collector.on_click)

    print("Instructions:")
    print("- Click two points (top-left and bottom-right) to create a cell rectangle.")
    print("- After each rectangle, enter label (last_name, first_name, patronymic, birth_date, phone).")
    print("- Enter index for order (0..n).")
    print("- Press 'q' in the image window when finished.")

    while True:
        preview = draw_preview(current_image, cells, scale)
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
        image_width=current_image.shape[1],
        image_height=current_image.shape[0],
        cells=cells,
    )
    output_path = Path(args.output)
    config.save(output_path)
    print(f"Saved config to {output_path}")


if __name__ == "__main__":
    main()
