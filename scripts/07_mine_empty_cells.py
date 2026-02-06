import argparse
import cv2
import numpy as np
from pathlib import Path

from ocr_app.config import SheetConfig
from ocr_app.preprocessing import align_image, load_image

WINDOW_NAME = "Empty Cell Miner"


class ManualMiner:
    def __init__(self, scan_paths: list[Path], config: SheetConfig, output_dir: Path, scale: int = 2):
        self.scan_paths = scan_paths
        self.config = config
        self.output_dir = output_dir
        self.scale = scale

        self.current_idx = 0
        self.aligned = None
        self.selected_indices = set()
        self.stop_signal = False

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Конвертируем координаты клика (экранные) в координаты картинки
            real_x = x // self.scale
            real_y = y // self.scale

            for idx, cell in enumerate(self.config.cells):
                # Проверяем попадание в ячейку конфига
                if cell.x <= real_x <= cell.x + cell.w and cell.y <= real_y <= cell.y + cell.h:
                    if idx in self.selected_indices:
                        self.selected_indices.remove(idx)
                    else:
                        self.selected_indices.add(idx)
                    self.refresh_view()
                    break

    def refresh_view(self):
        if self.aligned is None:
            return

        view = self.aligned.copy()
        for idx, cell in enumerate(self.config.cells):
            # Синий - выбрано, Зеленый - не выбрано
            if idx in self.selected_indices:
                color = (255, 0, 0)
                thickness = -1  # Заливка
                alpha = 0.3

                # Рисуем прозрачную заливку
                overlay = view.copy()
                cv2.rectangle(overlay, (cell.x, cell.y), (cell.x + cell.w, cell.y + cell.h), color, -1)
                cv2.addWeighted(overlay, alpha, view, 1 - alpha, 0, view)

                # И яркую рамку
                cv2.rectangle(view, (cell.x, cell.y), (cell.x + cell.w, cell.y + cell.h), color, 2)
            else:
                color = (0, 255, 0)
                thickness = 1
                cv2.rectangle(view, (cell.x, cell.y), (cell.x + cell.w, cell.y + cell.h), color, thickness)

        # Масштабируем для экрана
        h, w = view.shape[:2]
        resized = cv2.resize(view, (w * self.scale, h * self.scale), interpolation=cv2.INTER_NEAREST)

        info = f"Scan {self.current_idx + 1}/{len(self.scan_paths)} | Selected: {len(self.selected_indices)}"
        cv2.putText(resized, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(resized, "[Click] Select | [Enter] Save | [N] Skip", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (50, 50, 50), 2)

        cv2.imshow(WINDOW_NAME, resized)

    def save_selection(self, filename_stem):
        count = 0
        for idx in self.selected_indices:
            cell = self.config.cells[idx]

            # Вырезаем ячейку из ВЫРОВНЕННОГО изображения
            y1 = max(0, cell.y)
            y2 = min(self.aligned.shape[0], cell.y + cell.h)
            x1 = max(0, cell.x)
            x2 = min(self.aligned.shape[1], cell.x + cell.w)

            crop = self.aligned[y1:y2, x1:x2]

            # Фильтр черных квадратов (если вдруг выравнивание все же сбойнуло)
            if crop.mean() < 20:
                print(f"Warning: Skipping too dark crop (index {cell.index})")
                continue

            save_name = f"{filename_stem}_empty_{cell.index}_{count}.png"
            save_path = self.output_dir / save_name
            cv2.imwrite(str(save_path), crop)
            count += 1
        return count

    def run(self):
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

        for i, path in enumerate(self.scan_paths):
            if self.stop_signal:
                break

            self.current_idx = i
            self.selected_indices = set()

            print(f"Processing {path.name}...")
            try:
                # 1. Грузим сырой скан
                raw_image = load_image(str(path))

                # 2. Выравниваем его жестко под размеры из конфига!
                # Именно так делал конфигуратор, поэтому сетка совпадет.
                target_size = (self.config.image_width, self.config.image_height)
                self.aligned = align_image(raw_image, output_size=target_size)

            except Exception as e:
                print(f"Error alignment {path.name}: {e}")
                continue

            self.refresh_view()

            while True:
                key = cv2.waitKey(0) & 0xFF

                if key == 27:  # ESC
                    self.stop_signal = True
                    break
                elif key == 13:  # Enter
                    saved = self.save_selection(path.stem)
                    print(f"Saved {saved} empty samples.")
                    break
                elif key == ord('n'):  # Next / Skip
                    print("Skipped.")
                    break
                elif key == ord('a'):  # Select All
                    self.selected_indices = set(range(len(self.config.cells)))
                    self.refresh_view()
                elif key == ord('c'):  # Clear
                    self.selected_indices.clear()
                    self.refresh_view()

        cv2.destroyAllWindows()
        print("Done.")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--scans", default=str(repo_root / "scans"))
    parser.add_argument("--config", default=str(repo_root / "sheet_config.json"))
    parser.add_argument("--output", default=str(repo_root / "dataset_external" / "Empty"))
    # Scale 1 лучше, чтобы видеть весь бланк. Если мелко - поставьте 2.
    parser.add_argument("--scale", type=int, default=1)
    args = parser.parse_args()

    config = SheetConfig.load(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    scan_paths = sorted(Path(args.scans).glob("*"))
    miner = ManualMiner(scan_paths, config, output_dir, args.scale)
    miner.run()


if __name__ == "__main__":
    main()