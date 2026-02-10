import argparse
from pathlib import Path
import cv2
import numpy as np
from ocr_app.preprocessing import load_image

WINDOW_NAME = "Manual Form Aligner"


class ManualAligner:
    def __init__(self, image: np.ndarray, output_size: tuple[int, int]):
        self.original = image.copy()
        self.output_size = output_size
        self.markers = []  # 4 точки: TL, TR, BR, BL
        self.scale = 1
        self.offset_x = 0
        self.offset_y = 0

    def reset_markers(self):
        self.markers = []

    def add_marker(self, x: int, y: int):
        """Добавляет маркер в реальных координатах (с учетом масштаба и сдвига)"""
        real_x = int((x - self.offset_x) / self.scale)
        real_y = int((y - self.offset_y) / self.scale)

        # Ограничиваем координаты границами изображения
        real_x = max(0, min(self.original.shape[1] - 1, real_x))
        real_y = max(0, min(self.original.shape[0] - 1, real_y))

        if len(self.markers) < 4:
            self.markers.append((real_x, real_y))
            print(f"Маркер {len(self.markers)}/4: ({real_x}, {real_y})")

    def draw_preview(self) -> np.ndarray:
        """Рисует превью с маркерами"""
        # Масштабируем изображение для удобства
        h, w = self.original.shape[:2]
        scaled_w = int(w * self.scale)
        scaled_h = int(h * self.scale)

        display = cv2.resize(self.original, (scaled_w, scaled_h),
                             interpolation=cv2.INTER_LINEAR)

        # Рисуем уже поставленные маркеры
        marker_names = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]
        colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 128, 0)]

        for i, (mx, my) in enumerate(self.markers):
            # Переводим реальные координаты в экранные
            screen_x = int(mx * self.scale + self.offset_x)
            screen_y = int(my * self.scale + self.offset_y)

            cv2.circle(display, (screen_x, screen_y), 8, colors[i], -1)
            cv2.circle(display, (screen_x, screen_y), 10, (255, 255, 255), 2)

            # Подпись маркера
            cv2.putText(display, marker_names[i],
                        (screen_x + 15, screen_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)

        # Если все 4 маркера установлены, рисуем рамку
        if len(self.markers) == 4:
            screen_pts = []
            for mx, my in self.markers:
                sx = int(mx * self.scale + self.offset_x)
                sy = int(my * self.scale + self.offset_y)
                screen_pts.append((sx, sy))

            for i in range(4):
                cv2.line(display, screen_pts[i], screen_pts[(i + 1) % 4],
                         (0, 255, 0), 3)

        # Инструкция
        instructions = [
            "MANUAL ALIGNMENT MODE:",
            "1. Click 4 corners: TL -> TR -> BR -> BL",
            "2. +/- to zoom, Arrow keys to pan",
            "3. 'r' to reset markers",
            "4. 'Enter' to apply and save",
            "5. 'q' to quit without saving"
        ]

        y_offset = 30
        for line in instructions:
            cv2.putText(display, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25

        return display

    def apply_transform(self) -> np.ndarray:
        """Применяет перспективное преобразование"""
        if len(self.markers) != 4:
            raise ValueError("Нужно 4 маркера!")

        # Исходные точки (в порядке: TL, TR, BR, BL)
        src_pts = np.array(self.markers, dtype=np.float32)

        # Целевые точки (прямоугольник)
        dst_pts = np.array([
            [0, 0],
            [self.output_size[0] - 1, 0],
            [self.output_size[0] - 1, self.output_size[1] - 1],
            [0, self.output_size[1] - 1]
        ], dtype=np.float32)

        # Вычисляем матрицу трансформации
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Применяем
        aligned = cv2.warpPerspective(self.original, matrix, self.output_size)
        return aligned

    def run(self) -> np.ndarray | None:
        """Главный цикл взаимодействия"""
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1200, 800)

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.add_marker(x, y)

        cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

        while True:
            preview = self.draw_preview()
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKey(30)
            key_ascii = key & 0xFF

            # Zoom
            if key_ascii in (ord("+"), ord("=")):
                self.scale = min(5.0, self.scale * 1.2)
            elif key_ascii == ord("-"):
                self.scale = max(0.2, self.scale / 1.2)

            # Pan (стрелки)
            elif key == 2555904 or key_ascii == 82:  # UP
                self.offset_y += 20
            elif key == 2424832 or key_ascii == 84:  # DOWN
                self.offset_y -= 20
            elif key == 2490368 or key_ascii == 81:  # LEFT
                self.offset_x += 20
            elif key == 2621440 or key_ascii == 83:  # RIGHT
                self.offset_x -= 20

            # Reset
            elif key_ascii == ord("r"):
                self.reset_markers()
                print("Маркеры сброшены")

            # Apply
            elif key_ascii in (13, 10):  # Enter
                if len(self.markers) == 4:
                    try:
                        aligned = self.apply_transform()
                        cv2.destroyAllWindows()
                        return aligned
                    except Exception as e:
                        print(f"Ошибка трансформации: {e}")
                else:
                    print(f"Установлено {len(self.markers)}/4 маркеров. Нужно все 4!")

            # Quit
            elif key_ascii == ord("q"):
                cv2.destroyAllWindows()
                return None

        cv2.destroyAllWindows()
        return None


def main():
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Manually align form with precise corner selection.")
    parser.add_argument("--input", default=str(repo_root / "scans" / "IMG_0001.jpg"),
                        help="Path to scanned form")
    parser.add_argument("--output", default=str(repo_root / "scripts" / "aligned_form.jpg"),
                        help="Where to save aligned form")
    parser.add_argument("--width", type=int, default=2480,
                        help="Output width in pixels")
    parser.add_argument("--height", type=int, default=3508,
                        help="Output height in pixels")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Файл не найден: {input_path}")
        print("Укажите путь к скану через --input")
        return

    print(f"📄 Загрузка изображения: {input_path}")
    image = load_image(str(input_path))

    print(f"📐 Размер выходного изображения: {args.width}x{args.height}")
    print("\n" + "=" * 60)
    print("ИНСТРУКЦИЯ:")
    print("1. Кликните по 4 углам формы в порядке:")
    print("   TOP-LEFT -> TOP-RIGHT -> BOTTOM-RIGHT -> BOTTOM-LEFT")
    print("2. Используйте +/- для зума, стрелки для перемещения")
    print("3. 'r' - сбросить маркеры и начать заново")
    print("4. Enter - применить и сохранить")
    print("5. 'q' - выйти без сохранения")
    print("=" * 60 + "\n")

    aligner = ManualAligner(image, (args.width, args.height))
    aligned = aligner.run()

    if aligned is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), aligned)
        print(f"\n✅ Выровненная форма сохранена: {output_path}")
        print(f"Теперь запустите конфигуратор:")
        print(f"  python scripts/02_configurator.py")
    else:
        print("\n❌ Операция отменена")


if __name__ == "__main__":
    main()
