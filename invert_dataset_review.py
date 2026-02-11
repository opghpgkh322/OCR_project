import cv2
import numpy as np
from pathlib import Path


def normalize_dataset():
    dataset_path = Path("dataset_review")

    if not dataset_path.exists():
        print(f"❌ ОШИБКА: Папка {dataset_path.absolute()} не найдена!")
        return

    print(f"📂 Сканирование папки: {dataset_path.absolute()}")

    # Ищем все картинки: jpg, jpeg, png (регистр не важен)
    extensions = {".jpg", ".jpeg", ".png"}
    files = [
        p for p in dataset_path.rglob("*")
        if p.suffix.lower() in extensions and p.is_file()
    ]

    print(f"🔎 Найдено изображений: {len(files)}")

    inverted_count = 0
    skipped_count = 0
    error_count = 0

    for img_path in files:
        # Читаем в ч/б режиме
        # Для путей с кириллицей используем numpy (cv2.imread может не сработать напрямую)
        try:
            stream = open(img_path, "rb")
            bytes_data = bytearray(stream.read())
            numpy_array = np.asarray(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(numpy_array, cv2.IMREAD_GRAYSCALE)
            stream.close()

            if img is None:
                print(f"⚠️ Пустой файл или ошибка чтения: {img_path.name}")
                error_count += 1
                continue

            # Средняя яркость: 0 (черный) ... 255 (белый)
            mean_brightness = np.mean(img)

            # Если яркость < 127, значит фон темный -> инвертируем
            if mean_brightness < 127:
                img_inverted = cv2.bitwise_not(img)

                # Сохраняем обратно. cv2.imwrite тоже не любит кириллицу, поэтому кодируем
                is_success, im_buf = cv2.imencode(img_path.suffix, img_inverted)
                if is_success:
                    im_buf.tofile(str(img_path))
                    inverted_count += 1
                else:
                    print(f"❌ Не удалось сохранить: {img_path.name}")
                    error_count += 1
            else:
                skipped_count += 1

        except Exception as e:
            print(f"❌ Ошибка с файлом {img_path.name}: {e}")
            error_count += 1

    print("-" * 30)
    print(f"✅ Готово!")
    print(f"🔄 Инвертировано (были черными): {inverted_count}")
    print(f"⏭️  Пропущено (уже белые): {skipped_count}")
    if error_count > 0:
        print(f"⚠️ Ошибок: {error_count}")


if __name__ == "__main__":
    normalize_dataset()
