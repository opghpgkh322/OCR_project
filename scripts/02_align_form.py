import argparse
from pathlib import Path
import cv2
from ocr_app.preprocessing import load_image, align_image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align a scanned OCR form using marker squares with top padding."
    )

    parser.add_argument(
        "--image",
        default=r"C:\Users\vocraths\Desktop\OCR_project\scans\IMG_0001.jpg",
        help="Path to the raw scan image.",
    )

    parser.add_argument(
        "--output",
        default="scripts/aligned_form.jpg",
        help="Path to save the aligned image."
    )

    parser.add_argument(
        "--top-padding",
        type=int,
        default=15,
        help="Extra pixels to add at the top (default: 15)"
    )

    args = parser.parse_args()

    input_path = Path(args.image)
    if not input_path.exists():
        print(f"❌ Файл не найден: {input_path}")
        return

    print(f"📄 Загрузка: {input_path}")
    image = load_image(str(input_path))

    print(f"🔧 Выравнивание с отступом сверху: {args.top_padding}px")
    aligned = align_image(image, top_padding=args.top_padding)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), aligned)
    print(f"✅ Выровненная форма сохранена: {output_path}")
    print(f"   Размер: {aligned.shape[1]}x{aligned.shape[0]}")
    print(f"\nТеперь ваша старая конфигурация будет работать идеально!")
    print(f"Только перенастройте ячейки last_name через конфигуратор.")


if __name__ == "__main__":
    main()
