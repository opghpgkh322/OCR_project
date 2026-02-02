import argparse

import cv2

from ocr_app.preprocessing import align_image, load_image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align a scanned OCR form using marker squares."
    )
    parser.add_argument(
        "--image",
        default=r"C:\\Users\\vocraths\\Desktop\\OCR_project\\scans\\IMG_0001.jpg",
        help="Path to the raw scan image.",
    )
    parser.add_argument(
        "--output", default="aligned_form.jpg", help="Path to save the aligned image."
    )
    args = parser.parse_args()

    image = load_image(args.image)
    aligned = align_image(image)
    cv2.imwrite(args.output, aligned)
    print(f"Aligned image saved to {args.output}")


if __name__ == "__main__":
    main()
