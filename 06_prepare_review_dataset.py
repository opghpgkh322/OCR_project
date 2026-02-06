import argparse
from pathlib import Path

import cv2
import numpy as np


def invert_image(path: Path) -> bool:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return False
    inverted = 255 - image
    ok, encoded = cv2.imencode(path.suffix, inverted)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Invert review dataset colors to black-on-white.")
    parser.add_argument("--review-root", default="dataset_review", help="Folder with reviewed samples.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count files without modifying them.",
    )
    args = parser.parse_args()

    review_root = Path(args.review_root)
    if not review_root.exists():
        raise SystemExit(f"Review dataset not found: {review_root}")

    image_paths = [p for p in review_root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    print(f"Found {len(image_paths)} review images.")
    if args.dry_run:
        print("Dry run complete.")
        return

    success = 0
    for idx, path in enumerate(image_paths, start=1):
        if invert_image(path):
            success += 1
        if idx % 1000 == 0:
            print(f"Processed {idx}/{len(image_paths)}...")

    print(f"Inverted {success}/{len(image_paths)} images in {review_root}")


if __name__ == "__main__":
    main()