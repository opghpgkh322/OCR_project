import cv2
import numpy as np


class MarkerDetectionError(RuntimeError):
    pass


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def _order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def detect_markers(image: np.ndarray, min_area: int = 500) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    side_lengths = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if w == 0 or h == 0:
                continue
            aspect = w / float(h)
            extent = area / float(w * h)
            if 0.8 <= aspect <= 1.2 and extent >= 0.6:
                squares.append(approx.reshape(4, 2))
                side_lengths.append((w + h) / 2)
    if len(squares) < 4:
        raise MarkerDetectionError("Not enough marker squares detected.")
    squares = sorted(squares, key=cv2.contourArea, reverse=True)[:4]
    if side_lengths:
        side_lengths = sorted(side_lengths, reverse=True)[:4]
        avg_side = float(np.mean(side_lengths))
    else:
        avg_side = 0.0
    centers = np.array([sq.mean(axis=0) for sq in squares])
    return _order_points(centers), avg_side


def align_image(image: np.ndarray, output_size: tuple[int, int] | None = None) -> np.ndarray:
    markers, avg_side = detect_markers(image)
    tl, tr, br, bl = markers
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    width = int(max(width_top, width_bottom))
    height = int(max(height_left, height_right))
    padding = int(avg_side * 0.5) if avg_side else 0
    width = max(1, width + padding)
    height = max(1, height + padding)
    if output_size is None:
        output_size = (width, height)
    dst = np.array(
        [
            [0, 0],
            [output_size[0] - 1, 0],
            [output_size[0] - 1, output_size[1] - 1],
            [0, output_size[1] - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(markers.astype("float32"), dst)
    warped = cv2.warpPerspective(image, matrix, output_size)
    return warped


def preprocess_cell(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    resized = cv2.resize(thresh, size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    return normalized
