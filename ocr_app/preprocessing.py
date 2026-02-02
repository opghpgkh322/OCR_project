import cv2
import numpy as np


class MarkerDetectionError(RuntimeError):
    pass


def load_image(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
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


def _select_corner_markers(centers: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray | None:
    height, width = shape[:2]
    corners = np.array(
        [
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ],
        dtype="float32",
    )
    selected = []
    used: set[int] = set()
    for corner in corners:
        distances = [
            (idx, np.linalg.norm(center - corner))
            for idx, center in enumerate(centers)
            if idx not in used
        ]
        if not distances:
            return None
        closest_idx, _ = min(distances, key=lambda item: item[1])
        used.add(closest_idx)
        selected.append(centers[closest_idx])
    return np.array(selected, dtype="float32")


def detect_markers(
    image: np.ndarray,
    min_area: int = 500,
    min_fill: float = 0.7,
    aspect_tolerance: float = 0.25,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[np.ndarray] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if w == 0 or h == 0:
            continue
        aspect_ratio = w / float(h)
        if not (1 - aspect_tolerance <= aspect_ratio <= 1 + aspect_tolerance):
            continue
        fill_ratio = area / float(w * h)
        if fill_ratio < min_fill:
            continue
        candidates.append(approx.reshape(4, 2))

    if len(candidates) < 4:
        raise MarkerDetectionError("Not enough marker squares detected.")

    centers = np.array([square.mean(axis=0) for square in candidates])
    corner_markers = _select_corner_markers(centers, image.shape)
    if corner_markers is None:
        squares = sorted(candidates, key=cv2.contourArea, reverse=True)[:4]
        centers = np.array([sq.mean(axis=0) for sq in squares])
        return _order_points(centers)
    return _order_points(corner_markers)


def align_image(image: np.ndarray, output_size: tuple[int, int] | None = None) -> np.ndarray:
    markers = detect_markers(image)
    tl, tr, br, bl = markers
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    width = int(max(width_top, width_bottom))
    height = int(max(height_left, height_right))
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
