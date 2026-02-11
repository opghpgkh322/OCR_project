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
    corners = np.array([
        [0, 0], [width, 0], [width, height], [0, height],
    ], dtype="float32")
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


def detect_markers(image: np.ndarray, min_area: int = 500, min_fill: float = 0.7,
                   aspect_tolerance: float = 0.25) -> np.ndarray:
    # --- ЭТО ВЕРСИЯ, КОТОРАЯ РАБОТАЛА ИЗНАЧАЛЬНО ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[np.ndarray] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area: continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        if len(approx) != 4: continue

        x, y, w, h = cv2.boundingRect(approx)
        if w == 0 or h == 0: continue

        aspect_ratio = w / float(h)
        if not (1 - aspect_tolerance <= aspect_ratio <= 1 + aspect_tolerance): continue

        fill_ratio = area / float(w * h)
        if fill_ratio < min_fill: continue

        candidates.append(approx.reshape(4, 2))

    if len(candidates) < 4:
        # Если строгая проверка не прошла, пробуем просто взять 4 самых больших контура (fallback)
        sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
        if len(sorted_cnts) < 4:
            raise MarkerDetectionError("Not enough markers detected.")
        centers = []
        for c in sorted_cnts:
            M = cv2.moments(c)
            if M["m00"] != 0:
                centers.append([M["m10"] / M["m00"], M["m01"] / M["m00"]])
        if len(centers) < 4:
            raise MarkerDetectionError("Not enough markers detected.")
        return _order_points(np.array(centers, dtype="float32"))

    centers = np.array([square.mean(axis=0) for square in candidates])
    corner_markers = _select_corner_markers(centers, image.shape)
    if corner_markers is None:
        return _order_points(centers[:4])  # Fallback
    return _order_points(corner_markers)


def align_image(image: np.ndarray, output_size: tuple[int, int] | None = None, top_padding: int = 15) -> np.ndarray:
    """
    Выравнивает изображение формы по маркерам с дополнительным отступом сверху.

    Args:
        image: Исходное изображение
        output_size: Желаемый размер (ширина, высота). Если None, вычисляется автоматически
        top_padding: Дополнительные пиксели сверху для защиты от обрезания (по умолчанию 15)
    """
    markers = detect_markers(image)

    if output_size is None:
        # Если размер не задан, вычисляем по маркерам
        tl, tr, br, bl = markers
        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        width = int(max(width_top, width_bottom))
        height = int(max(height_left, height_right))
        output_size = (width, height + top_padding)  # Добавляем отступ к высоте
    else:
        # Если размер задан явно, тоже добавляем отступ
        output_size = (output_size[0], output_size[1] + top_padding)

    # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Целевые точки начинаются НЕ с (0,0), а с (0, top_padding)
    # Это сдвигает всё содержимое вниз, оставляя пустое место сверху
    dst = np.array([
        [0, top_padding],  # Верхний левый
        [output_size[0] - 1, top_padding],  # Верхний правый
        [output_size[0] - 1, output_size[1] - 1],  # Нижний правый
        [0, output_size[1] - 1],  # Нижний левый
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(markers, dst)
    return cv2.warpPerspective(image, matrix, output_size)


# --- НИЖЕ ФУНКЦИИ ПРЕПРОЦЕССИНГА (ОСТАВЛЯЕМ НОВЫЕ, ОНИ ХОРОШИЕ) ---

def clear_border(image: np.ndarray, border_size: int = 2) -> np.ndarray:
    h, w = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[border_size:h - border_size, border_size:w - border_size] = 255
    return cv2.bitwise_and(image, mask)


def center_content(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    coords = cv2.findNonZero(image)
    if coords is None:
        return cv2.resize(image, size)
    x, y, w, h = cv2.boundingRect(coords)
    roi = image[y:y + h, x:x + w]
    target_w, target_h = size
    scale = min((target_w - 6) / w, (target_h - 6) / h)
    scale = min(scale, 1.0)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized_roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized_roi
    return canvas


def preprocess_cell(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
    )

    # НОВОЕ: Проверка полярности
    # Если большая часть пикселей белая (>127), инвертируем
    mean_intensity = np.mean(thresh)
    if mean_intensity > 127:
        thresh = cv2.bitwise_not(thresh)

    cleaned = clear_border(thresh, border_size=3)
    centered = center_content(cleaned, size)
    return centered.astype("float32") / 255.0
