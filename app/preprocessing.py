from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from app.config import BOARD_SIZE


def _order_points(points):
    points = np.asarray(points, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def _find_board_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.15:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            candidates.append((area, approx.reshape(4, 2)))
    if not candidates:
        raise ValueError("Chessboard boundary could not be detected. Use a clear image with the full board visible.")
    return _order_points(max(candidates, key=lambda x: x[0])[1])


def process_uploaded_image(image_path: str | Path, output_dir: str | Path) -> str:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded file is not a readable image.")

    corners = _find_board_corners(image)
    size = BOARD_SIZE
    destination = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners, destination)
    warped = cv2.warpPerspective(image, matrix, (size, size))

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    square_size = size // 8
    for row in range(8):
        for col in range(8):
            square = warped[row*square_size:(row+1)*square_size, col*square_size:(col+1)*square_size]
            path = output / f"square_r{row}_c{col}.jpg"
            if not cv2.imwrite(str(path), square, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"Failed to write extracted square: {path}")
    return str(output)
