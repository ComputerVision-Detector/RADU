"""Image cleanup helpers for photographed hand-drawn flowcharts."""
from __future__ import annotations

import cv2
import numpy as np


def _order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def correct_perspective(image: np.ndarray) -> np.ndarray:
    """Rectify a prominent four-sided page; return the input if none is found."""
    if image is None or image.size == 0:
        raise ValueError("Image is empty")
    h, w = image.shape[:2]
    scale = min(1.0, 1200 / max(h, w))
    small = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image.copy()
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    quad = None
    for contour in candidates:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.contourArea(approx) > small.shape[0] * small.shape[1] * 0.2:
            quad = approx.reshape(4, 2).astype("float32") / scale
            break
    if quad is None:
        return image
    tl, tr, br, bl = _order_points(quad)
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 32 or height < 32:
        return image
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype="float32"), dst)
    return cv2.warpPerspective(image, matrix, (width, height))


def correct_rotation(image: np.ndarray) -> np.ndarray:
    """Correct small page skew estimated from dominant foreground lines."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1800, threshold=80,
                            minLineLength=max(30, min(gray.shape) // 8), maxLineGap=12)
    if lines is None:
        return image
    angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for [[x1, y1, x2, y2]] in lines]
    angles = [a if a <= 45 else a - 90 if a > 45 else a for a in angles]
    angles = [a for a in angles if abs(a) <= 15]
    if not angles:
        return image
    angle = float(np.median(angles))
    if abs(angle) < 0.3:
        return image
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def normalize_brightness(image: np.ndarray) -> np.ndarray:
    """Apply local contrast normalization while preserving color when possible."""
    if image.ndim == 2:
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Run perspective, rotation, and brightness correction in that order."""
    if image is None or image.size == 0:
        raise ValueError("Image is empty")
    return normalize_brightness(correct_rotation(correct_perspective(image)))
