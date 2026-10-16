"""Flowchart component detection with an optional Ultralytics YOLO model.

Without trained weights this module provides a transparent OpenCV proposal
baseline. Its boxes are candidates, not a substitute for a labeled detector.
"""
from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

from .preprocess import preprocess_image


def _box(x1: int, y1: int, x2: int, y2: int, image: np.ndarray, kind: str,
         confidence: float, source: str) -> dict[str, Any]:
    h, w = image.shape[:2]
    return {"id": "", "type": kind, "bbox": [max(0, x1), max(0, y1), min(w, x2), min(h, y2)],
            "confidence": round(float(confidence), 4), "source": source}


def detect_components(image: np.ndarray, model_path: str | None = None,
                      confidence: float = 0.25) -> list[dict[str, Any]]:
    """Detect Text, Arrow, and Box candidates; keep overlapping types separate."""
    weight_path = model_path or os.getenv("RADU_DETECTOR_WEIGHTS")
    if weight_path and os.path.isfile(weight_path):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install ultralytics to load RADU_DETECTOR_WEIGHTS") from exc
        model = YOLO(weight_path)
        names = model.names
        output = []
        for result in model.predict(image, conf=confidence, verbose=False):
            for item in result.boxes:
                label = str(names[int(item.cls.item())]).lower()
                kind = "Text" if "text" in label else "Arrow" if "arrow" in label else "Box" if any(k in label for k in ("box", "process", "decision", "terminator")) else None
                if kind:
                    x1, y1, x2, y2 = map(int, item.xyxy[0].tolist())
                    output.append(_box(x1, y1, x2, y2, image, kind, item.conf.item(), "yolo"))
        return _assign_ids(output)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 12)
    h, w = gray.shape[:2]
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    found: list[dict[str, Any]] = []
    shape_masks = np.zeros_like(binary)
    if hierarchy is not None:
        for i, contour in enumerate(contours):
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if area < max(100, h * w * 0.00015) or cw < 24 or ch < 18:
                continue
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
            ratio = cw / max(1, ch)
            child = hierarchy[0][i][2]
            if len(approx) >= 4 and len(approx) <= 8 and 0.55 <= ratio <= 5.0 and (child >= 0 or area > h * w * 0.0008):
                found.append(_box(x, y, x + cw, y + ch, image, "Box", 0.35, "opencv-proposal"))
                cv2.drawContours(shape_masks, [contour], -1, 255, thickness=max(2, int(min(h, w) * .008)))

    # Text proposals are connected ink regions after removing long lines and shape strokes.
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 35), 1)))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 35))))
    strokes = cv2.bitwise_or(horizontal, vertical)
    clean = cv2.subtract(binary, strokes)
    clean = cv2.subtract(clean, cv2.dilate(shape_masks, np.ones((3, 3), np.uint8)))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(clean)
    text_boxes = []
    for x, y, cw, ch, area in stats[1:]:
        if area >= 10 and cw >= 3 and ch >= 5 and cw < w * .95 and ch < h * .15:
            text_boxes.append([int(x), int(y), int(x + cw), int(y + ch)])
    # Merge neighboring glyphs into line proposals, while retaining a separate Box proposal if overlapping.
    text_boxes.sort(key=lambda b: (b[1] // 12, b[0]))
    merged: list[list[int]] = []
    for b in text_boxes:
        match = next((m for m in reversed(merged) if abs(m[1] - b[1]) < max(10, int((m[3] - m[1]) * .7)) and b[0] - m[2] < 20), None)
        if match:
            match[:] = [min(match[0], b[0]), min(match[1], b[1]), max(match[2], b[2]), max(match[3], b[3])]
        else:
            merged.append(b.copy())
    found.extend(_box(*b, image, "Text", 0.3, "opencv-proposal") for b in merged)

    # Arrow/line proposals come from long Hough segments; the trained detector handles endpoints.
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(20, min(h, w) // 20),
                            minLineLength=max(8, min(h, w) // 40), maxLineGap=max(8, min(h, w) // 80))
    if lines is not None:
        segments = []
        for [[x1, y1, x2, y2]] in lines:
            length = float(np.hypot(x2 - x1, y2 - y1))
            angle = float(np.arctan2(y2 - y1, x2 - x1))
            segments.append(((int(x1), int(y1)), (int(x2), int(y2)), length, angle))
        candidates = []
        for p1, p2, length, angle in segments:
            if length <= min(h, w) * .12:
                continue
            # Require a pair of short, converging diagonal strokes at one endpoint.
            # This rejects most box borders and ordinary text under the proposal detector.
            head_end = None
            for endpoint, opposite in ((p1, p2), (p2, p1)):
                arms = []
                for q1, q2, qlength, qangle in segments:
                    if qlength < 4 or qlength > min(48, min(h, w) * .15):
                        continue
                    delta = abs((qangle - angle + np.pi / 2) % np.pi - np.pi / 2)
                    if not np.deg2rad(18) <= delta <= np.deg2rad(78):
                        continue
                    near = min(np.hypot(endpoint[0]-q1[0], endpoint[1]-q1[1]),
                               np.hypot(endpoint[0]-q2[0], endpoint[1]-q2[1]))
                    if near <= max(8, min(h, w) * .018):
                        arms.append(qangle)
                has_pair = any(min(abs(a-b) % np.pi, np.pi - (abs(a-b) % np.pi)) > np.deg2rad(22)
                               for i, a in enumerate(arms) for b in arms[i+1:])
                if has_pair:
                    head_end = endpoint
                    break
            if head_end is None:
                continue
            start, end = (opposite, head_end) if head_end == p1 else (p1, p2)
            candidate = _box(min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[0], p2[0]) + 1, max(p1[1], p2[1]) + 1, image, "Arrow", .35, "opencv-arrowhead-proposal")
            candidate["start"] = list(start)
            candidate["end"] = list(end)
            candidate["arrowhead_detected"] = True
            if not any(_box_iou(candidate["bbox"], old["bbox"]) > .7 for old in candidates):
                candidates.append(candidate)
        found.extend(candidates)
    return _assign_ids(found)


def _assign_ids(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for i, item in enumerate(sorted(items, key=lambda x: (x["bbox"][1], x["bbox"][0], x["type"]))):
        item["id"] = f"{item['type'].lower()}-{i + 1:04d}"
    return sorted(items, key=lambda x: x["id"])


def _box_iou(a: list[int], b: list[int]) -> float:
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    intersection=max(0,x2-x1)*max(0,y2-y1)
    area_a=max(1,(a[2]-a[0])*(a[3]-a[1])); area_b=max(1,(b[2]-b[0])*(b[3]-b[1]))
    return intersection/(area_a+area_b-intersection)


def analyze_image(image: np.ndarray, model_path: str | None = None) -> dict[str, Any]:
    processed = preprocess_image(image)
    return {"version": "1.0", "image": {"width": int(processed.shape[1]), "height": int(processed.shape[0])},
            "elements": detect_components(processed, model_path=model_path), "relations": [], "warnings": [
                {"code": "proposal_detector", "message": "OpenCV 후보 검출 결과입니다. 학습된 검출 가중치가 있으면 RADU_DETECTOR_WEIGHTS로 지정하세요."}
            ]}
