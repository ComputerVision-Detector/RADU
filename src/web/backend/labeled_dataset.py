"""Hand-labelled note pages (Hugging Face images + team labels) for checking OCR in the web.

Images come from the dataset pinned in the label snapshot's catalog; labels are the frozen
``pilot-v2`` build of the ``RADU/labeling-data`` branch. Scoring reuses the matching and
CER definitions of ``src.evaluation.ocr_v2``. The labels cover text lines only, so diagram,
formula-structure and table output has no reference here.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import re
import threading

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "data/processed/ocr_v2/pilot-v2"
SNAPSHOT = ROOT / "data/processed/ocr_v2/snapshot-50aca98"
IMAGES = ROOT / "data/processed/ocr_v2/originals/images"
CACHE = ROOT / "data/processed/labeled_dataset_cache/images"
PAGE_ID_RE = re.compile(r"^hf-\d{4}$")
_GEOMETRY_LOCK = threading.Lock()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _load() -> dict:
    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    if _sha256(DATASET / "pages.jsonl") != manifest["artifacts_sha256"]["pages.jsonl"]:
        raise RuntimeError("라벨 데이터셋 pages.jsonl이 고정된 manifest와 다릅니다.")
    pages = {}
    for line in (DATASET / "pages.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            page = json.loads(line)
            pages[page["id"]] = page
    catalog = json.loads((SNAPSHOT / "dataset.catalog.json").read_text(encoding="utf-8"))
    snapshot = manifest["snapshot_manifest"]
    return {"pages": pages, "catalog": {sample["id"]: sample for sample in catalog["samples"]},
            "dataset": catalog["dataset"], "revision": catalog["revision"],
            "labels_branch": snapshot["branch"].removeprefix("refs/heads/"), "labels_commit": snapshot["labels_commit"]}


def get_page(page_id: str) -> dict:
    if not PAGE_ID_RE.fullmatch(page_id) or page_id not in _load()["pages"]:
        raise KeyError(page_id)
    return _load()["pages"][page_id]


def list_pages() -> dict:
    data = _load()
    pages = []
    for page in sorted(data["pages"].values(), key=lambda item: item["id"]):
        kinds = [region.get("type_confirmed") or region.get("type_suggestion") for region in page["regions"]]
        pages.append({"id": page["id"], "split": page["split"], "width": page["width"], "height": page["height"],
                      "annotator": page["annotator"], "regions": len(kinds), "text_regions": kinds.count("text"),
                      "formula_regions": kinds.count("formula"), "on_hold": bool(page["hold_reasons"]) or not page.get("eligible", False)})
    return {"dataset": data["dataset"], "revision": data["revision"], "labels_branch": data["labels_branch"],
            "labels_commit": data["labels_commit"], "pages": pages}


def image_path(page_id: str) -> Path:
    """The hash-verified original PNG: local restore first, otherwise the pinned Hugging Face file."""
    page = get_page(page_id)
    for folder in (IMAGES, CACHE):
        path = folder / f"{page_id}.png"
        if path.is_file():
            return path
    from huggingface_hub import hf_hub_download
    from PIL import Image, ImageOps

    data = _load()
    sample = data["catalog"][page_id]
    source = hf_hub_download(data["dataset"], sample["hf_file"], repo_type="dataset",
                             revision=sample.get("hf_revision", data["revision"]))
    with Image.open(source) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    if image.size != (page["width"], page["height"]) or hashlib.sha256(encoded.getvalue()).hexdigest() != page["source_sha256"]:
        raise RuntimeError(f"Hugging Face 이미지가 라벨의 원본과 다릅니다: {page_id}")
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{page_id}.png"
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(encoded.getvalue())
    temporary.replace(path)
    return path


class _Recorder:
    """Stands in for cv2 inside src.cv.preprocess to record the warps it applies."""

    def __init__(self, cv2, np):
        self._cv2, self._np = cv2, np
        self.matrix = np.eye(3)
        self.steps = []

    def __getattr__(self, name):
        return getattr(self._cv2, name)

    def warpPerspective(self, image, matrix, size, *args, **kwargs):
        self.matrix = self._np.asarray(matrix, dtype=float) @ self.matrix
        self.steps.append("perspective")
        return self._cv2.warpPerspective(image, matrix, size, *args, **kwargs)

    def warpAffine(self, image, matrix, size, *args, **kwargs):
        self.matrix = self._np.vstack([self._np.asarray(matrix, dtype=float), [0, 0, 1]]) @ self.matrix
        self.steps.append("rotation")
        return self._cv2.warpAffine(image, matrix, size, *args, **kwargs)


def preprocess_geometry(image):
    """(3x3 matrix original->processed, processed (width, height), applied steps).

    The web analyses the preprocessed page. preprocess.py is frozen and does not return
    its transform, so its own code runs once more here with the warp calls recorded.
    """
    import cv2
    import numpy as np
    from src.cv import preprocess

    with _GEOMETRY_LOCK:
        recorder = _Recorder(cv2, np)
        preprocess.cv2 = recorder
        try:
            warped = preprocess.correct_rotation(preprocess.correct_perspective(image))
        finally:
            preprocess.cv2 = cv2
    return recorder.matrix, (int(warped.shape[1]), int(warped.shape[0])), recorder.steps


def transform_box(box, matrix, width: int, height: int) -> list[float]:
    """Axis-aligned box of the transformed corners, clamped to the processed page."""
    x1, y1, x2, y2 = box
    xs, ys = [], []
    for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
        px, py, pw = (matrix[row][0] * x + matrix[row][1] * y + matrix[row][2] for row in range(3))
        xs.append(px / pw)
        ys.append(py / pw)
    return [max(0.0, min(width, min(xs))), max(0.0, min(height, min(ys))),
            max(0.0, min(width, max(xs))), max(0.0, min(height, max(ys)))]


def _line_box(raw, width: int, height: int):
    if raw and isinstance(raw[0], (list, tuple)):
        xs, ys = [point[0] for point in raw], [point[1] for point in raw]
        raw = [min(xs), min(ys), max(xs), max(ys)]
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    box = [max(0.0, min(width, float(raw[0]))), max(0.0, min(height, float(raw[1]))),
           max(0.0, min(width, float(raw[2]))), max(0.0, min(height, float(raw[3])))]
    return box if box[2] > box[0] and box[3] > box[1] else None


def _inside(box, reference) -> float:
    overlap = max(0, min(box[2], reference[2]) - max(box[0], reference[0])) * max(0, min(box[3], reference[3]) - max(box[1], reference[1]))
    return overlap / ((box[2] - box[0]) * (box[3] - box[1]))


def evaluate_page(page: dict, lines: list[dict], matrix, width: int, height: int, threshold: float = 0.5) -> dict:
    """Score OCR lines (processed-page pixels) against one labelled page.

    Regions on hold are not scored, and predictions lying mostly inside them are ignored,
    as in the saved-prediction evaluator. Unlike it, an ambiguous overlap is scored rather
    than refused, so every page returns a result.
    """
    from src.evaluation.evaluate import _prf
    from src.evaluation.ocr_metrics import error_counts, normalize
    from src.evaluation.ocr_v2 import match_boxes, summarize

    regions = []
    for region in sorted(page["regions"], key=lambda item: (item["order"], item["id"])):
        box = transform_box(region["bbox"], matrix, width, height)
        if box[2] > box[0] and box[3] > box[1]:
            regions.append({"id": region["id"], "bbox": box, "text": region["text"],
                            "type": region.get("type_confirmed") or region.get("type_suggestion") or "unclassified",
                            "held": bool(region["hold_reasons"]), "prediction": None, "iou": None, "errors": None, "cer": None})
    references = [region for region in regions if not region["held"]]
    held = [region for region in regions if region["held"]]
    predictions, ignored = [], 0
    for line in lines:
        box = _line_box(line.get("bbox") or [], width, height)
        if box is None:
            continue
        if any(_inside(box, r["bbox"]) >= 0.5 for r in held) and not any(_inside(box, r["bbox"]) >= 0.5 for r in references):
            ignored += 1
            continue
        predictions.append({"bbox": box, "text": str(line.get("text") or "")})
    matches = match_boxes([p["bbox"] for p in predictions], [r["bbox"] for r in references], threshold)
    matched_prediction = {j: (i, iou) for i, j, iou in matches}
    used = {i for i, _, _ in matches}
    by_type: dict[str, list] = {}
    for index, region in enumerate(references):
        if index in matched_prediction:
            i, iou = matched_prediction[index]
            counts = error_counts(normalize(region["text"]), normalize(predictions[i]["text"]))
            region.update(prediction=predictions[i]["text"], iou=iou, errors=counts["errors"],
                          cer=counts["errors"] / counts["reference_units"] if counts["reference_units"] else None)
            by_type.setdefault(region["type"], []).append((region["text"], predictions[i]["text"]))
        else:
            region.update(prediction=None, iou=None, errors=None, cer=None)
    matched_pairs = [(r["text"], r["prediction"]) for r in references if r["prediction"] is not None]
    return {
        "page": {"id": page["id"], "split": page["split"], "annotator": page["annotator"], "status": page["status"],
                 "eligible": bool(page.get("eligible", False)), "hold_reasons": page["hold_reasons"], "no_text": page["no_text"]},
        "image": {"width": width, "height": height},
        "iou_threshold": threshold,
        "detection": {**_prf(len(matches), len(predictions) - len(matches), len(references) - len(matches)),
                      "references": len(references), "predictions": len(predictions), "matched": len(matches),
                      "held_references": len(held), "ignored_predictions": ignored},
        # All predicted text in OCR order against all label text in label order: order and extra regions count.
        "page_text": summarize([("\n".join(r["text"] for r in references), "\n".join(p["text"] for p in predictions))]) if references else None,
        "matched_regions": summarize(matched_pairs),
        # A label region with no matching prediction counts as fully deleted.
        "all_regions": summarize([(r["text"], r["prediction"] or "") for r in references]),
        "by_type": {kind: summarize(pairs) for kind, pairs in sorted(by_type.items())},
        "regions": regions,
        "extra_predictions": [p for i, p in enumerate(predictions) if i not in used],
    }


def evaluate_session(page_id: str, document: dict, source_path: Path) -> dict:
    """Check the session really analysed this dataset page, then score its stored OCR lines."""
    import cv2
    import numpy as np
    from PIL import Image, ImageOps

    page = get_page(page_id)
    with Image.open(image_path(page_id)) as original:
        expected = np.asarray(original.convert("RGB"))
    with Image.open(source_path) as source:
        actual = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
    if expected.shape != actual.shape or not np.array_equal(expected, actual):
        raise ValueError("이 문서는 선택한 라벨 데이터셋 페이지의 원본 이미지로 분석한 것이 아닙니다.")
    if not isinstance(document.get("ocr"), dict):
        raise ValueError("이 문서에는 OCR 결과가 없습니다. OCR 모드를 선택해 분석해 주세요.")
    matrix, (width, height), steps = preprocess_geometry(cv2.cvtColor(expected, cv2.COLOR_RGB2BGR))
    if (width, height) != (document["image"]["width"], document["image"]["height"]):
        raise ValueError("분석 문서의 크기가 전처리 결과와 다릅니다.")
    result = evaluate_page(page, document["ocr"].get("lines", []), matrix.tolist(), width, height)
    data = _load()
    result.update(preprocess_steps=steps, ocr_engine=document["ocr"].get("engine"),
                  structure_mode=(document.get("structure") or {}).get("mode", "opencv"),
                  dataset=data["dataset"], revision=data["revision"], labels_commit=data["labels_commit"],
                  scope="text lines only; no reference for diagram, formula structure, or table output")
    return result
