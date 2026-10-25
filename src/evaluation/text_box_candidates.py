"""Generate non-destructive OCR line-box proposals for the RADU labeler."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2

from src.cv.ocr_engine import recognize


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_DIR = ROOT / "labeling" / "assets"
DEFAULT_OUTPUT = ROOT / "labeling" / "data" / "text-box-proposals.json"
SUPPORTED_ENGINES = ("paddleocrv6", "paddleocrvl16")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _xyxy(polygon: Any, image_width: int, image_height: int,
          source_width: int, source_height: int) -> list[int] | None:
    """Convert a polygon to clipped source-image pixel coordinates."""
    try:
        points = [float(value) for point in polygon for value in point]
    except (TypeError, ValueError):
        try:
            values = [float(value) for value in polygon]
        except (TypeError, ValueError):
            return None
        if len(values) != 4:
            return None
        x1, y1, x2, y2 = values
    else:
        if len(points) < 6 or len(points) % 2:
            return None
        xs, ys = points[0::2], points[1::2]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    scale_x, scale_y = source_width / image_width, source_height / image_height
    left = max(0, min(source_width, math.floor(x1 * scale_x)))
    top = max(0, min(source_height, math.floor(y1 * scale_y)))
    right = max(0, min(source_width, math.ceil(x2 * scale_x)))
    bottom = max(0, min(source_height, math.ceil(y2 * scale_y)))
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def _confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _load_catalog() -> tuple[dict[str, Any], dict[str, Any]]:
    seed = json.loads((ROOT / "labeling" / "data" / "seed.json").read_text(encoding="utf-8"))
    assets = json.loads((ROOT / "labeling" / "data" / "assets.manifest.json").read_text(encoding="utf-8"))
    asset_by_id = {asset["id"]: asset for asset in assets["images"]}
    missing = set(seed["items"]) - set(asset_by_id)
    if missing:
        raise ValueError(f"Asset manifest is missing image IDs: {', '.join(sorted(missing))}")
    return seed["items"], asset_by_id


def generate(image_dir: Path, engine: str, output: Path, device: str | None = None) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Proposal output already exists; choose a new --output path: {output}")
    items, assets = _load_catalog()
    images: dict[str, Any] = {}
    model_metadata: dict[str, Any] | None = None
    proposal_version = engine

    for image_id, item in items.items():
        image_file = item["image"]["file"]
        asset = assets[image_id]
        input_path = image_dir / Path(image_file).name
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing labeling image: {input_path}")
        input_sha256 = _sha256(input_path)
        if input_sha256 != asset.get("sha256"):
            raise ValueError(f"Input image fingerprint does not match the labeling asset: {input_path}")
        image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to decode labeling image: {input_path}")
        image_height, image_width = image.shape[:2]
        source_width = int(item["image"]["source_width"])
        source_height = int(item["image"]["source_height"])

        _text, details = recognize(image, engine, device=device)
        details = details if isinstance(details, dict) else {}
        model_name = str(details.get("model") or engine)
        current_metadata = {
            "name": model_name,
            "device": details.get("device") or device or "configured-by-engine",
            "metadata": details.get("metadata") or {},
        }
        if model_metadata is None:
            model_metadata = current_metadata
            proposal_version = f"{engine}-{model_name.lower().replace(' ', '-') }".rstrip("-")

        proposals = []
        skipped = 0
        for order, line in enumerate(details.get("lines", []), start=1):
            if not isinstance(line, dict):
                skipped += 1
                continue
            box = _xyxy(line.get("bbox", []), image_width, image_height, source_width, source_height)
            if box is None:
                skipped += 1
                continue
            score = _confidence(line.get("confidence"))
            proposal = {
                "id": f"{image_id}-line-{order:04d}",
                "bbox": box,
                "prediction": str(line.get("text") or ""),
                "confidence": score,
                "order": len(proposals) + 1,
            }
            proposals.append(proposal)

        record = {
            "file": image_file,
            "source_sha256": asset.get("source_sha256"),
            "source_size": {"width": source_width, "height": source_height},
            "asset_sha256": asset.get("sha256"),
            "asset_size": {"width": asset.get("width"), "height": asset.get("height")},
            "input_sha256": input_sha256,
            "input_size": {"width": image_width, "height": image_height},
            "proposal_version": proposal_version,
            "lines": proposals,
        }
        if skipped:
            record["skipped_lines"] = skipped
        images[image_id] = record
        print(f"{image_id}: {len(proposals)} line proposals", flush=True)

    manifest = {
        "schema_version": 1,
        "dataset_version": "v2-proposals",
        "proposal_version": proposal_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine": engine,
        "model": model_metadata or {"name": engine, "device": device or "configured-by-engine", "metadata": {}},
        "images": images,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8", newline="\n") as destination:
            json.dump(manifest, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
    except FileExistsError as error:
        raise FileExistsError(f"Proposal output already exists; choose a new --output path: {output}") from error
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR,
                        help="directory containing the catalog's web labeling images")
    parser.add_argument("--engine", choices=SUPPORTED_ENGINES, default="paddleocrv6")
    parser.add_argument("--device", help="Paddle device passed to the OCR adapter, such as cpu or gpu:0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="new proposal manifest path; existing files are never overwritten")
    args = parser.parse_args()
    try:
        manifest = generate(args.image_dir, args.engine, args.output, args.device)
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Wrote {len(manifest['images'])} image proposals to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
