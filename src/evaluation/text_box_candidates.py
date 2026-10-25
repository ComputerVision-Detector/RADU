"""Generate non-destructive OCR line-box proposals for the RADU labeler."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import time
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


def _vl_batches(paths: list[Path], device: str, batch_size: int):
    """Load VL once; keep original spotting polygons and raw output for audit."""
    from src.evaluation.paddleocr_vl_worker import prepare_model, _unwrap
    from paddlex.inference.pipelines.paddleocr_vl.uilts import post_process_for_spotting
    from PIL import Image

    model = prepare_model(device)
    import paddle, paddlex
    weights_sha256 = _sha256(ROOT / "models/vision_language/PaddleOCR-VL-1.6/model.safetensors")
    try:
        for offset in range(0, len(paths), batch_size):
            batch = paths[offset:offset + batch_size]
            started = time.perf_counter()
            results = list(model.predict(
                [{"image": str(path), "query": "Spotting:"} for path in batch],
                batch_size=batch_size, min_pixels=112_896, max_pixels=602_112,
                max_new_tokens=2048, use_cache=True, skip_special_tokens=False,
            ))
            if len(results) != len(batch):
                raise RuntimeError("VL returned a different number of images than requested")
            for path, result in zip(batch, results):
                raw = _unwrap(result)["result"]
                with Image.open(path) as image:
                    width, height = image.size
                _text, spotting = post_process_for_spotting(raw, width, height)
                texts, polygons = spotting.get("rec_texts", []), spotting.get("rec_polys", [])
                yield {
                    "model": "PaddleOCR-VL-1.6", "device": device,
                    "lines": [{"text": text, "bbox": polygons[i] if i < len(polygons) else []}
                              for i, text in enumerate(texts)],
                    "metadata": {"mode": "direct spotting", "batch_size": batch_size,
                                 "max_pixels": 602_112, "max_new_tokens": 2048,
                                 "weights_sha256": weights_sha256, "paddle": paddle.__version__, "paddlex": paddlex.__version__},
                    "batch_seconds": round(time.perf_counter() - started, 3),
                }
    finally:
        close = getattr(model, "close", None)
        if callable(close):
            close()


def generate(image_dir: Path, engine: str, output: Path, device: str | None = None,
             vl_batch_size: int = 1, resume: bool = False) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Proposal output already exists; choose a new --output path: {output}")
    items, assets = _load_catalog()
    images: dict[str, Any] = {}
    model_metadata: dict[str, Any] | None = None
    proposal_version = engine
    checkpoint = output.with_suffix('.checkpoint.json')
    if resume and checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding='utf-8'))
        if saved.get('engine') != engine:
            raise ValueError('Checkpoint engine differs from requested engine')
        if not isinstance(saved.get('images'), dict) or set(saved['images']) - set(items):
            raise ValueError('Checkpoint contains unknown image IDs')
        images = saved['images']
        model_metadata = saved.get('model')
        proposal_version = saved['proposal_version']
    # Verify every input before GPU work, including resumed entries.
    for image_id, item in items.items():
        path = image_dir / Path(item['image']['file']).name
        if not path.is_file() or _sha256(path) != assets[image_id]['sha256']:
            raise ValueError(f'Input image fingerprint does not match the labeling asset: {path}')
        if image_id in images and (images[image_id]['input_sha256'] != assets[image_id]['sha256']
                                  or images[image_id]['source_sha256'] != assets[image_id]['source_sha256']):
            raise ValueError(f'Checkpoint image fingerprint differs: {image_id}')
    pending = {key: value for key, value in items.items() if key not in images}
    batched = iter(_vl_batches([image_dir / Path(item['image']['file']).name for item in pending.values()],
                              device or 'gpu:0', vl_batch_size)) if engine == 'paddleocrvl16' and pending else None

    for image_id, item in pending.items():
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

        if batched is not None:
            details = next(batched)
        else:
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
        if not proposals:
            record["warnings"] = ["No text detected: manually inspect for missed text; this is not a no_text label."]
        if skipped:
            record["skipped_lines"] = skipped
        images[image_id] = record
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_suffix('.tmp')
        temporary.write_text(json.dumps({'engine': engine, 'images': images, 'model': model_metadata,
                                        'proposal_version': proposal_version}, ensure_ascii=False), encoding='utf-8')
        os.replace(temporary, checkpoint)
        print(f"[{len(images)}/{len(items)}] {image_id}: {len(proposals)} line proposals", flush=True)

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
    parser.add_argument("--engine", choices=SUPPORTED_ENGINES, default="paddleocrvl16")
    parser.add_argument("--device", help="Paddle device passed to the OCR adapter, such as cpu or gpu:0")
    parser.add_argument('--vl-batch-size', type=int, choices=(1,), default=1)
    parser.add_argument('--resume', action='store_true', help='resume a matching incomplete checkpoint')
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="new proposal manifest path; existing files are never overwritten")
    args = parser.parse_args()
    try:
        manifest = generate(args.image_dir, args.engine, args.output, args.device, args.vl_batch_size, args.resume)
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as error:
        parser.error(f'{type(error).__name__}: {error}')
    print(f"Wrote {len(manifest['images'])} image proposals to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
