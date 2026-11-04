"""Run PaddleOCR-VL 1.6 over split-manifest images; append resumable provenance JSONL.

For the post-inference merge and its evaluation; OCR is not a training input. Uses the
team's spotting path (src.evaluation.paddleocr_vl_worker) unchanged, so results match the
service pipeline. Failures are recorded and never become empty OCR;
an image with no detected text stays an explicit empty result from a successful run.
Run with the isolated Paddle environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

SOURCES = {"fc_a": ("source", ("train", "test")), "hdbpmn": ("coco_icdar2021", ("train", "val", "test"))}


def manifest_images(root, manifest):
    """Resolve manifest file names to paths relative to root, inside official partitions."""
    prefix, partitions = SOURCES[manifest["dataset"]]
    relatives = []
    for name in sorted(manifest["samples"]):
        found = [p for p in partitions if (Path(root) / manifest["dataset"] / prefix / p / name).is_file()]
        if len(found) != 1:
            raise ValueError(f"expected one source image for {name}, found {found}")
        relatives.append(f"{manifest['dataset']}/{prefix}/{found[0]}/{name}")
    return relatives


def to_lines(texts, polygons, width, height, to_box):
    """Build schema OCR lines; count dropped lines instead of hiding them."""
    lines, dropped = [], 0
    for index, text in enumerate(texts):
        box = to_box(polygons[index], width, height, width, height) if index < len(polygons) else None
        if box is None or not str(text).strip():
            dropped += 1
            continue
        lines.append({"id": f"line{len(lines) + 1}", "bbox": box, "text": str(text)})
    return lines, dropped


def latest_status(path):
    status = {}
    if Path(path).is_file():
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                status[row["image"]] = row["status"]
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", action="append", required=True)
    parser.add_argument("--root", default="data/raw/radu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--revisions", default="models/ocr_revisions.json")
    parser.add_argument("--sample", type=int, help="Seeded per-manifest sample for a pilot; results stay valid for the full run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    relatives = []
    for manifest_path in args.split_manifest:
        images = manifest_images(args.root, json.loads(Path(manifest_path).read_text(encoding="utf-8")))
        relatives += random.Random(args.seed).sample(images, args.sample) if args.sample else images
    status = latest_status(args.output)
    pending = [relative for relative in relatives if status.get(relative) != "ok"]
    print(f"{len(relatives)} requested, {len(pending)} pending", flush=True)
    if not pending:
        return

    from PIL import Image
    from src.evaluation.paddleocr_vl_worker import prepare_model, run_one
    from src.evaluation.text_box_candidates import _xyxy

    revision = json.loads(Path(args.revisions).read_text(encoding="utf-8"))["PaddlePaddle/PaddleOCR-VL-1.6"]["revision"]
    provenance = {"engine": "paddleocr-vl-spotting", "model": "PaddleOCR-VL-1.6", "revision": revision}
    model = prepare_model(args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for index, relative in enumerate(pending, 1):
            path = Path(args.root) / relative
            row = {"image": relative, "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **provenance}
            started = time.perf_counter()
            try:
                with Image.open(path) as image:
                    width, height = image.size
                _, raw, _ = run_one(model, path, args.device, 0.0)
                spotting = raw["spotting_res"]
                row["lines"], row["dropped_lines"] = to_lines(spotting.get("rec_texts", []), spotting.get("rec_polys", []), width, height, _xyxy)
                row["status"] = "ok"
            except Exception as error:
                row.update(status="error", error=f"{type(error).__name__}: {error}")
            row["seconds"] = round(time.perf_counter() - started, 3)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{index}/{len(pending)} {relative} {row['status']} lines={len(row.get('lines', []))} {row['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
