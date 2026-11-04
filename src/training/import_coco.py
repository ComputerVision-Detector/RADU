"""Import FC_A/hdBPMN diagram COCO plus an explicit split/coverage sidecar.

Category mapping and coverage are supplied after reviewing a source version.
No nearest-neighbor endpoint inference or auto split.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from .prepare import read_jsonl
from .schema import require, validate_record


def convert(coco, sidecars, category_map, dataset, revision):
    require(dataset in {"fc_a", "hdbpmn"}, "diagram COCO importer supports fc_a/hdbpmn only")
    categories = {c["id"]: c["name"] for c in coco.get("categories", [])}
    images = {im["id"]: im for im in coco["images"]}
    by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        require(annotation["image_id"] in images, "annotation references missing image")
        by_image[annotation["image_id"]].append(annotation)
    seen = set()
    for side in sidecars:
        image_id = side["image_id"]
        require(image_id in images and image_id not in seen, "unknown/duplicate sidecar image_id")
        seen.add(image_id)
        im = images[image_id]
        tasks = side["tasks"]
        require("nodes" in tasks, "COCO importer requires nodes supervision")
        target = {task: [] for task in tasks}
        labels = by_image[image_id]
        local_ids = {a["id"] for a in labels}
        require(len(local_ids) == len(labels), "duplicate annotation IDs")
        nodes, arrows = [], []
        for a in labels:
            name = a.get("category", categories.get(a.get("category_id")))
            require(name in category_map, f"unmapped category: {name}")
            kind = category_map[name]
            require(isinstance(kind, dict) and kind.get("type") in {"node", "arrow", "text", "ignore"}, "invalid category mapping")
            x, y, w, h = a["bbox"]
            # Integer pixels: enclosing box, no float noise spent on training tokens.
            box = [math.floor(x), math.floor(y), math.ceil(x + w), math.ceil(y + h)]
            if kind["type"] == "node":
                require(isinstance(kind.get("kind"), str), "node kind mapping required")
                nodes.append((box, a, kind["kind"]))
            elif kind["type"] == "arrow":
                arrows.append((box, a))
        # Source annotation IDs cannot be predicted from pixels, so supervising them adds
        # irreducible loss. Number elements per image in reading order instead.
        reading_order = lambda item: (item[0][1], item[0][0], item[0][3], item[0][2], item[1]["id"])
        nodes.sort(key=reading_order)
        arrows.sort(key=reading_order)
        element_ids = {a["id"]: f"n{i}" for i, (_, a, _) in enumerate(nodes, 1)}
        target["nodes"] = [{"id": f"n{i}", "bbox": box, "kind": kind} for i, (box, _, kind) in enumerate(nodes, 1)]
        if "arrows" in tasks:
            element_ids.update({a["id"]: f"a{i}" for i, (_, a) in enumerate(arrows, 1)})
            for i, (box, a) in enumerate(arrows, 1):
                # Endpoint order is source-specific; sidecar must explicitly supply it.
                geometry = side.get("arrow_geometry", {}).get(str(a["id"]), {})
                target["arrows"].append({"id": f"a{i}", "bbox": box, "start": geometry.get("start"), "end": geometry.get("end")})
        if "relations" in tasks:
            for i, (_, a) in enumerate(arrows, 1):
                require("arrow_prev" in a and "arrow_next" in a, "missing source connectivity; omit relations task")
                endpoints = []
                for field in ("arrow_prev", "arrow_next"):
                    endpoint = a[field]
                    require(endpoint is None or endpoint in element_ids, "connection references omitted/unknown annotation")
                    endpoints.append(None if endpoint is None else element_ids[endpoint])
                target["relations"].append({"id": f"r{i}", "source": endpoints[0], "target": endpoints[1], "directed": True,
                                            "arrow_id": f"a{i}" if "arrows" in tasks else None})
        # COCO train/val/test files restart image IDs at zero. Keep source IDs
        # distinct when their converted records are concatenated for prepare.
        sample_id = side.get("sample_id", f"{side.get('source_partition', side['split'])}:{image_id}")
        record = {"id":sample_id, "dataset":dataset, "image":side["image"], "width":im["width"], "height":im["height"],
                  "split":side["split"], "groups":side["groups"], "source_revision":revision,
                  "annotation_origin":"source_annotation", "tasks":tasks,
                  "coverage":side["coverage"], "target":target,
                  "source_ids":{new: f"ann-{old}" for old, new in element_ids.items()}}
        yield validate_record(record)
    require(seen == set(images), "sidecar must account for every image in this COCO file; subset COCO explicitly for a pilot")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--category-map", required=True)
    parser.add_argument("--dataset", choices=["fc_a", "hdbpmn"], required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    require(not out.exists(), "output exists; use a new filename")
    coco = json.loads(Path(args.coco).read_text(encoding="utf-8"))
    mapping = json.loads(Path(args.category_map).read_text(encoding="utf-8"))
    rows = list(convert(coco, read_jsonl(args.sidecar), mapping, args.dataset, args.revision))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} normalized records; image paths are relative to {out.parent}")


if __name__ == "__main__":
    main()
