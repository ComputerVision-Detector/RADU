"""Join diagram COCO and a versioned split manifest into image-only normalized.jsonl.

Supervises nodes, arrows and relations, or the subset a manifest sample names in "tasks".
Fails rather than writing a partial source.

Arrow keypoints: COCO metadata names them ['head', 'tail'], but the first point is the
arrow's tail and the second its tip. Checked against arrow_prev/arrow_next on all 11,829
FC_A and hdBPMN arrows (one tie, the rest agree) and by visual overlay. The same check on
FC_B: 4,568 of 4,569 agree. FA: 1,417 agree, none disagree, 431 self-loops tie.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .import_coco import convert
from .schema import require

ALL_PARTITIONS = ("train", "val", "test")
SOURCES = {"fc_a": ("source", ("train", "test")), "hdbpmn": ("coco_icdar2021", ALL_PARTITIONS),
           "fc_b": ("source", ALL_PARTITIONS), "fc_b_scan": ("source", ALL_PARTITIONS), "fa": ("source", ALL_PARTITIONS)}
TASKS = ["nodes", "arrows", "relations"]


def arrow_geometry(annotations):
    geometry = {}
    for annotation in annotations:
        # FA's validation arrows carry no connectivity, only visible keypoints.
        points = annotation.get("keypoints", [])
        if "arrow_prev" not in annotation and "arrow_next" not in annotation and not (len(points) >= 6 and (points[2] > 0 or points[5] > 0)):
            continue
        require(len(points) >= 6, "arrow requires two keypoints")
        geometry[str(annotation["id"])] = {
            "start": [round(points[0]), round(points[1])] if points[2] > 0 else None,
            "end": [round(points[3]), round(points[4])] if points[5] > 0 else None,
        }
    return geometry


def normalized_rows(document, partition, manifest, image_prefix, category_map, dataset, revision):
    """Yield records for the manifest samples in one COCO partition."""
    images = [im for im in document["images"] if im["file_name"] in manifest["samples"]]
    ids = {im["id"] for im in images}
    annotations = [a for a in document["annotations"] if a["image_id"] in ids]
    subset = {"images": images, "annotations": annotations, "categories": document.get("categories", [])}
    sidecars = []
    for image in images:
        name = image["file_name"]
        sample = manifest["samples"][name]
        own = [a for a in annotations if a["image_id"] == image["id"]]
        tasks = sample.get("tasks", TASKS)
        sidecars.append({"image_id": image["id"], "image": f"{image_prefix}/{partition}/{name}", "sample_id": Path(name).stem,
                         "split": sample["split"], "groups": sample["groups"], "tasks": tasks,
                         "coverage": {task: "complete" for task in tasks}, "arrow_geometry": arrow_geometry(own)})
    return convert(subset, sidecars, category_map, dataset, revision)


def build(dataset, root, manifest, category_map, revision):
    prefix, partitions = SOURCES[dataset]
    rows = []
    for partition in partitions:
        document = json.loads((Path(root) / dataset / prefix / f"{partition}.json").read_text(encoding="utf-8"))
        rows.extend(normalized_rows(document, partition, manifest, prefix, category_map, dataset, revision))
    missing = set(Path(name).stem for name in manifest["samples"]) - {row["id"] for row in rows}
    require(not missing, f"manifest samples absent from COCO: {sorted(missing)[:5]}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(SOURCES), required=True)
    parser.add_argument("--root", default="data/raw/radu")
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--category-map", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    out = Path(args.root) / args.dataset / "normalized.jsonl"
    require(not out.exists(), f"{out} exists; move it aside deliberately before rebuilding")
    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    require(manifest["dataset"] == args.dataset, "split manifest belongs to another dataset")
    category_map = json.loads(Path(args.category_map).read_text(encoding="utf-8"))
    rows = build(args.dataset, args.root, manifest, category_map, f"{args.revision}+{manifest['version']}")
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} normalized records to {out}")


if __name__ == "__main__":
    main()
