"""Strict normalized contract. Missing annotations are not negative labels.

Image-only structure extraction: OCR is not a training input. Text recognition runs
separately and is merged with this structure after inference.
"""
from __future__ import annotations

import math

DATASETS = {"hdbpmn", "fc_a", "ai2d_rst", "didi", "casia_ohfc", "flowlearn"}
TASKS = {"nodes", "arrows", "relations"}
PROMPT = """Extract only the requested diagram structure from the image.
Do not transcribe, correct, translate, summarize, solve formulas, or infer invisible
connections. Coordinates are original-image pixels.
Return a JSON object with exactly the requested keys. nodes: [{id,bbox,kind}];
arrows: [{id,bbox,start,end}] (start/end may be null when the head is unresolved);
relations: [{id,source,target,directed,arrow_id}] (arrow_id may be null;
source/target reference a node or arrow). Use null for unresolved relation endpoints.
Requested keys: {tasks}
Original image: {width} x {height}
"""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def bbox(box, width, height):
    require(isinstance(box, list) and len(box) == 4 and all(finite(v) for v in box), "invalid bbox")
    x1, y1, x2, y2 = box
    require(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, "bbox outside original image")


def validate_record(record):
    require(isinstance(record, dict), "record must be an object")
    require(record.get("dataset") in DATASETS, "unknown dataset")
    for field in ("id", "image", "source_revision", "annotation_origin"):
        require(isinstance(record.get(field), str) and bool(record[field].strip()), f"missing {field}")
    require(record.get("annotation_origin") in {"human", "source_annotation", "synthetic_program"}, "pseudo labels require human validation")
    require(record.get("split") in {"train", "validation", "test"}, "explicit split required")
    groups = record.get("groups")
    require(isinstance(groups, list) and groups and all(isinstance(g, str) and g.strip() for g in groups), "writer/template/source groups required")
    for field in ("width", "height"):
        require(type(record.get(field)) is int and record[field] > 0, f"invalid {field}")
    w, h = record["width"], record["height"]
    tasks = record.get("tasks")
    require(isinstance(tasks, list) and tasks and len(set(tasks)) == len(tasks) and set(tasks) <= TASKS, "invalid tasks")
    target = record.get("target")
    require(isinstance(target, dict) and set(target) == set(tasks), "only explicitly supervised fields belong in target")
    require(all(isinstance(v, list) for v in target.values()), "target fields must be lists")
    coverage = record.get("coverage")
    require(isinstance(coverage, dict) and set(coverage) == set(tasks) and all(v == "complete" for v in coverage.values()), "each requested task requires complete annotation")
    ids = set()
    for task in ("nodes", "arrows"):
        for item in target.get(task, []):
            require(isinstance(item, dict), "element must be an object")
            require(isinstance(item.get("id"), str) and item["id"] and item["id"] not in ids, "duplicate/invalid element id")
            ids.add(item["id"])
            bbox(item.get("bbox"), w, h)
            expected = {"id", "bbox", "kind"} if task == "nodes" else {"id", "bbox", "start", "end"}
            require(set(item) == expected, "unexpected element fields (text is not a target)")
            if task == "nodes":
                require(isinstance(item.get("kind"), str) and bool(item["kind"].strip()), "node kind required")
            else:
                for key in ("start", "end"):
                    point = item[key]
                    require(point is None or isinstance(point, list) and len(point) == 2 and all(finite(v) for v in point) and 0 <= point[0] <= w and 0 <= point[1] <= h, "invalid arrow endpoint")
    relation_ids = set()
    for item in target.get("relations", []):
        require(isinstance(item, dict) and set(item) == {"id", "source", "target", "directed", "arrow_id"}, "invalid relation fields")
        require(isinstance(item["id"], str) and item["id"] and item["id"] not in relation_ids, "duplicate relation id")
        relation_ids.add(item["id"])
        require(type(item["directed"]) is bool, "directed must be boolean")
        require(item["arrow_id"] is None or item["arrow_id"] in {a["id"] for a in target.get("arrows", [])}, "invalid arrow reference")
        require(all(item[k] is None or item[k] in ids for k in ("source", "target")), "dangling relation reference; include referenced elements in target")
    return record
