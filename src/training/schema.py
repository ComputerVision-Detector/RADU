"""Strict normalized contract. Missing annotations are not negative labels.

Image-only structure extraction: OCR is not a training input. Text recognition runs
separately and is merged with this structure after inference.
"""
from __future__ import annotations

import math

DATASETS = {"hdbpmn", "fc_a", "fc_b", "fc_b_scan", "fa", "ai2d_rst", "didi", "casia_ohfc", "flowlearn",
            "crohme_2023", "unimer", "mathwriting", "pubtabnet", "fintabnet_c",
            "hdflowmind", "hme100k", "ibem", "tabrecset", "ctdar_2019", "read_abp"}
TASKS = {"nodes", "arrows", "relations", "formulas", "tables"}
TASK_INSTRUCTIONS = {
    "formulas": "formulas: [{id,bbox,latex}]; here, and only here, write the formula itself as LaTeX.\n",
    "tables": "tables: [{id,bbox,rows,cols,cells}]; each cell is [row,col,rowspan,colspan] counted from 0, followed by x0,y0,x1,y1 when its box is known. Never write cell text.\n",
}
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


def validate_formula(item, width, height):
    require(isinstance(item, dict) and set(item) == {"id", "bbox", "latex"}, "invalid formula fields")
    bbox(item["bbox"], width, height)
    require(isinstance(item["latex"], str) and bool(item["latex"].strip()), "formula latex required")


def validate_table(item, width, height):
    require(isinstance(item, dict) and set(item) == {"id", "bbox", "rows", "cols", "cells"}, "invalid table fields (cell text is not a target)")
    bbox(item["bbox"], width, height)
    rows, cols = item["rows"], item["cols"]
    require(type(rows) is int and type(cols) is int and rows > 0 and cols > 0, "invalid table size")
    require(isinstance(item["cells"], list) and bool(item["cells"]), "table cells required")
    occupied = set()
    for cell in item["cells"]:
        require(isinstance(cell, list) and len(cell) in (4, 8) and all(type(v) is int for v in cell), "cell is [row,col,rowspan,colspan] plus optional box")
        row, col, rowspan, colspan = cell[:4]
        require(0 <= row and 0 <= col and rowspan > 0 and colspan > 0 and row + rowspan <= rows and col + colspan <= cols, "cell outside table grid")
        spanned = {(r, c) for r in range(row, row + rowspan) for c in range(col, col + colspan)}
        require(not spanned & occupied, "overlapping table cells")
        occupied |= spanned
        if len(cell) == 8:
            bbox(cell[4:], width, height)


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
    for task, check in (("formulas", validate_formula), ("tables", validate_table)):
        for item in target.get(task, []):
            check(item, w, h)
            require(isinstance(item["id"], str) and item["id"] and item["id"] not in ids, "duplicate/invalid element id")
            ids.add(item["id"])
    relation_ids = set()
    for item in target.get("relations", []):
        require(isinstance(item, dict) and set(item) == {"id", "source", "target", "directed", "arrow_id"}, "invalid relation fields")
        require(isinstance(item["id"], str) and item["id"] and item["id"] not in relation_ids, "duplicate relation id")
        relation_ids.add(item["id"])
        require(type(item["directed"]) is bool, "directed must be boolean")
        require(item["arrow_id"] is None or item["arrow_id"] in {a["id"] for a in target.get("arrows", [])}, "invalid arrow reference")
        require(all(item[k] is None or item[k] in ids for k in ("source", "target")), "dangling relation reference; include referenced elements in target")
    return record
