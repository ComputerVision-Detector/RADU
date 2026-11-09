"""COCO-style box AP from saved evaluate.py predictions, without running the model again.

The model writes boxes as text and gives no confidence, so every predicted box has the same score.
The precision-recall curve is then a single point (recall R, precision P), and COCO's 101-point
interpolated AP reduces to P times the share of recall thresholds at or below R. AP50 therefore
says little beyond precision and recall; the useful part is how AP falls as the IoU threshold rises.

Categories follow COCO: each node kind is its own category, plus arrow, table, table cell and formula.
A category counts for a source only if that source's ground truth has boxes for it (PubTabNet cells
have no boxes). Output that is not valid JSON has no boxes, as in evaluate.py.

  python -m src.training.box_ap --data data/processed/radu_final/validation.jsonl \
      --eval models/radu/eval_final/ckpt-6000 --eval models/radu/eval_final/ckpt-18000
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .evaluate import iou, valid_box
from .prepare import read_jsonl

THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]
RECALL_POINTS = [i / 100 for i in range(101)]


def boxes_by_category(answer):
    """Map category name to the list of boxes an answer gives for it."""
    found = defaultdict(list)
    if not isinstance(answer, dict):
        return found
    listed = lambda key: [item for item in answer.get(key, []) if isinstance(item, dict)] if isinstance(answer.get(key), list) else []
    for node in listed("nodes"):
        if valid_box(node.get("bbox")) and isinstance(node.get("kind"), str):
            found[f"node:{node['kind']}"].append(node["bbox"])
    for arrow in listed("arrows"):
        if valid_box(arrow.get("bbox")):
            found["arrow"].append(arrow["bbox"])
    for formula in listed("formulas"):
        if valid_box(formula.get("bbox")):
            found["formula"].append(formula["bbox"])
    for table in listed("tables"):
        if valid_box(table.get("bbox")):
            found["table"].append(table["bbox"])
        for cell in table.get("cells", []) if isinstance(table.get("cells"), list) else []:
            if isinstance(cell, list) and len(cell) == 8 and valid_box(cell[4:8]):
                found["cell"].append(cell[4:8])
    return found


def matches(predicted, gold, threshold):
    """Greedy one-to-one matching by highest IoU; returns the number of matched pairs."""
    pairs = sorted(((iou(p, g), i, j) for i, p in enumerate(predicted) for j, g in enumerate(gold)), reverse=True)
    used_p, used_g, count = set(), set(), 0
    for value, i, j in pairs:
        if value < threshold:
            break
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            count += 1
    return count


def single_point_ap(tp, predicted, gold):
    if not gold:
        return None
    recall = tp / gold
    precision = tp / predicted if predicted else 0.0
    return precision * sum(point <= recall + 1e-12 for point in RECALL_POINTS) / len(RECALL_POINTS)


def evaluate_run(rows, eval_dir):
    """Return {source: {category: {threshold: (tp, predicted, gold)}}}."""
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0, 0])))
    for line in (Path(eval_dir) / "predictions.jsonl").open(encoding="utf-8"):
        record = json.loads(line)
        row = rows[record["id"]]
        try:
            answer = json.loads(record["prediction"])
        except ValueError:
            answer = {}
        predicted, gold = boxes_by_category(answer), boxes_by_category(json.loads(row["completion"]))
        for category in set(predicted) | set(gold):
            for threshold in THRESHOLDS:
                cell = counts[row["dataset"]][category][threshold]
                cell[0] += matches(predicted[category], gold[category], threshold)
                cell[1] += len(predicted[category])
                cell[2] += len(gold[category])
    return counts


def summarize(counts):
    """Per source: AP50, AP75 and mAP@[.50:.95] for each category group and over all categories."""
    group_of = lambda category: "nodes" if category.startswith("node:") else category
    result = {}
    for source, categories in sorted(counts.items()):
        per_category = {}
        for category, by_threshold in categories.items():
            aps = {t: single_point_ap(*by_threshold[t]) for t in THRESHOLDS}
            if aps[THRESHOLDS[0]] is not None:  # no ground truth boxes: not a category of this source
                per_category[category] = aps
        groups = defaultdict(list)
        for category, aps in per_category.items():
            groups[group_of(category)].append(aps)
            groups["all"].append(aps)
        result[source] = {group: {"AP50": round(sum(a[0.5] for a in items) / len(items), 4),
                                  "AP75": round(sum(a[0.75] for a in items) / len(items), 4),
                                  "mAP": round(sum(sum(a.values()) / len(a) for a in items) / len(items), 4),
                                  "categories": len(items)}
                          for group, items in groups.items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--eval", action="append", required=True, help="evaluate.py output directory; repeatable")
    args = parser.parse_args()
    rows = {row["id"]: row for row in read_jsonl(args.data)}
    for eval_dir in args.eval:
        result = summarize(evaluate_run(rows, eval_dir))
        (Path(eval_dir) / "box_ap.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{eval_dir}/box_ap.json")


if __name__ == "__main__":
    main()
