"""Score diagram structure predictions, for a LoRA adapter or the untouched base model.

Elements are matched one-to-one: nodes by IoU with the same kind, arrows by both endpoints
(thin arrows make bbox IoU unreliable). A relation is correct when its source and target
resolve to the matched gold pair. Unparseable output counts as an empty prediction.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from .prepare import read_jsonl
from .schema import require
from .train import load_base_model

NODE_IOU = 0.5
ARROW_ENDPOINT_TOLERANCE = 0.03  # Fraction of the image diagonal.


def iou(a, b):
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    intersection = width * height
    return intersection / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection)


def valid_box(box):
    return isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box) and box[0] < box[2] and box[1] < box[3]


def valid_point(point):
    return isinstance(point, list) and len(point) == 2 and all(isinstance(v, (int, float)) for v in point)


def greedy_match(candidates):
    """Best-first one-to-one matching of (score, predicted id, gold id)."""
    used_pred, used_gold, mapping = set(), set(), {}
    for _, pred, gold in sorted(candidates, key=lambda item: -item[0]):
        if pred not in used_pred and gold not in used_gold:
            used_pred.add(pred)
            used_gold.add(gold)
            mapping[pred] = gold
    return mapping


def elements(items, check):
    """Keep usable objects with a unique id; hand-written models can emit anything."""
    kept, seen = [], set()
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"] not in seen and check(item):
            seen.add(item["id"])
            kept.append(item)
    return kept


def score_record(text, gold, width, height):
    """Return true-positive/predicted/gold counts per task, plus whether output parsed."""
    try:
        pred = json.loads(text)
        parsed = isinstance(pred, dict)
    except ValueError:
        pred, parsed = {}, False
    pred = pred if parsed else {}
    nodes = elements(pred.get("nodes"), lambda n: valid_box(n.get("bbox")) and isinstance(n.get("kind"), str))
    arrows = elements(pred.get("arrows"), lambda a: valid_point(a.get("start")) and valid_point(a.get("end")))
    node_map = greedy_match([(iou(p["bbox"], g["bbox"]), p["id"], g["id"]) for p in nodes for g in gold.get("nodes", [])
                             if p["kind"] == g["kind"] and iou(p["bbox"], g["bbox"]) >= NODE_IOU])
    tolerance = ARROW_ENDPOINT_TOLERANCE * math.hypot(width, height)
    candidates = []
    for p in arrows:
        for g in gold.get("arrows", []):
            if g["start"] is None or g["end"] is None:
                continue
            distance = max(math.dist(p["start"], g["start"]), math.dist(p["end"], g["end"]))
            if distance <= tolerance:
                candidates.append((-distance, p["id"], g["id"]))
    arrow_map = greedy_match(candidates)
    mapping = {**node_map, **arrow_map}
    gold_pairs = Counter((r["source"], r["target"]) for r in gold.get("relations", []))
    relations = [r for r in pred.get("relations", []) if isinstance(r, dict)] if isinstance(pred.get("relations"), list) else []
    pred_pairs = Counter((mapping.get(r.get("source")), mapping.get(r.get("target"))) for r in relations)
    correct = sum(min(count, gold_pairs[pair]) for pair, count in pred_pairs.items() if None not in pair)
    return {"parsed": parsed,
            "nodes": (len(node_map), len(nodes), len(gold.get("nodes", []))),
            "arrows": (len(arrow_map), len(arrows), len(gold.get("arrows", []))),
            "relations": (correct, sum(pred_pairs.values()), sum(gold_pairs.values()))}


def aggregate(scores):
    """Micro precision/recall/F1 per task."""
    result = {"samples": len(scores), "parsed": sum(s["parsed"] for s in scores)}
    for task in ("nodes", "arrows", "relations"):
        tp, predicted, gold = (sum(s[task][i] for s in scores) for i in range(3))
        precision = tp / predicted if predicted else 0.0
        recall = tp / gold if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[task] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "true_positive": tp, "predicted": predicted, "gold": gold}
    return result


def structure_bucket(row_id, manifests):
    """FC_A only: test results split by whether the graph structure appeared in training."""
    dataset, _, sample = row_id.partition(":")
    for name, info in manifests.get(dataset, {}).items():
        if Path(name).stem == sample and "structure_seen_in_train" in info:
            return "structure_seen" if info["structure_seen_in_train"] else "structure_unseen"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", help="LoRA adapter directory; omit to score the base model")
    parser.add_argument("--data", required=True, help="Prepared JSONL, for example data/processed/radu/test.jsonl")
    parser.add_argument("--output", required=True, help="New directory for predictions and metrics")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split-manifest", action="append", default=[])
    args = parser.parse_args()
    import torch
    from PIL import Image

    output = Path(args.output)
    require(not output.exists(), "output exists; use a new directory")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rows = list(read_jsonl(args.data))[:args.limit]
    manifests = {}
    for path in args.split_manifest:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        manifests[manifest["dataset"]] = manifest["samples"]
    processor, model = load_base_model(config)
    if not config.get("load_in_4bit", False):
        model.to("cuda")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    device = next(model.parameters()).device
    output.mkdir(parents=True)
    groups, scores = defaultdict(list), []
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            with Image.open(row["image"]) as image:
                image = image.convert("RGB")
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": row["prompt"]}]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            text = processor.tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            with Image.open(row["image"]) as image:
                width, height = image.size
            score = score_record(text, json.loads(row["completion"]), width, height)
            scores.append(score)
            groups[row["dataset"]].append(score)
            bucket = structure_bucket(row["id"], manifests)
            if bucket:
                groups[f"{row['dataset']}:{bucket}"].append(score)
            handle.write(json.dumps({"id": row["id"], "prediction": text, "score": score}, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{index}/{len(rows)} {row['id']} parsed={score['parsed']}", flush=True)
    metrics = {"adapter": args.adapter, "data": args.data, "overall": aggregate(scores), "by_group": {key: aggregate(value) for key, value in sorted(groups.items())}}
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics["overall"], ensure_ascii=False))


if __name__ == "__main__":
    main()
