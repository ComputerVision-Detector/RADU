"""Draw ground truth and prediction side by side as one PNG per evaluated image.

Reads an evaluate.py output directory and works while predictions are still being written,
so it can be rerun to pick up new images; images already drawn are skipped.

  python -m src.training.render_eval --data data/processed/radu_final/validation.jsonl \
      --eval models/radu/eval_final/ckpt-4000
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .infer_image import draw_structure, formula_table_boxes, parsed_object, side_by_side
from .prepare import read_jsonl
from .visualize_predictions import salvage_elements

MAX_SIDE = 2400
TEXT_LINES = 4


def f1_text(score, task):
    tp, predicted, gold = score.get(task, (0, 0, 0))
    return f"{task} {tp}/{predicted}/{gold}" if predicted or gold else ""


def latex_band(panel, label, formulas):
    """Append the first formulas' LaTeX under a panel, since formula scores compare text, not boxes."""
    from PIL import Image, ImageDraw, ImageFont

    texts = [item.get("latex", "") for item in formulas if isinstance(item, dict)][:TEXT_LINES]
    font = ImageFont.load_default(size=16)
    width = max(1, panel.width // 9)
    lines = [label] + [text[:width] + ("…" if len(text) > width else "") for text in texts]
    band = Image.new("RGB", (panel.width, 22 * len(lines) + 10), "white")
    draw = ImageDraw.Draw(band)
    for index, line in enumerate(lines):
        draw.text((8, 5 + 22 * index), line, fill="black", font=font)
    combined = Image.new("RGB", (panel.width, panel.height + band.height), "white")
    combined.paste(panel, (0, 0))
    combined.paste(band, (0, panel.height))
    return combined


def render(row, prediction, path):
    from PIL import Image

    with Image.open(row["image"]) as source:
        image = source.convert("RGB")
    gold = json.loads(row["completion"])
    text = prediction["prediction"]
    nodes, arrows, _, truncated = salvage_elements(text)
    answer = parsed_object(text)
    score = prediction["score"]
    counts = " ".join(filter(None, (f1_text(score, task) for task in ("nodes", "arrows", "relations", "formulas", "tables"))))
    gold_panel = draw_structure(image, gold.get("nodes", []), gold.get("arrows", []), "Ground truth", formula_table_boxes(gold))
    title = "Prediction" + (" (invalid JSON, partial)" if truncated else "") + f"  correct/pred/gold: {counts}"
    pred_panel = draw_structure(image, nodes, arrows, title, formula_table_boxes(answer))
    if gold.get("formulas"):
        gold_panel = latex_band(gold_panel, "gold LaTeX:", gold["formulas"])
        pred_panel = latex_band(pred_panel, "predicted LaTeX:", answer.get("formulas", []) if isinstance(answer.get("formulas"), list) else [])
    canvas = side_by_side([gold_panel, pred_panel])
    if max(canvas.size) > MAX_SIDE:
        scale = MAX_SIDE / max(canvas.size)
        canvas = canvas.resize((round(canvas.width * scale), round(canvas.height * scale)))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Prepared JSONL the evaluation was run on")
    parser.add_argument("--eval", required=True, help="evaluate.py output directory")
    args = parser.parse_args()
    rows = {row["id"]: row for row in read_jsonl(args.data)}
    output = Path(args.eval) / "images"
    drawn = 0
    with (Path(args.eval) / "predictions.jsonl").open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            try:
                prediction = json.loads(line)
            except ValueError:
                break  # last line still being written
            row = rows[prediction["id"]]
            path = output / row["dataset"] / f"{index:03d}_{re.sub(r'[^A-Za-z0-9._-]+', '_', prediction['id'])[:120]}.png"
            if not path.exists():
                render(row, prediction, path)
                drawn += 1
    print(f"{drawn} new images in {output}")


if __name__ == "__main__":
    main()
